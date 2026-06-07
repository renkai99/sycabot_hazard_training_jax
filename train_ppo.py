"""Unconstrained PPO for SycaBot Hazard Training – pure JAX / GPU parallel."""

import argparse
import csv
import glob
import os
import pickle
from datetime import datetime

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np  # noqa: F401  (kept for orthogonal/constant init helpers)
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import serialization
from typing import Sequence, NamedTuple, Any
from tqdm import tqdm
import wandb

from sycabot_env_jax import SycaBotEnvJAX, EnvParams, NUM_ROBOTS, NUM_TASKS
from wrappers import LogWrapper, VecEnv

# ========================================================================== #
#  CONFIG                                                                     #
# ========================================================================== #

config = {
    # PPO
    "LR": 3e-4,
    "LR_DECAY_START_FRAC": 0.8,
    "LR_END_FACTOR": 0.1,
    "UPDATE_EPOCHS": 4,
    "MINIBATCH_SIZE": 32768,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF":0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,   
    "CLIP_VF": True,
    "KL_THRESHOLD": 0.015,
    # Rollout

    "NUM_ENVS": 1024, # of environments to run in parallel on the GPU; adjust based on GPU memory and env complexity
    "NUM_STEPS": 64, # minibatch
    "TOTAL_UPDATES": 10000, # total PPO updates (not env steps; total env steps = NUM_ENVS * NUM_STEPS * TOTAL_UPDATES)

    # Network
    "ACTIVATION": "tanh",
    "HIDDEN_SIZE": 256,
    # Misc
    "SEED": 42,
    "PRINT_INTERVAL": 1000,
    "CHECKPOINT_INTERVAL": 10000,   # save params every N updates; 0 = disabled
}

config["NUM_MINIBATCHES"] = (config["NUM_ENVS"] * config["NUM_STEPS"]) // config["MINIBATCH_SIZE"]
assert (config["NUM_ENVS"] * config["NUM_STEPS"]) % config["MINIBATCH_SIZE"] == 0


# ========================================================================== #
#  DIAGONAL GAUSSIAN DISTRIBUTION (pure JAX – no numpyro)                   #
# ========================================================================== #

class DiagGaussian:
    """Diagonal-covariance Gaussian matching the numpyro .to_event(1) API."""

    def __init__(self, mean, std):
        self._mean = mean
        self._std  = std

    def sample(self, key):
        return self._mean + self._std * jax.random.normal(key, self._mean.shape)

    def log_prob(self, x):
        return jnp.sum(
            -0.5 * ((x - self._mean) / self._std) ** 2
            - jnp.log(self._std)
            - 0.5 * jnp.log(2.0 * jnp.pi),
            axis=-1,
        )

    def entropy(self):
        return jnp.sum(jnp.log(self._std) + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)), axis=-1)

    @property
    def mean(self):
        return self._mean

    @property
    def base_dist(self):
        return self   # .base_dist.scale compatibility

    @property
    def scale(self):
        return self._std


# ========================================================================== #
#  NETWORK                                                                    #
# ========================================================================== #

class ActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 256
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        act = nn.tanh  # extend here if needed

        actor = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor = act(actor)
        actor = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = act(actor)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = DiagGaussian(actor_mean, jnp.exp(log_std))

        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = act(critic)
        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = act(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


# ========================================================================== #
#  DATA STRUCTURES                                                            #
# ========================================================================== #

class Transition(NamedTuple):
    done:         jnp.ndarray
    action:       jnp.ndarray
    value:        jnp.ndarray
    reward:       jnp.ndarray
    log_prob:     jnp.ndarray
    obs:          jnp.ndarray
    info:         dict
    action_mean:  jnp.ndarray
    action_std:   jnp.ndarray


# ========================================================================== #
#  TRAINING FUNCTION                                                          #
# ========================================================================== #

def make_train(config, init_params=None, save_dir=None):
    env_base   = SycaBotEnvJAX()
    env_params = EnvParams()
    env        = VecEnv(LogWrapper(env_base))

    obs_dim    = env_base.observation_space(env_params).shape[0]
    action_dim = env_base.action_space(env_params).shape[0]

    total_updates   = config["TOTAL_UPDATES"]
    decay_start     = int(total_updates * config["LR_DECAY_START_FRAC"])

    def train(rng):
        network = ActorCritic(action_dim, hidden_size=config["HIDDEN_SIZE"],
                              activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        dummy_obs     = jnp.zeros(obs_dim)
        network_params = network.init(_rng, dummy_obs)["params"]
        if init_params is not None:
            network_params = init_params

        lr_schedule = optax.join_schedules(
            schedules=[
                optax.constant_schedule(config["LR"]),
                optax.linear_schedule(
                    init_value=config["LR"],
                    end_value=config["LR"] * config["LR_END_FACTOR"],
                    transition_steps=total_updates - decay_start,
                ),
            ],
            boundaries=[decay_start],
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr_schedule, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx)

        rng, _rng = jax.random.split(rng)
        reset_keys = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_keys, env_params)

        # ------------------------------------------------------------------ #
        #  Single PPO update step (JIT-compiled)                              #
        # ------------------------------------------------------------------ #

        @jax.jit
        def _update_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state

            # --- Rollout --- #
            def _env_step(carry, _):
                train_state, env_state, obs, rng = carry
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply({"params": train_state.params}, obs)
                action     = pi.sample(_rng)
                log_prob   = pi.log_prob(action)
                action_mean = pi.mean
                action_std  = jnp.broadcast_to(pi.base_dist.scale, action_mean.shape)

                rng, _rng = jax.random.split(rng)
                step_keys = jax.random.split(_rng, config["NUM_ENVS"])
                new_obs, new_env_state, reward, done, info = env.step(
                    step_keys, env_state, action, env_params)

                trans = Transition(done, action, value, reward, log_prob,
                                   obs, info, action_mean, action_std)
                return (train_state, new_env_state, new_obs, rng), trans

            rollout_carry = (train_state, env_state, last_obs, rng)
            rollout_carry, traj = jax.lax.scan(
                _env_step, rollout_carry, None, length=config["NUM_STEPS"])
            train_state, env_state, last_obs, rng = rollout_carry

            # --- GAE advantages --- #
            _, last_val = network.apply({"params": train_state.params}, last_obs)

            def _gae(carry, trans):
                gae, next_val = carry
                delta = trans.reward + config["GAMMA"] * next_val * (1 - trans.done) - trans.value
                gae   = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - trans.done) * gae
                return (gae, trans.value), gae

            _, advantages = jax.lax.scan(
                _gae, (jnp.zeros_like(last_val), last_val), traj, reverse=True, unroll=16)
            targets    = advantages + traj.value
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- PPO epochs --- #
            def _update_epoch(carry, _):
                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                batch_size  = config["NUM_ENVS"] * config["NUM_STEPS"]
                perm        = jax.random.permutation(_rng, batch_size)
                flat_batch  = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), (traj, advantages, targets))
                shuffled    = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), flat_batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape([config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled)

                def _loss(params, traj_mb, adv_mb, targ_mb):
                    pi, value = network.apply({"params": params}, traj_mb.obs)
                    log_prob  = pi.log_prob(traj_mb.action)

                    v_pred = traj_mb.value + jnp.clip(
                        value - traj_mb.value, -config["CLIP_EPS"], config["CLIP_EPS"]) \
                        if config["CLIP_VF"] else value
                    v_loss = 0.5 * jnp.square(targ_mb - v_pred).mean()

                    ratio   = jnp.exp(log_prob - traj_mb.log_prob)
                    pg_loss = -jnp.minimum(
                        ratio * adv_mb,
                        jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * adv_mb
                    ).mean()
                    entropy = pi.entropy().mean()
                    total   = pg_loss + config["VF_COEF"] * v_loss - config["ENT_COEF"] * entropy
                    kl      = ((ratio - 1) - jnp.log(ratio)).mean()
                    return total, {"loss": total, "pg": pg_loss, "vf": v_loss,
                                   "ent": entropy, "kl": kl}

                def _update_mb(carry, i):
                    ts, stop = carry
                    traj_mb, adv_mb, targ_mb = jax.tree_util.tree_map(
                        lambda x: x[i], minibatches)

                    def do_update():
                        (loss, m), grads = jax.value_and_grad(_loss, has_aux=True)(
                            ts.params, traj_mb, adv_mb, targ_mb)
                        return ts.apply_gradients(grads=grads), m, m["kl"] > config["KL_THRESHOLD"]

                    def skip_update():
                        traj0, adv0, targ0 = jax.tree_util.tree_map(lambda x: x[0], minibatches)
                        _, m = _loss(ts.params, traj0, adv0, targ0)
                        return ts, m, True

                    new_ts, metrics, kl_over = jax.lax.cond(stop, skip_update, do_update)
                    return (new_ts, stop | kl_over), None

                (train_state, _), _ = jax.lax.scan(
                    _update_mb, (train_state, False), jnp.arange(config["NUM_MINIBATCHES"]))
                return (train_state, rng), None

            (train_state, rng), _ = jax.lax.scan(
                _update_epoch, (train_state, rng), None, length=config["UPDATE_EPOCHS"])

            return (train_state, env_state, last_obs, rng), traj

        runner_state = (train_state, env_state, obsv, rng)
        all_metrics  = []
        best_return  = -float("inf")
        total_completed_episodes = 0
        total_episode_max_delivered = 0.0

        ckpt_interval = config.get("CHECKPOINT_INTERVAL", 0)

        for step in tqdm(range(total_updates), desc="PPO Updates"):
            runner_state, traj = _update_step(runner_state, step)

            inf_cpu = jax.device_get(traj.info)
            returned = inf_cpu["returned_episode"].astype(bool)
            ep_rets = inf_cpu["returned_episode_returns"][returned]
            avg_ep  = float(ep_rets.mean()) if ep_rets.size > 0 else 0.0
            ep_max_delivered = inf_cpu["returned_episode_max_delivered"][returned]
            completed_episodes = int(ep_max_delivered.size)
            total_completed_episodes += completed_episodes
            total_episode_max_delivered += float(ep_max_delivered.sum())
            task_rescue_rate = (
                total_episode_max_delivered / total_completed_episodes / float(NUM_TASKS) * 100.0
                if total_completed_episodes > 0 else 0.0
            )

            if (step + 1) % config["PRINT_INTERVAL"] == 0:
                print(f"\n[Step {step+1}/{total_updates}]  "
                      f"avg_ep_return={avg_ep:.1f}  "
                      f"task_rescue_rate={task_rescue_rate:.1f}%  "
                      f"alive={inf_cpu['alive_robots'].mean():.2f}  "
                      f"delivered={inf_cpu['delivered_tasks'].mean():.3f}  "
                      f"contaminated={inf_cpu['contaminated_tasks'].mean():.3f}  "
                      f"safety={inf_cpu['safety_indicator'].mean():.3f}")

            metrics = {
                "num_robots": NUM_ROBOTS,
                "num_tasks": NUM_TASKS,
                "charts/avg_episode_return": avg_ep,
                "charts/safety_indicator":   float(inf_cpu["safety_indicator"].mean()),
                "charts/alive_robots":       float(inf_cpu["alive_robots"].mean()),
                "charts/delivered_tasks":    float(inf_cpu["delivered_tasks"].mean()),
                "charts/task_rescue_rate":    task_rescue_rate,
                "charts/contaminated_tasks": float(inf_cpu["contaminated_tasks"].mean()),
                "rewards/progress":          float(inf_cpu["reward_progress"].mean()),
                "rewards/pickup":            float(inf_cpu["reward_pickup"].mean()),
                "rewards/delivery":          float(inf_cpu["reward_delivery"].mean()),
                "rewards/smooth_penalty":    float(inf_cpu["smooth_penalty"].mean()),
                "rewards/proximity_penalty": float(inf_cpu["proximity_penalty"].mean()),
            }
            wandb.log(metrics, step=step)
            all_metrics.append({"step": step, **metrics})

            # --- Checkpoint: periodic and best-return --- #
            current_params = jax.device_get(runner_state[0]).params
            if ckpt_interval > 0 and (step + 1) % ckpt_interval == 0:
                ckpt_path = os.path.join(save_dir, f"checkpoint_{step+1:07d}.pkl")
                with open(ckpt_path, "wb") as f:
                    f.write(serialization.to_bytes(current_params))
            if avg_ep > best_return:
                best_return = avg_ep
                best_path = os.path.join(save_dir, "best_params.pkl")
                with open(best_path, "wb") as f:
                    f.write(serialization.to_bytes(current_params))

        final_stats = {
            "num_robots": NUM_ROBOTS,
            "num_tasks": NUM_TASKS,
            "final_task_rescue_rate": all_metrics[-1]["charts/task_rescue_rate"] if all_metrics else 0.0,
        }
        return {"runner_state": runner_state, "metrics": all_metrics, "final_stats": final_stats}

    return train


# ========================================================================== #
#  MAIN                                                                       #
# ========================================================================== #

def _find_newest_params():
    best = glob.glob("results/**/best_params.pkl", recursive=True) + \
           glob.glob("best_params.pkl")
    if best:
        return max(best, key=os.path.getmtime)
    fallback = glob.glob("results/**/trained_params.pkl", recursive=True) + \
               glob.glob("trained_params.pkl")
    if fallback:
        print("Warning: no best_params.pkl found, falling back to trained_params.pkl")
        return max(fallback, key=os.path.getmtime)
    raise FileNotFoundError("No params file found under results/")


def _load_params(path, network, obs_dim):
    dummy    = jnp.zeros(obs_dim)
    template = network.init(jax.random.PRNGKey(0), dummy)["params"]
    with open(path, "rb") as f:
        raw = f.read()
    return serialization.from_bytes(template, raw)


def _save_training_metrics(save_dir, metrics, final_stats):
    metrics_pkl_path = os.path.join(save_dir, "training_metrics.pkl")
    with open(metrics_pkl_path, "wb") as f:
        pickle.dump({"metrics": metrics, "final_stats": final_stats}, f)

    if metrics:
        metrics_csv_path = os.path.join(save_dir, "training_metrics.csv")
        fieldnames = list(metrics[0].keys())
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
    else:
        metrics_csv_path = None

    stats_path = os.path.join(save_dir, "final_training_stats.txt")
    with open(stats_path, "w") as f:
        for key, value in final_stats.items():
            f.write(f"{key}: {value}\n")

    print(f"Saved training metrics → {metrics_pkl_path}")
    if metrics_csv_path is not None:
        print(f"Saved training metrics CSV → {metrics_csv_path}")
    print(f"Saved final training stats → {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SycaBot-JAX PPO policy")
    parser.add_argument("--warm-start",  action="store_true",
                        help="Initialize from the most recently trained checkpoint")
    parser.add_argument("--init-params", type=str, default=None,
                        help="Path to trained_params.pkl to warm-start from")
    args = parser.parse_args()

    rng = jax.random.PRNGKey(config["SEED"])

    # ---- Warm-start checkpoint loading ---- #
    init_params = None
    if args.init_params or args.warm_start:
        ckpt_path = args.init_params or _find_newest_params()
        _net_tmp  = ActorCritic(
            action_dim=SycaBotEnvJAX().action_space(EnvParams()).shape[0],
            hidden_size=config["HIDDEN_SIZE"],
            activation=config["ACTIVATION"],
        )
        _obs_dim  = SycaBotEnvJAX().observation_space(EnvParams()).shape[0]
        init_params = _load_params(ckpt_path, _net_tmp, _obs_dim)
        print(f"Warm-starting from: {ckpt_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = (f"PPO_hazard_jax"
                 f"_r{NUM_ROBOTS}t{NUM_TASKS}"
                 f"_lr{config['LR']}"
                 f"_envs{config['NUM_ENVS']}"
                 f"_steps{config['NUM_STEPS']}"
                 f"_mb{config['MINIBATCH_SIZE']}"
                 f"_{timestamp}")

    save_dir = os.path.join("results", run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved in: {save_dir}")

    # Save hyperparameters
    with open(os.path.join(save_dir, "hyperparameters.txt"), "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    wandb.init(
        project="sycabot-hazard",
        name=run_name,
        config={**config, "num_robots": NUM_ROBOTS, "num_tasks": NUM_TASKS},
        dir=save_dir,
    )

    train_fn = make_train(config, init_params=init_params, save_dir=save_dir)
    print(f"Starting training  |  {jax.device_count()} device(s): {jax.devices()}")
    out = train_fn(rng)

    _save_training_metrics(save_dir, out["metrics"], out["final_stats"])
    wandb.summary["final_task_rescue_rate"] = out["final_stats"]["final_task_rescue_rate"]
    wandb.finish()

    # ---- Save final params ---- #
    final_runner = jax.device_get(out["runner_state"])
    trained_params = final_runner[0].params
    params_path    = os.path.join(save_dir, "trained_params.pkl")
    with open(params_path, "wb") as f:
        f.write(serialization.to_bytes(trained_params))
    print(f"Saved trained parameters → {params_path}")
    print(f"\nAll outputs saved in: {save_dir}")
