"""Unconstrained PPO for SycaBot Hazard Training – pure JAX / GPU parallel."""

import argparse
import glob
import os
import pickle
from datetime import datetime

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import serialization
from typing import Sequence, NamedTuple, Any
from tqdm import tqdm
import matplotlib.pyplot as plt

from sycabot_env_jax import SycaBotEnvJAX, EnvParams
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
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "CLIP_VF": True,
    "KL_THRESHOLD": 0.015,
    # Rollout
    "NUM_ENVS": 1024,
    "NUM_STEPS": 64,
    "TOTAL_UPDATES": 100000,
    # Network
    "ACTIVATION": "tanh",
    "HIDDEN_SIZE": 256,
    # Misc
    "SEED": 42,
    "PRINT_INTERVAL": 1000,
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

def make_train(config, init_params=None):
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

        for step in tqdm(range(total_updates), desc="PPO Updates"):
            runner_state, traj = _update_step(runner_state, step)

            if (step + 1) % config["PRINT_INTERVAL"] == 0:
                tb  = jax.device_get(traj)
                inf = jax.device_get(traj.info)

                ep_ret = inf["returned_episode_returns"][inf["returned_episode"]]
                avg_ep = float(ep_ret.mean()) if ep_ret.size > 0 else 0.0

                print(f"\n[Step {step+1}/{total_updates}]  "
                      f"avg_ep_return={avg_ep:.1f}  "
                      f"alive={inf['alive_robots'].mean():.2f}  "
                      f"delivered={inf['delivered_tasks'].mean():.3f}  "
                      f"contaminated={inf['contaminated_tasks'].mean():.3f}  "
                      f"safety={inf['safety_indicator'].mean():.3f}")

            inf_cpu = jax.device_get(traj.info)
            ep_rets = inf_cpu["returned_episode_returns"][inf_cpu["returned_episode"]]
            all_metrics.append({
                "step":              step,
                "avg_return":        float(ep_rets.mean()) if ep_rets.size > 0 else 0.0,
                "safety":            float(inf_cpu["safety_indicator"].mean()),
                "alive":             float(inf_cpu["alive_robots"].mean()),
                "delivered":         float(inf_cpu["delivered_tasks"].mean()),
                "contaminated":      float(inf_cpu["contaminated_tasks"].mean()),
                "reward_progress":   float(inf_cpu["reward_progress"].mean()),
                "reward_pickup":     float(inf_cpu["reward_pickup"].mean()),
                "reward_delivery":   float(inf_cpu["reward_delivery"].mean()),
                "smooth_penalty":    float(inf_cpu["smooth_penalty"].mean()),
            })

        return {"runner_state": runner_state, "metrics": all_metrics}

    return train


# ========================================================================== #
#  MAIN                                                                       #
# ========================================================================== #

def _find_newest_params():
    candidates = glob.glob("results/**/trained_params.pkl", recursive=True) + \
                 glob.glob("trained_params.pkl")
    if not candidates:
        raise FileNotFoundError("No trained_params.pkl found under results/")
    return max(candidates, key=os.path.getmtime)


def _load_params(path, network, obs_dim):
    dummy    = jnp.zeros(obs_dim)
    template = network.init(jax.random.PRNGKey(0), dummy)["params"]
    with open(path, "rb") as f:
        raw = f.read()
    return serialization.from_bytes(template, raw)


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

    train_fn = make_train(config, init_params=init_params)
    print(f"Starting training  |  {jax.device_count()} device(s): {jax.devices()}")
    out = train_fn(rng)

    # ---- Save params ---- #
    final_runner = jax.device_get(out["runner_state"])
    trained_params = final_runner[0].params
    params_path    = os.path.join(save_dir, "trained_params.pkl")
    with open(params_path, "wb") as f:
        f.write(serialization.to_bytes(trained_params))
    print(f"Saved trained parameters → {params_path}")

    # ---- Plots ---- #
    metrics = out["metrics"]
    steps   = [m["step"] for m in metrics]

    fig, axs = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
    fig.suptitle(f"Training Metrics – {run_name}", fontsize=13)

    def _plot(ax, key, label, color):
        vals = np.array([m[key] for m in metrics])
        ax.plot(steps, vals, color=color, linewidth=1.5, label=label)
        ax.set_ylabel(label); ax.legend(); ax.grid(True)

    _plot(axs[0, 0], "avg_return",      "Avg Episode Return",       "steelblue")
    _plot(axs[0, 1], "safety",          "Safety Indicator",         "green")
    _plot(axs[1, 0], "alive",           "Avg Alive Robots",         "darkorange")
    _plot(axs[1, 1], "delivered",       "Delivered Tasks",          "purple")
    _plot(axs[2, 0], "reward_progress", "Progress Reward",          "royalblue")
    _plot(axs[2, 1], "smooth_penalty",  "Smoothness Penalty",       "red")

    axs[2, 0].set_xlabel("PPO Update Step")
    axs[2, 1].set_xlabel("PPO Update Step")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training plot  → {plot_path}")

    # Reward components subplot
    fig2, axs2 = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig2.suptitle("Reward Components", fontsize=13)
    _plot2 = lambda ax, k, l, c: (_plot(ax, k, l, c), ax.set_xlabel("Step"))
    _plot(axs2[0, 0], "reward_pickup",   "Pickup Reward",    "darkorange")
    _plot(axs2[0, 1], "reward_delivery", "Delivery Reward",  "forestgreen")
    _plot(axs2[1, 0], "reward_progress", "Progress Reward",  "royalblue")
    _plot(axs2[1, 1], "contaminated",    "Contaminated Tasks", "firebrick")
    for ax in axs2[1]:
        ax.set_xlabel("PPO Update Step")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_components.png"), dpi=150)

    print(f"\nAll outputs saved in: {save_dir}")
