"""Monte Carlo policy evaluation for SycaBot hazard training.

Tests
-----
hazards   Vary num_initial_fires (1–N); fixed r=2, t=2 policy.
spread    Vary fire_spread_prob over a range; fixed r=2, t=2 policy.
tasks     Vary num_tasks (2–5); loads the best r2t{n} policy for each count.

For hazards and spread, the r2t2 best_params.pkl (most recently modified) is
loaded automatically.  For the tasks sweep, the best_params.pkl (or
trained_params.pkl fallback) from the most recent r2t{n} run is used.

Usage
-----
    python monte_carlo_analysis.py                                 # all three tests
    python monte_carlo_analysis.py --test hazards spread           # subset
    python monte_carlo_analysis.py --episodes 100 --seed 42
    python monte_carlo_analysis.py --hazard-counts 1 2 3 4 5 6
    python monte_carlo_analysis.py --spread-rates 0.005 0.01 0.02 0.05 0.10
"""

import argparse
import glob
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import serialization
from tqdm import tqdm

# ActorCritic and its hyperparameters do not depend on NUM_ROBOTS / NUM_TASKS
# (they take explicit action_dim / hidden_size args), so importing once is safe.
from train_ppo import ActorCritic, config as train_config


# =========================================================================== #
#  Policy discovery & loading                                                  #
# =========================================================================== #

def _find_best_policy(n_robots, n_tasks):
    """Return path to best_params.pkl (or trained_params.pkl) for r{n}t{m}."""
    for fname in ("best_params.pkl", "trained_params.pkl"):
        pat = f"results/PPO_hazard_jax_r{n_robots}t{n_tasks}_*/{fname}"
        hits = glob.glob(pat)
        if hits:
            path = max(hits, key=os.path.getmtime)
            if fname != "best_params.pkl":
                print(f"    [warn] no best_params.pkl found; using {fname}")
            return path
    return None


def _load_params(path, network, obs_dim):
    dummy    = jnp.zeros(obs_dim)
    template = network.init(jax.random.PRNGKey(0), dummy)["params"]
    with open(path, "rb") as f:
        raw = f.read()
    return serialization.from_bytes(template, raw)


# =========================================================================== #
#  Environment configuration                                                   #
# =========================================================================== #

def _patch_env_globals(n_robots, n_tasks):
    """Patch sycabot_env_jax module globals so all subsequent method calls
    use the new robot / task counts.  Works because Python looks up module
    globals at function-call time, not at class-definition time."""
    import sycabot_env_jax as em
    em.NUM_ROBOTS = n_robots
    em.NUM_TASKS  = n_tasks


def _build_env_and_network(n_robots, n_tasks, params_path):
    """Patch globals, instantiate env + network, load policy weights."""
    _patch_env_globals(n_robots, n_tasks)

    import sycabot_env_jax as em
    env        = em.SycaBotEnvJAX()
    env_params = em.EnvParams()
    obs_dim    = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).shape[0]

    network = ActorCritic(
        action_dim  = action_dim,
        hidden_size = train_config["HIDDEN_SIZE"],
        activation  = train_config["ACTIVATION"],
    )
    trained_params = _load_params(params_path, network, obs_dim)
    return env, env_params, network, trained_params


# =========================================================================== #
#  Episode runner                                                              #
# =========================================================================== #

def _make_episode_keys(n_episodes, seed):
    """Pre-generate one independent key per episode.

    Passing the same array to _run_episodes for every configuration within a
    test guarantees matched episodes: episode i always starts from the same
    initial robot / task / fire-seed positions regardless of which env_params
    variant is being evaluated.  Only the env_params (spread rate, hazard
    count, …) differ across variants.
    """
    return jax.random.split(jax.random.PRNGKey(seed), n_episodes)


def _run_episodes(env, env_params, network, trained_params,
                  episode_keys, max_steps):
    """Run one episode per key; return float32 array of (delivered / total_tasks).

    episode_keys  pre-generated array of shape (n_episodes, 2); the same array
                  should be passed for every configuration within a test so that
                  episode i always has the same initial state.
    """
    import sycabot_env_jax as em
    n_tasks    = em.NUM_TASKS
    n_episodes = len(episode_keys)

    # JIT the policy; one compilation per (network, params) pair.
    @jax.jit
    def _infer(obs, key):
        pi, _ = network.apply({"params": trained_params}, obs)
        return pi.mean   # deterministic

    results = np.empty(n_episodes, dtype=np.float32)

    for ep in tqdm(range(n_episodes), desc="    episodes", leave=False, ncols=70):
        # Each episode starts from a fixed, configuration-independent key so
        # that the reset (robot positions, task positions, fire seed locations)
        # is identical across all configurations being compared.
        rng = episode_keys[ep]
        rng, reset_key = jax.random.split(rng)
        obs, state = env.reset_env(reset_key, env_params)

        done = False
        step = 0
        while not done and step < max_steps:
            rng, ak, sk = jax.random.split(rng, 3)
            action = _infer(obs, ak)
            obs, state, _, done, _ = env.step_env(sk, state, action, env_params)
            done = bool(done)
            step += 1

        delivered   = float(jnp.sum(state.task_status == 2))
        results[ep] = delivered / n_tasks

    return results


# =========================================================================== #
#  Individual tests                                                            #
# =========================================================================== #

def test_hazards(n_episodes, max_steps, seed, hazard_counts, out_dir):
    """Box plot: rescue rate vs. num_initial_fires  (fixed r=2, t=3)."""
    print("\n=== Test 1: number of initial hazards (r=2, t=3) ===")
    path = _find_best_policy(2, 3)
    if path is None:
        print("  No r2t3 policy found — skipping")
        return
    print(f"  Policy: {path}")

    env, base_params, network, params = _build_env_and_network(2, 3, path)

    # Shared keys: episode i uses the same reset key for every hazard count.
    # _spawn_fire samples _MAX_INITIAL_FIRES cells with replace=False then
    # activates only the first num_fires of them, so hazard count k+1 always
    # includes all cells from hazard count k plus one additional seed.
    episode_keys = _make_episode_keys(n_episodes, seed)

    data = {}
    for n in hazard_counts:
        ep_params = base_params.replace(num_initial_fires=n)
        pct = _run_episodes(env, ep_params, network, params,
                            episode_keys, max_steps)
        data[n] = pct
        print(f"  hazards={n:2d}:  mean={pct.mean():.1%}  "
              f"median={np.median(pct):.1%}  std={pct.std():.1%}")

    _save_csv(data, var_col="hazard_count", n_tasks=3,
              out_path=os.path.join(out_dir, "hazards_results.csv"))
    _boxplot(data,
             xlabel="Number of initial hazards",
             title="Rescue rate vs. number of initial hazards\n(2 robots, 2 tasks)",
             out_path=os.path.join(out_dir, "hazards_boxplot.png"))


def test_spread(n_episodes, max_steps, seed, spread_rates, out_dir):
    """Box plot: rescue rate vs. fire_spread_prob  (fixed r=2, t=3)."""
    print("\n=== Test 2: fire spread rate (r=2, t=3) ===")
    path = _find_best_policy(2, 3)
    if path is None:
        print("  No r2t3 policy found — skipping")
        return
    print(f"  Policy: {path}")

    env, base_params, network, params = _build_env_and_network(2, 3, path)

    # Shared keys: episode i starts from identical robot / task / fire positions
    # for every spread rate; only the fire propagation dynamics differ.
    episode_keys = _make_episode_keys(n_episodes, seed)

    data = {}
    for rate in spread_rates:
        ep_params = base_params.replace(fire_spread_prob=float(rate))
        pct = _run_episodes(env, ep_params, network, params,
                            episode_keys, max_steps)
        data[rate] = pct
        print(f"  spread={rate:.4f}:  mean={pct.mean():.1%}  "
              f"median={np.median(pct):.1%}  std={pct.std():.1%}")

    _save_csv(data, var_col="spread_rate", n_tasks=3,
              out_path=os.path.join(out_dir, "spread_results.csv"),
              label_fmt="{:.4f}")
    _boxplot(data,
             xlabel="Fire spread probability per step",
             title="Rescue rate vs. fire spread rate\n(2 robots, 2 tasks)",
             out_path=os.path.join(out_dir, "spread_boxplot.png"),
             label_fmt="{:.3f}")


def test_tasks(n_episodes, max_steps, seed, task_counts, out_dir):
    """Box plot: rescue rate vs. num_tasks, using the matching r2t{n} policy."""
    print("\n=== Test 3: number of tasks (r=2, policy matched per task count) ===")

    # Shared keys so that episode i uses the same PRNG state for every task
    # count.  Note: because reset_env splits the key into (1 + 2*N_robots +
    # N_tasks + 1) sub-keys, the robot-position keys are the same across task
    # counts but the fire key shifts with N_tasks.  Robot and task spawn
    # positions are therefore consistent for the shared robot keys.
    episode_keys = _make_episode_keys(n_episodes, seed)

    data = {}
    for n_tasks in task_counts:
        path = _find_best_policy(2, n_tasks)
        if path is None:
            print(f"  No r2t{n_tasks} policy found — skipping n_tasks={n_tasks}")
            continue
        print(f"  n_tasks={n_tasks}: {path}")

        env, env_params, network, params = _build_env_and_network(2, n_tasks, path)
        pct = _run_episodes(env, env_params, network, params,
                            episode_keys, max_steps)
        data[n_tasks] = pct
        print(f"    mean={pct.mean():.1%}  median={np.median(pct):.1%}  "
              f"std={pct.std():.1%}")

    if data:
        # n_tasks=None → each row uses its key (the task count) as n_tasks
        _save_csv(data, var_col="num_tasks", n_tasks=None,
                  out_path=os.path.join(out_dir, "tasks_results.csv"))
        _boxplot(data,
                 xlabel="Number of tasks",
                 title="Rescue rate vs. number of tasks\n(2 robots, matched policy)",
                 out_path=os.path.join(out_dir, "tasks_boxplot.png"))


# =========================================================================== #
#  CSV export                                                                  #
# =========================================================================== #

def _save_csv(data_dict, var_col, n_tasks, out_path, label_fmt="{}"):
    """Write per-configuration summary statistics to a CSV file.

    n_tasks  fixed int for hazard/spread tests; pass None for the tasks test,
             in which case each row uses its key value as n_tasks.

    Columns
    -------
    <var_col>           value of the varied parameter
    num_episodes        episodes run per configuration
    n_tasks             number of tasks per episode
    rescue_rate_mean    mean(delivered / n_tasks) across episodes  [0–1]
    rescue_rate_median  median of the same
    rescue_rate_std     standard deviation
    rescue_rate_q25     25th percentile
    rescue_rate_q75     75th percentile
    tasks_rescued_pct   rescue_rate_mean * 100                     [0–100]
    tasks_rescued_mean  rescue_rate_mean * n_tasks                 expected tasks/ep
    full_success_rate   fraction of episodes where ALL tasks delivered
    full_success_count  full_success_rate * num_episodes
    failure_count       (1 – full_success_rate) * num_episodes
    """
    import csv

    keys = sorted(data_dict.keys())
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            var_col, "num_episodes", "n_tasks",
            "rescue_rate_mean", "rescue_rate_median", "rescue_rate_std",
            "rescue_rate_q25", "rescue_rate_q75",
            "tasks_rescued_pct", "tasks_rescued_mean",
            "full_success_rate", "full_success_count", "failure_count",
        ])
        for k in keys:
            pct  = data_dict[k]          # values in [0, 1]
            n    = len(pct)
            nt   = k if n_tasks is None else n_tasks   # per-row n_tasks
            full_success_rate = float(np.mean(pct >= 1.0 - 1e-6))
            writer.writerow([
                label_fmt.format(k),
                n,
                nt,
                f"{pct.mean():.6f}",
                f"{np.median(pct):.6f}",
                f"{pct.std():.6f}",
                f"{np.percentile(pct, 25):.6f}",
                f"{np.percentile(pct, 75):.6f}",
                f"{pct.mean() * 100:.4f}",
                f"{pct.mean() * nt:.4f}",
                f"{full_success_rate:.6f}",
                f"{full_success_rate * n:.4f}",
                f"{(1 - full_success_rate) * n:.4f}",
            ])

    print(f"  Saved → {out_path}")


# =========================================================================== #
#  Plotting                                                                    #
# =========================================================================== #

def _boxplot(data_dict, xlabel, title, out_path, label_fmt="{}"):
    keys    = sorted(data_dict.keys())
    arrays  = [data_dict[k] * 100.0 for k in keys]
    labels  = [label_fmt.format(k) for k in keys]
    pos     = list(range(1, len(keys) + 1))

    fig, ax = plt.subplots(figsize=(max(5, len(keys) * 1.5), 5))

    # White (unfilled) boxes with black outlines
    bp = ax.boxplot(arrays, positions=pos, labels=labels,
                    patch_artist=True,
                    boxprops=dict(facecolor="white", color="black", linewidth=1.2),
                    medianprops=dict(color="orange", linewidth=2.0),
                    whiskerprops=dict(color="black", linewidth=1.0),
                    capprops=dict(color="black", linewidth=1.0),
                    flierprops=dict(marker="o", markerfacecolor="white",
                                   markeredgecolor="black", markersize=4))

    # Line connecting the medians
    medians = [np.median(arr) for arr in arrays]
    ax.plot(pos, medians, color="red", linestyle="-", linewidth=1.5,
            marker="o", markersize=4, zorder=5, label="Median")

    # Jittered individual episode points
    rng = np.random.default_rng(0)
    for i, arr in enumerate(arrays):
        jitter = rng.uniform(-0.15, 0.15, size=len(arr))
        ax.scatter(np.full_like(arr, pos[i]) + jitter, arr,
                   color="black", alpha=0.25, s=10, zorder=3)

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Task rescue rate (%)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-5, 105)
    ax.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# =========================================================================== #
#  CLI                                                                         #
# =========================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo policy evaluation — SycaBot hazard training")
    parser.add_argument("--test", nargs="+",
                        choices=["hazards", "spread", "tasks"],
                        default=["hazards", "spread", "tasks"],
                        help="Which tests to run (default: all three)")
    parser.add_argument("--episodes",  type=int,   default=50,
                        help="Episodes per configuration (default: 50)")
    parser.add_argument("--max-steps", type=int,   default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--out-dir",   type=str,   default="mc_results",
                        help="Directory for saved plots (default: mc_results/)")
    # Per-test sweep ranges
    parser.add_argument("--hazard-counts", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5],
                        help="num_initial_fires values to sweep (default: 1–5)")
    parser.add_argument("--spread-rates",  nargs="+", type=float,
                        default=[0.020, 0.040, 0.060, 0.080, 0.100],
                        help="fire_spread_prob values to sweep")
    parser.add_argument("--task-counts",   nargs="+", type=int,
                        default=[2, 3, 4, 5],
                        help="num_tasks values to sweep (default: 2–5)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Episodes per config : {args.episodes}")
    print(f"Max steps / episode : {args.max_steps}")
    print(f"Output directory    : {args.out_dir}")
    print(f"JAX devices         : {jax.devices()}")

    if "hazards" in args.test:
        test_hazards(args.episodes, args.max_steps, args.seed,
                     args.hazard_counts, args.out_dir)

    if "spread" in args.test:
        test_spread(args.episodes, args.max_steps, args.seed,
                    args.spread_rates, args.out_dir)

    if "tasks" in args.test:
        test_tasks(args.episodes, args.max_steps, args.seed,
                   args.task_counts, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
