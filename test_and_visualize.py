"""Test and visualize a trained SycaBot JAX policy.

Modes
-----
--render          Real-time pygame rendering of N episodes.
--plot            Trajectory + metrics plots saved to PNG (no display needed).
--both            Both of the above (default).
--video           Save all episodes as a single video (MP4 or GIF fallback).

Usage examples
--------------
    python test_and_visualize.py                           # pick newest params
    python test_and_visualize.py --params results/.../trained_params.pkl
    python test_and_visualize.py --render --episodes 5
    python test_and_visualize.py --plot   --episodes 20
    python test_and_visualize.py --video  --episodes 5    # save episodes.mp4
    python test_and_visualize.py --deterministic           # use policy mean
"""

import argparse
import glob
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from flax import serialization

from sycabot_env_jax import SycaBotEnvJAX, EnvParams, GRID_X, GRID_Y, _X_MIN, _Y_MIN, _CELL_SIZE

# One distinct colour per robot slot; extend if NUM_ROBOTS ever exceeds 6.
_ROBOT_PALETTE = ["royalblue", "darkorange", "forestgreen", "crimson", "orchid", "sienna"]
from train_ppo import ActorCritic, config as train_config


# ========================================================================== #
#  Argument parsing                                                           #
# ========================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained SycaBot-JAX policy")
    parser.add_argument("--params",       type=str,  default=None,
                        help="Path to trained_params.pkl (auto-selects newest if omitted)")
    parser.add_argument("--episodes",     type=int,  default=5,
                        help="Number of episodes to run")
    parser.add_argument("--max-steps",    type=int,  default=500,
                        help="Max steps per episode (overrides env default)")
    parser.add_argument("--render",       action="store_true")
    parser.add_argument("--plot",         action="store_true")
    parser.add_argument("--both",         action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use policy mean instead of sampling")
    parser.add_argument("--video",        action="store_true",
                        help="Save all episodes as a single video file (MP4 or GIF)")
    parser.add_argument("--fps",          type=int,  default=15,
                        help="Rendering / video FPS")
    parser.add_argument("--seed",         type=int,  default=0)
    parser.add_argument("--out-dir",      type=str,  default="test_results",
                        help="Directory for saved plots")
    return parser.parse_args()


# ========================================================================== #
#  Parameter loading                                                          #
# ========================================================================== #

def find_newest_params():
    best = glob.glob("results/**/best_params.pkl", recursive=True) + \
           glob.glob("best_params.pkl")
    if best:
        return max(best, key=os.path.getmtime)
    fallback = glob.glob("results/**/trained_params.pkl", recursive=True) + \
               glob.glob("trained_params.pkl")
    if fallback:
        print("Warning: no best_params.pkl found, falling back to trained_params.pkl")
        return max(fallback, key=os.path.getmtime)
    sys.exit("No params file found.  Run train_ppo.py first, or pass --params <path>.")


def load_params(path, network, obs_dim):
    dummy = jnp.zeros(obs_dim)
    template = network.init(jax.random.PRNGKey(0), dummy)["params"]
    with open(path, "rb") as f:
        raw = f.read()
    return serialization.from_bytes(template, raw)


# ========================================================================== #
#  Single episode rollout                                                     #
# ========================================================================== #

def run_episode(network, trained_params, env, env_params, key,
                max_steps: int, deterministic: bool, renderer=None, fps: int = 15):
    """Run one episode; return trajectory data dict."""
    key, reset_key = jax.random.split(key)
    obs, state = env.reset_env(reset_key, env_params)

    # JIT inference once (compiled on first call)
    @jax.jit
    def policy_step(obs, key):
        pi, value = network.apply({"params": trained_params}, obs)
        action = pi.mean if deterministic else pi.sample(key)
        return action, pi.mean, jnp.exp(pi.base_dist.scale)

    history = {
        "pos":          [],   # list of (NUM_ROBOTS, 2)
        "alive":        [],
        "carrying":     [],
        "task_status":  [],
        "task_pos":     [],
        "fire_grid":    [],
        "reward":       [],
        "safety":       [],
        "delivered":    [],
        "contaminated": [],
        "step":         [],
    }

    total_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        key, act_key, step_key = jax.random.split(key, 3)
        action, mean_act, std_act = policy_step(obs, act_key)
        obs, state, reward, done, info = env.step_env(step_key, state, action, env_params)

        # Collect to host
        history["pos"].append(np.array(state.robot_pos))
        history["alive"].append(np.array(state.robot_alive))
        history["carrying"].append(np.array(state.robot_carrying))
        history["task_status"].append(np.array(state.task_status))
        history["task_pos"].append(np.array(state.task_pos))
        history["fire_grid"].append(np.array(state.fire_grid))
        history["reward"].append(float(reward))
        history["safety"].append(float(info["safety_indicator"]))
        history["delivered"].append(float(info["delivered_tasks"]))
        history["contaminated"].append(float(info["contaminated_tasks"]))
        history["step"].append(step)

        total_reward += float(reward)
        step += 1

        if renderer is not None:
            alive = renderer.render(env, state, env_params, fps=fps)
            if not alive:
                break

        done = bool(done)

    history["total_reward"] = total_reward
    history["length"]       = step
    return history


# ========================================================================== #
#  Trajectory plot                                                            #
# ========================================================================== #

def plot_trajectory(history, env, env_params, episode_idx: int, out_dir: str):
    """Full trajectory plot: map + reward + safety over time."""
    obs_start = np.array(env.obs_start)
    obs_end   = np.array(env.obs_end)
    exits     = np.array(env.exits)
    params    = env_params

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Episode {episode_idx + 1}  |  "
                 f"Return={history['total_reward']:.1f}  "
                 f"Steps={history['length']}", fontsize=14)

    # ---- Left panel: spatial map ---- #
    ax_map = fig.add_subplot(1, 2, 1)
    ax_map.set_aspect("equal")
    ax_map.set_xlim(params.x_min - 0.05, params.x_max + 0.05)
    ax_map.set_ylim(params.y_min - 0.05, params.y_max + 0.05)
    ax_map.set_xlabel("x (m)"); ax_map.set_ylabel("y (m)")
    ax_map.set_title("Spatial Trajectory")
    ax_map.set_facecolor("#f5f5f5")

    # Obstacles
    for s, e in zip(obs_start, obs_end):
        ax_map.plot([s[0], e[0]], [s[1], e[1]], "k-", linewidth=2.5)

    # Exits
    for ex in exits:
        ax_map.plot(ex[0], ex[1], "^", color="limegreen", markersize=12, zorder=5)

    # Final fire grid
    final_fire = history["fire_grid"][-1]
    for gx in range(GRID_X):
        for gy in range(GRID_Y):
            if final_fire[gx, gy] > 0:
                cx = _X_MIN + (gx + 0.5) * _CELL_SIZE
                cy = _Y_MIN + (gy + 0.5) * _CELL_SIZE
                rect = plt.Rectangle(
                    (cx - _CELL_SIZE / 2, cy - _CELL_SIZE / 2),
                    _CELL_SIZE, _CELL_SIZE,
                    color="orangered", alpha=0.45, zorder=1)
                ax_map.add_patch(rect)

    # Initial task positions
    init_task_pos = history["task_pos"][0]   # (NUM_TASKS, 2)
    for i, tp in enumerate(init_task_pos):
        ax_map.plot(tp[0], tp[1], "*", color="mediumpurple",
                    markersize=14, zorder=6, label=f"Task {i}" if i == 0 else None)

    # Robot trajectories
    positions = np.stack(history["pos"])          # (T, NUM_ROBOTS, 2)
    alive_arr = np.stack(history["alive"])        # (T, NUM_ROBOTS)
    n_robots  = positions.shape[1]

    for r in range(n_robots):
        rc = _ROBOT_PALETTE[r]
        xy = positions[:, r, :]                   # (T, 2)
        alive_mask = alive_arr[:, r] > 0.5        # (T,)

        carry_arr  = np.stack(history["carrying"])[:, r]  # (T,)
        segments   = np.stack([xy[:-1], xy[1:]], axis=1)  # (T-1, 2, 2)
        seg_carry  = carry_arr[:-1] > 0.5
        alive_segs = alive_mask[:-1]

        for t in range(len(segments)):
            if not alive_segs[t]:
                continue
            ls = "--" if seg_carry[t] else "-"
            ax_map.plot(segments[t, :, 0], segments[t, :, 1],
                        color=rc, linewidth=1.5, linestyle=ls, alpha=0.7)

        ax_map.plot(xy[0, 0], xy[0, 1], "o", color=rc, markersize=10, zorder=7)
        last_alive = alive_arr[-1, r] > 0.5
        ax_map.plot(xy[-1, 0], xy[-1, 1],
                    "o" if last_alive else "x",
                    color=rc, markersize=10, zorder=7, markeredgewidth=2)

    # Legend — built dynamically from actual robot count
    legend_handles = [
        mpatches.Patch(color=_ROBOT_PALETTE[r], label=f"Robot {r}")
        for r in range(n_robots)
    ] + [
        mpatches.Patch(color="mediumpurple", label="Task (initial)"),
        mpatches.Patch(color="limegreen",    label="Exit"),
        mpatches.Patch(color="orangered",    alpha=0.5, label="Fire (final)"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Carrying"),
    ]
    ax_map.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # ---- Right panel: time-series metrics ---- #
    steps     = history["step"]
    rewards   = np.cumsum(history["reward"])
    safety    = history["safety"]
    delivered = history["delivered"]
    contam    = history["contaminated"]

    gs_right = matplotlib.gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=fig.add_subplot(1, 2, 2).get_subplotspec(),
        hspace=0.4)
    # Hack: remove the dummy subplot and re-use its grid spec area
    fig.axes[-1].remove()

    ax1 = fig.add_subplot(gs_right[0])
    ax2 = fig.add_subplot(gs_right[1], sharex=ax1)
    ax3 = fig.add_subplot(gs_right[2], sharex=ax1)
    ax4 = fig.add_subplot(gs_right[3], sharex=ax1)

    ax1.plot(steps, rewards,   color="steelblue");   ax1.set_ylabel("Cumulative Return"); ax1.grid(True)
    ax2.plot(steps, safety,    color="green");        ax2.set_ylabel("Safety");    ax2.set_ylim(-0.05, 1.05); ax2.grid(True)
    ax3.plot(steps, delivered, color="purple");       ax3.set_ylabel("Delivered");  ax3.grid(True)
    ax4.plot(steps, contam,    color="firebrick");    ax4.set_ylabel("Contaminated"); ax4.set_xlabel("Step"); ax4.grid(True)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"episode_{episode_idx + 1:03d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved trajectory plot → {save_path}")


# ========================================================================== #
#  Fire spread animation (matplotlib)                                         #
# ========================================================================== #

def plot_fire_spread(history, env, env_params, episode_idx: int, out_dir: str,
                     n_frames: int = 8):
    """Grid of fire snapshots across the episode."""
    obs_start = np.array(env.obs_start)
    obs_end   = np.array(env.obs_end)
    params    = env_params
    T         = len(history["fire_grid"])
    frame_idx = np.linspace(0, T - 1, n_frames, dtype=int)

    fig, axs = plt.subplots(2, n_frames // 2, figsize=(3 * n_frames // 2, 7))
    fig.suptitle(f"Fire Spread – Episode {episode_idx + 1}", fontsize=13)
    axs = axs.flatten()

    for col, fi in enumerate(frame_idx):
        ax = axs[col]
        fg = history["fire_grid"][fi]
        ax.set_xlim(params.x_min, params.x_max)
        ax.set_ylim(params.y_min, params.y_max)
        ax.set_aspect("equal")
        ax.set_title(f"t={fi}", fontsize=9)
        ax.axis("off")

        for s, e in zip(obs_start, obs_end):
            ax.plot([s[0], e[0]], [s[1], e[1]], "k-", linewidth=1.5)

        for gx in range(GRID_X):
            for gy in range(GRID_Y):
                if fg[gx, gy] > 0:
                    cx = _X_MIN + (gx + 0.5) * _CELL_SIZE
                    cy = _Y_MIN + (gy + 0.5) * _CELL_SIZE
                    rect = plt.Rectangle(
                        (cx - _CELL_SIZE / 2, cy - _CELL_SIZE / 2),
                        _CELL_SIZE, _CELL_SIZE, color="orangered", alpha=0.6)
                    ax.add_patch(rect)

        pos = history["pos"][fi]
        alive = history["alive"][fi]
        for r in range(pos.shape[0]):
            if alive[r] > 0.5:
                ax.plot(pos[r, 0], pos[r, 1], "o", color=_ROBOT_PALETTE[r], markersize=6, zorder=5)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"fire_spread_{episode_idx + 1:03d}.png")
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fire-spread plot  → {save_path}")


# ========================================================================== #
#  Summary plot across all episodes                                           #
# ========================================================================== #

def plot_summary(all_histories, out_dir: str):
    returns    = [h["total_reward"] for h in all_histories]
    lengths    = [h["length"]       for h in all_histories]
    delivered  = [max(h["delivered"])    if h["delivered"]    else 0 for h in all_histories]
    contam     = [max(h["contaminated"]) if h["contaminated"] else 0 for h in all_histories]
    final_safe = [h["safety"][-1]        if h["safety"]       else 0 for h in all_histories]
    eps        = np.arange(1, len(returns) + 1)

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Test Summary", fontsize=14)

    def bar(ax, vals, label, color):
        ax.bar(eps, vals, color=color, alpha=0.8)
        ax.axhline(np.mean(vals), color="black", linestyle="--",
                   label=f"mean={np.mean(vals):.2f}")
        ax.set_xlabel("Episode"); ax.set_ylabel(label)
        ax.set_title(label); ax.legend(fontsize=8); ax.grid(True, axis="y")

    bar(axs[0, 0], returns,    "Episode Return",         "steelblue")
    bar(axs[0, 1], lengths,    "Episode Length (steps)", "darkorange")
    bar(axs[0, 2], final_safe, "Final Safety Indicator", "green")
    bar(axs[1, 0], delivered,  "Max Delivered Tasks",    "purple")
    bar(axs[1, 1], contam,     "Max Contaminated Tasks", "firebrick")

    # Cumulative return over time for each episode
    ax6 = axs[1, 2]
    ax6.set_title("Cumulative Return Curves")
    ax6.set_xlabel("Step"); ax6.set_ylabel("Cumulative Return")
    ax6.grid(True)
    for i, h in enumerate(all_histories):
        ax6.plot(np.cumsum(h["reward"]), alpha=0.7, label=f"Ep {i+1}")
    if len(all_histories) <= 10:
        ax6.legend(fontsize=7)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved summary plot → {save_path}")


# ========================================================================== #
#  Video export                                                               #
# ========================================================================== #

_TASK_STATUS_COLOR = {0: "mediumpurple", 1: "gold", 2: "limegreen", 3: "gray"}
_TRAIL_LEN = 30   # steps of position trail shown behind each robot


def save_video(all_histories, env, env_params, out_dir: str, fps: int = 15):
    """Render all episodes into a single MP4 (falls back to GIF if ffmpeg absent)."""
    import matplotlib.animation as animation

    obs_start = np.array(env.obs_start)
    obs_end   = np.array(env.obs_end)
    exits     = np.array(env.exits)
    params    = env_params

    # Build flat frame list: (episode_index, step_within_episode)
    frame_index = []
    for ep_idx, history in enumerate(all_histories):
        for t in range(history["length"]):
            frame_index.append((ep_idx, t))

    # Pre-stack position arrays per episode for fast trail rendering
    pos_stacked = [np.stack(h["pos"]) for h in all_histories]  # list of (T, N, 2)

    fig, ax = plt.subplots(figsize=(5, 10))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.03)

    def draw_frame(idx):
        ep_idx, t = frame_index[idx]
        history   = all_histories[ep_idx]
        pos_arr   = pos_stacked[ep_idx]        # (T, N, 2)

        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(params.x_min - 0.05, params.x_max + 0.05)
        ax.set_ylim(params.y_min - 0.05, params.y_max + 0.05)
        ax.set_facecolor("#f5f5f5")
        ax.set_title(
            f"Ep {ep_idx + 1}/{len(all_histories)}  "
            f"step {t + 1}/{history['length']}  "
            f"return={history['total_reward']:.1f}",
            fontsize=9,
        )

        # Obstacles
        for s, e in zip(obs_start, obs_end):
            ax.plot([s[0], e[0]], [s[1], e[1]], "k-", linewidth=2.5)

        # Exits
        for ex in exits:
            ax.plot(ex[0], ex[1], "^", color="limegreen", markersize=10, zorder=5)

        # Fire grid
        fg = history["fire_grid"][t]
        for gx in range(GRID_X):
            for gy in range(GRID_Y):
                if fg[gx, gy] > 0:
                    cx = _X_MIN + (gx + 0.5) * _CELL_SIZE
                    cy = _Y_MIN + (gy + 0.5) * _CELL_SIZE
                    ax.add_patch(plt.Rectangle(
                        (cx - _CELL_SIZE / 2, cy - _CELL_SIZE / 2),
                        _CELL_SIZE, _CELL_SIZE,
                        color="orangered", alpha=0.5, zorder=1))

        # Tasks
        for tp, ts in zip(history["task_pos"][t], history["task_status"][t]):
            color = _TASK_STATUS_COLOR.get(int(ts), "mediumpurple")
            ax.plot(tp[0], tp[1], "*", color=color, markersize=13, zorder=6)

        # Robots — dot + short trail
        n_robots = pos_arr.shape[1]
        for r in range(n_robots):
            rc    = _ROBOT_PALETTE[r]
            alive = history["alive"][t][r] > 0.5
            rx, ry = pos_arr[t, r]

            t0 = max(0, t - _TRAIL_LEN)
            ax.plot(pos_arr[t0:t + 1, r, 0], pos_arr[t0:t + 1, r, 1],
                    "-", color=rc, alpha=0.35, linewidth=1.5, zorder=4)

            if alive:
                carrying = history["carrying"][t][r] > 0.5
                marker   = "D" if carrying else "o"
                ax.plot(rx, ry, marker, color=rc, markersize=9, zorder=7)
            else:
                ax.plot(rx, ry, "x", color=rc, markersize=9,
                        markeredgewidth=2, zorder=7)

    total_frames = len(frame_index)
    print(f"\nRendering video: {total_frames} frames across "
          f"{len(all_histories)} episode(s)...")

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=total_frames, interval=1000 // fps, blit=False
    )

    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, "episodes.mp4")
    gif_path = os.path.join(out_dir, "episodes.gif")

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                        extra_args=["-pix_fmt", "yuv420p"])
        anim.save(mp4_path, writer=writer, dpi=100)
        print(f"Saved video → {mp4_path}")
    except Exception as exc:
        print(f"ffmpeg unavailable ({exc}); saving as GIF instead...")
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=80)
        print(f"Saved video → {gif_path}")

    plt.close(fig)


# ========================================================================== #
#  Main                                                                       #
# ========================================================================== #

def main():
    args = parse_args()

    do_render = args.render or args.both or not (args.render or args.plot)
    do_plot   = args.plot   or args.both or not (args.render or args.plot)

    # Locate params
    params_path = args.params or find_newest_params()
    print(f"Loading parameters from: {params_path}")

    # Build env and network
    env        = SycaBotEnvJAX()
    env_params = EnvParams()
    obs_dim    = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).shape[0]

    network = ActorCritic(
        action_dim  = action_dim,
        hidden_size = train_config["HIDDEN_SIZE"],
        activation  = train_config["ACTIVATION"],
    )
    trained_params = load_params(params_path, network, obs_dim)
    print(f"Loaded parameters.  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"Running on: {jax.devices()}")

    # Renderer (created only if rendering)
    renderer = None
    if do_render:
        from sycabot_render_jax import SycaBotRendererJAX
        renderer = SycaBotRendererJAX(screen_width=500, screen_height=1000)

    os.makedirs(args.out_dir, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)
    all_histories = []

    print(f"\nRunning {args.episodes} episode(s) "
          f"({'deterministic' if args.deterministic else 'stochastic'} policy, "
          f"max {args.max_steps} steps/ep)...\n")

    for ep in range(args.episodes):
        rng, ep_key = jax.random.split(rng)
        print(f"  Episode {ep + 1}/{args.episodes} ...", end="", flush=True)

        history = run_episode(
            network        = network,
            trained_params = trained_params,
            env            = env,
            env_params     = env_params,
            key            = ep_key,
            max_steps      = args.max_steps,
            deterministic  = args.deterministic,
            renderer       = renderer if do_render else None,
            fps            = args.fps,
        )
        all_histories.append(history)

        delivered  = max(history["delivered"])  if history["delivered"]  else 0
        contam     = max(history["contaminated"]) if history["contaminated"] else 0
        safe_final = history["safety"][-1] if history["safety"] else 0.0
        print(f"  return={history['total_reward']:7.1f}  "
              f"steps={history['length']:4d}  "
              f"delivered={delivered:.0f}  "
              f"contaminated={contam:.0f}  "
              f"safe={safe_final:.2f}")

        if do_plot:
            plot_trajectory(history, env, env_params, ep, args.out_dir)
            plot_fire_spread(history, env, env_params, ep, args.out_dir)

    if renderer is not None:
        renderer.close()

    if do_plot:
        plot_summary(all_histories, args.out_dir)

    if args.video:
        save_video(all_histories, env, env_params, args.out_dir, fps=args.fps)

    # Print aggregate stats
    print("\n=== Aggregate Stats ===")
    returns = [h["total_reward"] for h in all_histories]
    print(f"  Return:  mean={np.mean(returns):7.1f}  std={np.std(returns):6.1f}  "
          f"min={np.min(returns):7.1f}  max={np.max(returns):7.1f}")
    lengths = [h["length"] for h in all_histories]
    print(f"  Length:  mean={np.mean(lengths):5.1f}  std={np.std(lengths):5.1f}")
    delivered_ep = [max(h["delivered"]) if h["delivered"] else 0 for h in all_histories]
    print(f"  Delivered/ep: mean={np.mean(delivered_ep):.2f}")


if __name__ == "__main__":
    main()
