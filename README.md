# SycaBot Hazard Rescue — JAX

Pure-JAX PPO implementation of the multi-robot hazard-aware rescue environment. Robots navigate a fire-spreading lab maze, pick up task items, and deliver them to exits without being destroyed. The entire pipeline — environment, rollout collection, and policy update — runs under `jax.jit` across 1024 parallel environments simultaneously.

## Stack

| Component | Library |
|-----------|---------|
| Accelerated numerics | JAX / JAXlib |
| Neural network | Flax Linen |
| Optimiser | Optax |
| Environment base class | Gymnax |
| Rendering (optional) | Pygame |

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone <repo-url>
cd sycabot_hazard_training_jax

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Rendering support (optional)
pip install pygame
```

`requirements.txt` includes: `jax`, `jaxlib`, `flax`, `optax`, `chex`, `gymnax`, `numpy`, `scipy`, `matplotlib`, `tqdm`

---

## Environment overview

The environment is a bounded 2D workspace (`~3.1 m × 6.2 m`) with static wall obstacles and 5 exits. A stochastic cellular fire spreads from a random seed cell each episode. Robots start near exits and must pick up task items from the interior and deliver them before the fire destroys them or the robots.

- 2 unicycle robots, jointly controlled via a flat action vector `[v₁, ω₁, v₂, ω₂]`
- Tasks transition: `pending → carried → delivered` (or `contaminated` if fire reaches them)
- Robots are destroyed by: boundary exit, obstacle contact, inter-robot collision, or fire proximity
- Episode ends when all robots are destroyed, all tasks are resolved, or `max_steps` is reached

The environment follows the **Gymnax decoupled API**: all state is carried explicitly in an `EnvState` pytree, making it fully compatible with `jax.vmap` and `jax.lax.scan`.

---

## Observation space

The observation is a flat float32 vector of length `N_robots × 22 + N_tasks × 4 + 2 + 2 × N_robots`.

For the default configuration (2 robots, 2 tasks) this gives **58 features**.

### Robot block — 22 features per robot (ordered by robot index)

| Feature | Description |
|---------|-------------|
| `x` | World x-position (meters) |
| `y` | World y-position (meters) |
| `sin_theta` | Sine of heading angle — smooth, bounded heading encoding |
| `cos_theta` | Cosine of heading angle — eliminates the ±π discontinuity of raw `theta` |
| `alive` | 1.0 if robot is alive, 0.0 if destroyed |
| `carrying` | 1.0 if the robot is carrying a task item, 0.0 otherwise |
| `task_d` | Euclidean distance to the nearest pending task (meters) |
| `task_orient` | Robot-relative angle to the nearest pending task (radians, −π to π) |
| `exit_d` | Euclidean distance to the nearest exit point (meters) |
| `exit_orient` | Robot-relative angle to the nearest exit (radians, −π to π) |
| `fire1_d` | Edge distance to the nearest burning cell (meters; 10.0 if no fire) |
| `fire1_o` | Robot-relative angle to the nearest burning cell (radians) |
| `fire2_d` | Edge distance to the 2nd nearest burning cell |
| `fire2_o` | Robot-relative angle to the 2nd nearest burning cell |
| `fire3_d` | Edge distance to the 3rd nearest burning cell |
| `fire3_o` | Robot-relative angle to the 3rd nearest burning cell |
| `obs1_d` | Distance to the nearest obstacle segment (meters) |
| `obs1_o` | Robot-relative angle to the nearest obstacle segment |
| `obs2_d` | Distance to the 2nd nearest obstacle segment |
| `obs2_o` | Robot-relative angle to the 2nd nearest obstacle segment |
| `obs3_d` | Distance to the 3rd nearest obstacle segment |
| `obs3_o` | Robot-relative angle to the 3rd nearest obstacle segment |

The top-3 fire and obstacle sensors replace the raw fire grid used in the Gymnasium version, giving directional local awareness at a fraction of the vector size.

### Task block — 4 features per task (ordered by task index)

| Feature | Description |
|---------|-------------|
| `x` | World x-position of the task item (meters) |
| `y` | World y-position of the task item (meters) |
| `status` | 0=pending, 1=carried, 2=delivered, 3=contaminated |
| `carried` | 1.0 if currently being carried by a robot, 0.0 otherwise |

### Global indicators — 2 scalars

| Feature | Description |
|---------|-------------|
| `global_safety_indicator` | 1.0 at episode start; flips permanently to 0.0 on the first robot death |
| `global_task_indicator` | 0.0=no task touched, 1.0=at least one picked up, 2.0=at least one delivered |

### Action history — 2 × N_robots scalars

The previous joint action `[v₁, ω₁, ..., vₙ, ωₙ]` is appended verbatim from `EnvState.prev_joint_action`. This lets the policy directly observe the motion it is being penalised for (smoothness, jerk, direction flips) rather than having to infer it implicitly from reward shaping alone.

---

## Reward components

All components are summed each step. Key weights are configurable via `EnvParams`.

| Component | Formula | Default weight | Notes |
|-----------|---------|----------------|-------|
| **Pickup reward** | `+pickup_reward × pickups_this_step` | 600 | Sparse; triggers when a robot enters ≤0.18 m of a pending task |
| **Delivery reward** | `+delivery_reward × deliveries_this_step` | 600 | Sparse; triggers when a carrying robot enters ≤0.20 m of an exit |
| **Task progress** | `+weight × Σ max(prev_visible_task_dist − curr_visible_task_dist, 0)` | 30 | Potential-based shaping; only counts when task has line-of-sight; resets when visibility is lost |
| **Exit progress** | `+weight × Σ max(prev_visible_exit_dist − curr_visible_exit_dist, 0)` | 30 | Identical logic, but for carrying robots approaching exits |
| **Safety event penalty** | `−3.0 × safety_events` | fixed | Counts every robot death in the step |
| **Action smoothness penalty** | `−smooth_action_weight × mean(Δv²)` | 1.00 | Penalises abrupt linear velocity changes |
| **Turn smoothness penalty** | `−turn_smooth_weight × mean(Δω²)` | 0.20 | Penalises abrupt angular velocity changes |
| **Jerk penalty** | `−jerk_weight × mean(jerk²)` | 0.08 | Second-order finite difference of actions |
| **Direction flip penalty** | `−direction_flip_weight × num_flips` | 0.10 | Counts robots that reverse linear direction within one step |
| **Time step penalty** | `−0.01` | fixed | Constant per-step cost to encourage task efficiency |
| **All-robots-destroyed override** | `−20.0` (replaces everything) | fixed | Triggered only if all robots die in a single step |

---

## Suggestions: improving the observation space

**1. Egocentric (robot-relative) coordinates**
The current observation uses absolute world coordinates for robot and task positions. Switching to positions expressed relative to each robot's own pose improves generalisation across spawn locations and reduces what the policy must learn implicitly.

**2. Explicit inter-robot relative positions**
Each robot's observation currently lacks direct information about where the other robot is relative to itself. Adding the relative position and heading of each teammate makes collision avoidance and coordination significantly easier to learn.

**3. Per-task fire risk**
Include for each task the distance from the task to the nearest burning cell. This lets the policy prioritise rescuing tasks that are about to be contaminated without requiring any manual priority logic.

**4. Normalised positions**
All distances and coordinates should be normalised by the workspace dimensions or a fixed scale factor to keep inputs in a consistent range (e.g., [−1, 1]). This avoids gradient scaling issues and makes the policy more robust to arena size changes.

---

## Suggestions: improving the reward structure

**1. Continuous fire proximity penalty**
The current fire penalty is purely stochastic (kill probability near fire cells). Adding a continuous shaping term like `−k / (fire_d + ε)` gives the agent a smooth gradient away from fire long before a death event occurs, making fire avoidance easier to learn.

**2. Contamination prevention bonus**
Give a small positive reward when a robot picks up a task that is within a threshold distance of an active fire cell. This rewards urgency and teaches the agent to prioritise at-risk items.

**3. Survival-per-step bonus**
A small per-step reward (+0.01 to +0.05) for each alive robot counters the tendency to sacrifice robots when expedient, and balances the time-step penalty.

**4. Collision proximity shaping**
Rather than only penalising actual deaths, add a potential-based penalty for inter-robot distance below a soft threshold (e.g., `−k × max(0, collision_distance × 2 − dist)²`). This creates a repulsive field that guides the policy away from near-collisions before they become fatal.

**5. Fire-weighted task progress**
Weight the task progress reward by the urgency of the targeted task: tasks closer to fire get a higher progress multiplier. This guides robots to rescue at-risk tasks first without requiring any additional manual priority logic.

---

## Tunable environment parameters

All parameters live in `EnvParams` and can be overridden at construction time.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.20 s | Integration timestep |
| `v_min / v_max` | ±0.20 m/s | Linear velocity bounds |
| `w_min / w_max` | ±π/6 rad/s | Angular velocity bounds |
| `robot_radius` | 0.08 m | Robot body radius; also sets obstacle contact threshold |
| `collision_distance` | 0.16 m | Inter-robot collision threshold (2 × radius) |
| `fire_spread_prob` | 0.02 | Probability per step that fire spreads to each unburned neighbour |
| `fire_kill_prob` | 0.20 | Probability per step that a robot in fire range is destroyed |
| `fire_cell_size` | 0.08 m | Fire grid resolution |
| `pickup_reward` | 600 | Reward for picking up a task item |
| `delivery_reward` | 600 | Reward for delivering a task item to an exit |
| `task_progress_reward_weight` | 30 | Weight for move-to-task shaping reward |
| `exit_progress_reward_weight` | 30 | Weight for move-to-exit shaping reward (when carrying) |
| `smooth_action_weight` | 1.00 | Weight for linear velocity smoothness penalty |
| `turn_smooth_weight` | 0.20 | Weight for angular velocity smoothness penalty |
| `jerk_weight` | 0.08 | Weight for jerk penalty |
| `direction_flip_weight` | 0.10 | Weight for direction reversal penalty |
| `pickup_radius` | 0.18 m | Distance threshold for task pickup |
| `delivery_radius` | 0.20 m | Distance threshold for task delivery |
| `exit_departure_radius` | 0.28 m | Distance past which a robot is considered to have left its spawn exit |
| `max_steps` | 1000 | Episode length cutoff |

---

## PPO hyperparameters

Configured via the `config` dict at the top of [train_ppo.py](train_ppo.py).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LR` | 3e-4 | Initial Adam learning rate |
| `LR_DECAY_START_FRAC` | 0.8 | Fraction of total updates before linear LR decay begins |
| `LR_END_FACTOR` | 0.1 | Final LR as a fraction of the initial LR |
| `UPDATE_EPOCHS` | 4 | PPO epochs per rollout |
| `MINIBATCH_SIZE` | 32768 | Transitions per minibatch |
| `GAMMA` | 0.99 | Discount factor |
| `GAE_LAMBDA` | 0.95 | GAE λ |
| `CLIP_EPS` | 0.2 | PPO clipping ε |
| `ENT_COEF` | 0.01 | Entropy bonus coefficient |
| `VF_COEF` | 0.5 | Value-function loss coefficient |
| `MAX_GRAD_NORM` | 0.5 | Global gradient norm clip |
| `CLIP_VF` | True | Whether to clip the value-function loss |
| `KL_THRESHOLD` | 0.015 | Early-stops remaining minibatches within an epoch when approximate KL exceeds this |
| `NUM_ENVS` | 1024 | Parallel environments (vmapped) |
| `NUM_STEPS` | 64 | Rollout steps per environment per update |
| `TOTAL_UPDATES` | 100 000 | Total PPO update steps |
| `HIDDEN_SIZE` | 256 | Hidden layer width for actor and critic |

Total transitions per update = `NUM_ENVS × NUM_STEPS` = 65 536.

---

## Network architecture

`ActorCritic` (Flax `nn.Module`) with **shared input, separate actor and critic towers**:

- **Actor**: Linear(256) → tanh → Linear(256) → tanh → Linear(action\_dim); a learned `log_std` parameter gives a diagonal Gaussian policy.
- **Critic**: Linear(256) → tanh → Linear(256) → tanh → Linear(1).
- Kernel initialisations: √2 orthogonal for hidden layers, 0.01 orthogonal for the actor output, 1.0 orthogonal for the critic output.

---

## Parallelism

```
VecEnv(LogWrapper(SycaBotEnvJAX))
  └─ jax.vmap over NUM_ENVS independent PRNG keys
       └─ jax.lax.scan over NUM_STEPS
            └─ jax.jit over the full _update_step
```

The entire rollout and gradient update runs on-device. CPU↔device transfers happen only at `PRINT_INTERVAL` checkpoints for logging.

---

## Run

**Train from scratch:**

```bash
python3 train_ppo.py
```

**Warm-start from the most recent checkpoint:**

```bash
python3 train_ppo.py --warm-start
```

**Warm-start from a specific checkpoint:**

```bash
python3 train_ppo.py --init-params results/<run_name>/trained_params.pkl
```

Results are saved automatically to `results/<run_name>/`:

```
results/PPO_hazard_jax_lr0.0003_envs1024_steps64_mb32768_<timestamp>/
├── hyperparameters.txt      # full config snapshot
├── trained_params.pkl       # Flax serialised parameters
├── training_metrics.png     # return, safety, alive robots, deliveries, …
└── reward_components.png    # pickup, delivery, progress, contamination
```

**Render a trained policy** using `SycaBotRendererJAX` from [sycabot_render_jax.py](sycabot_render_jax.py):

```python
import jax
from sycabot_env_jax import SycaBotEnvJAX, EnvParams
from sycabot_render_jax import SycaBotRendererJAX

env    = SycaBotEnvJAX()
params = EnvParams()
key    = jax.random.PRNGKey(0)

renderer = SycaBotRendererJAX()
key, k = jax.random.split(key)
obs, state = env.reset_env(k, params)

running = True
while running:
    action = ...  # your policy here
    key, k = jax.random.split(key)
    obs, state, reward, done, info = env.step_env(k, state, action, params)
    running = renderer.render(env, state, params, fps=30)
    if done:
        key, k = jax.random.split(key)
        obs, state = env.reset_env(k, params)

renderer.close()
```
