import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct

from environment_configs import get_lab_environment_config

NUM_ROBOTS = 2
NUM_TASKS = 2
NUM_EXITS = 5

_X_MIN, _X_MAX = -1.55, 1.55
_Y_MIN, _Y_MAX = -3.10, 3.10
_CELL_SIZE = 0.08
GRID_X = int(np.ceil((_X_MAX - _X_MIN) / _CELL_SIZE))   # 39
GRID_Y = int(np.ceil((_Y_MAX - _Y_MIN) / _CELL_SIZE))   # 78


@struct.dataclass
class EnvState:
    robot_pos: chex.Array             # (NUM_ROBOTS, 2)
    robot_theta: chex.Array           # (NUM_ROBOTS,)
    robot_alive: chex.Array           # (NUM_ROBOTS,)  float32
    robot_carrying: chex.Array        # (NUM_ROBOTS,)  float32
    robot_departed_exit: chex.Array   # (NUM_ROBOTS,)  float32
    task_pos: chex.Array              # (NUM_TASKS, 2)
    task_status: chex.Array           # (NUM_TASKS,)   int32  0=pending 1=carried 2=delivered 3=contaminated
    task_carrier: chex.Array          # (NUM_TASKS,)   int32  -1 = none
    fire_grid: chex.Array             # (GRID_X, GRID_Y)  float32
    global_safety_indicator: float
    prev_visible_task_dist: chex.Array  # (NUM_ROBOTS,)
    prev_visible_exit_dist: chex.Array  # (NUM_ROBOTS,)
    prev_joint_action: chex.Array       # (2*NUM_ROBOTS,)
    prev_prev_joint_action: chex.Array  # (2*NUM_ROBOTS,)
    step_count: int


@struct.dataclass
class EnvParams:
    x_min: float = _X_MIN
    x_max: float = _X_MAX
    y_min: float = _Y_MIN
    y_max: float = _Y_MAX
    dt: float = 0.2
    v_min: float = -0.20
    v_max: float = 0.20
    w_min: float = -np.pi / 6.0
    w_max: float = np.pi / 6.0
    robot_radius: float = 0.08
    collision_distance: float = 0.16   # 2 * robot_radius
    fire_spread_prob: float = 0.020
    fire_kill_prob: float = 0.2
    fire_cell_size: float = _CELL_SIZE
    pickup_reward: float = 600.0
    delivery_reward: float = 600.0
    smooth_action_weight: float = 1.0
    turn_smooth_weight: float = 0.20
    jerk_weight: float = 0.08
    direction_flip_weight: float = 0.10
    task_progress_reward_weight: float = 30.0
    exit_progress_reward_weight: float = 30.0
    max_steps: int = 1000
    pickup_radius: float = 0.18
    delivery_radius: float = 0.20
    exit_departure_radius: float = 0.28


class SycaBotEnvJAX(environment.Environment):

    def __init__(self):
        super().__init__()
        config = get_lab_environment_config()

        obstacles_raw = config["obstacles"]
        self.obs_start = jnp.array([o[0] for o in obstacles_raw], dtype=jnp.float32)  # (N,2)
        self.obs_end = jnp.array([o[1] for o in obstacles_raw], dtype=jnp.float32)    # (N,2)
        self.exits = jnp.array(config["exits"], dtype=jnp.float32)                    # (5,2)

        # Precompute grid cell centres  (GRID_X, GRID_Y, 2)
        cx = _X_MIN + (jnp.arange(GRID_X) + 0.5) * _CELL_SIZE
        cy = _Y_MIN + (jnp.arange(GRID_Y) + 0.5) * _CELL_SIZE
        GX, GY = jnp.meshgrid(cx, cy, indexing="ij")
        self.grid_centers = jnp.stack([GX, GY], axis=-1)  # (GRID_X, GRID_Y, 2)

        # All precomputations run in NumPy once at init (before any JAX tracing).
        obs_start_np = np.array(self.obs_start)
        obs_end_np   = np.array(self.obs_end)
        grid_np      = np.array(self.grid_centers)
        exits_np     = np.array(self.exits)
        r = 0.08  # robot_radius

        def _min_obs_dist_np(p):
            return min(self._seg_dist_np(p, s, e)
                       for s, e in zip(obs_start_np, obs_end_np))

        # Non-obstacle mask (GRID_X, GRID_Y)
        mask = np.ones((GRID_X, GRID_Y), dtype=bool)
        for gx in range(GRID_X):
            for gy in range(GRID_Y):
                pt = grid_np[gx, gy]
                for s, e in zip(obs_start_np, obs_end_np):
                    if self._seg_dist_np(pt, s, e) < r:
                        mask[gx, gy] = False
                        break
        self.non_obstacle_mask = jnp.array(mask, dtype=jnp.float32)

        # Valid robot spawn positions – dense grid of jitter offsets around each exit,
        # filtered for obstacle clearance.  Replaces the while_loop in _sample_near_exit.
        jitter = np.linspace(-0.12, 0.12, 15)   # 15×15 = 225 candidates per exit
        robot_spawns = []
        for ex in exits_np:
            for dx in jitter:
                for dy in jitter:
                    pt = ex + np.array([dx, dy])
                    if not (_X_MIN <= pt[0] <= _X_MAX and _Y_MIN <= pt[1] <= _Y_MAX):
                        continue
                    if _min_obs_dist_np(pt) >= r * 2.0:
                        robot_spawns.append(pt.copy())
        self.valid_robot_spawns = jnp.array(robot_spawns, dtype=jnp.float32)  # (N, 2)

        # Valid task spawn positions – regular grid across the arena, filtered for
        # obstacle clearance and minimum exit distance.  Replaces _sample_task_pos.
        task_spawns = []
        for x in np.arange(_X_MIN + 0.15, _X_MAX - 0.15, 0.10):
            for y in np.arange(_Y_MIN + 0.15, _Y_MAX - 0.15, 0.10):
                pt = np.array([x, y])
                if _min_obs_dist_np(pt) < r:
                    continue
                if np.min(np.linalg.norm(exits_np - pt, axis=1)) < 0.35:
                    continue
                task_spawns.append(pt.copy())
        self.valid_task_spawns = jnp.array(task_spawns, dtype=jnp.float32)    # (M, 2)

        # Flat indices of non-obstacle grid cells for fire spawning.
        self.valid_fire_cells = jnp.array(
            np.where(mask.reshape(-1))[0], dtype=jnp.int32)                   # (K,)

    # ------------------------------------------------------------------ #
    #  NumPy helper (init only)                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _seg_dist_np(p, a, b):
        ab = b - a
        denom = np.dot(ab, ab)
        if denom < 1e-9:
            return float(np.linalg.norm(p - a))
        t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * ab)))

    # ------------------------------------------------------------------ #
    #  JAX geometry primitives                                            #
    # ------------------------------------------------------------------ #

    def _seg_dist(self, p, a, b):
        ab = b - a
        denom = jnp.maximum(jnp.dot(ab, ab), 1e-9)
        t = jnp.clip(jnp.dot(p - a, ab) / denom, 0.0, 1.0)
        return jnp.linalg.norm(p - (a + t * ab))

    def _min_obs_dist(self, pos):
        dists = jax.vmap(self._seg_dist, in_axes=(None, 0, 0))(pos, self.obs_start, self.obs_end)
        return jnp.min(dists)

    def _in_bounds(self, pos, params):
        return ((pos[0] >= params.x_min) & (pos[0] <= params.x_max) &
                (pos[1] >= params.y_min) & (pos[1] <= params.y_max))

    def _ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _segs_intersect(self, p1, p2, q1, q2):
        return (self._ccw(p1, q1, q2) != self._ccw(p2, q1, q2)) & \
               (self._ccw(p1, p2, q1) != self._ccw(p1, p2, q2))

    def _has_los(self, start, end):
        hits = jax.vmap(self._segs_intersect, in_axes=(None, None, 0, 0))(
            start, end, self.obs_start, self.obs_end)
        return ~jnp.any(hits)

    def _nearest_exit_dist(self, pos):
        return jnp.min(jnp.linalg.norm(self.exits - pos, axis=1))

    def _nearest_task_dist(self, pos, task_pos, task_status):
        pending = task_status == 0
        dists = jnp.linalg.norm(task_pos - pos, axis=1)
        masked = jnp.where(pending, dists, jnp.inf)
        return jnp.where(jnp.all(~pending), 0.0, jnp.min(masked))

    def _nearest_visible_task_dist(self, pos, task_pos, task_status):
        pending = task_status == 0
        dists = jnp.linalg.norm(task_pos - pos, axis=1)
        visible = jax.vmap(self._has_los, in_axes=(None, 0))(pos, task_pos)
        masked = jnp.where(pending & visible, dists, jnp.inf)
        min_d = jnp.min(masked)
        return jnp.where(jnp.isinf(min_d), jnp.nan, min_d)

    def _nearest_visible_exit_dist(self, pos):
        dists = jnp.linalg.norm(self.exits - pos, axis=1)
        visible = jax.vmap(self._has_los, in_axes=(None, 0))(pos, self.exits)
        masked = jnp.where(visible, dists, jnp.inf)
        min_d = jnp.min(masked)
        return jnp.where(jnp.isinf(min_d), jnp.nan, min_d)

    def _nearest_fire_dist(self, pos, fire_grid):
        flat_fire = fire_grid.reshape(-1) > 0
        flat_centers = self.grid_centers.reshape(-1, 2)
        dists = jnp.linalg.norm(flat_centers - pos, axis=1)
        masked = jnp.where(flat_fire, dists, jnp.inf)
        min_d = jnp.min(masked)
        return jnp.where(jnp.isinf(min_d), 10.0, min_d)

    def _top3_obs_dist_orient(self, pos, theta):
        """Distances and robot-relative orientations to the 3 closest obstacle segments."""
        wall_dirs = self.obs_end - self.obs_start                                   # (N, 2)
        norm_sq   = jnp.maximum(jnp.sum(jnp.square(wall_dirs), axis=1), 1e-9)
        t         = jnp.clip(
            jnp.sum((pos - self.obs_start) * wall_dirs, axis=1) / norm_sq, 0.0, 1.0)
        closest   = self.obs_start + t[:, jnp.newaxis] * wall_dirs                 # (N, 2)
        dists     = jnp.linalg.norm(pos - closest, axis=1)                         # (N,)
        _, idx3   = jax.lax.top_k(-dists, k=3)
        top_dists = dists[idx3]                                                     # (3,)
        vecs      = closest[idx3] - pos                                             # (3, 2)
        angles    = jnp.arctan2(vecs[:, 1], vecs[:, 0])
        orients   = (angles - theta + jnp.pi) % (2 * jnp.pi) - jnp.pi            # (3,)
        return top_dists, orients

    def _task_orientation(self, pos, theta, task_pos, task_status):
        """Robot-relative orientation to the nearest pending task."""
        pending = task_status == 0
        dists   = jnp.linalg.norm(task_pos - pos, axis=1)
        nearest = task_pos[jnp.argmin(jnp.where(pending, dists, jnp.inf))]
        angle   = jnp.arctan2(nearest[1] - pos[1], nearest[0] - pos[0])
        return (angle - theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    def _exit_orientation(self, pos, theta):
        """Robot-relative orientation to the nearest exit."""
        nearest = self.exits[jnp.argmin(jnp.linalg.norm(self.exits - pos, axis=1))]
        angle   = jnp.arctan2(nearest[1] - pos[1], nearest[0] - pos[0])
        return (angle - theta + jnp.pi) % (2 * jnp.pi) - jnp.pi

    def _top3_fire_dist_orient(self, pos, theta, fire_grid):
        """Distances to edge and relative orientations of the 3 nearest burning cells.

        Distance is to the cell edge (centre distance minus half cell-size), clamped to 0.
        Returns 10.0 / 0.0 for slots where fewer than 3 cells are burning.
        """
        flat_fire    = fire_grid.reshape(-1) > 0                          # (GRID_X*GRID_Y,)
        flat_centers = self.grid_centers.reshape(-1, 2)                   # (GRID_X*GRID_Y, 2)

        dists_center = jnp.linalg.norm(flat_centers - pos, axis=1)
        dists_edge   = jnp.maximum(0.0, dists_center - _CELL_SIZE / 2.0) # approx edge dist
        masked       = jnp.where(flat_fire, dists_edge, jnp.inf)

        _, idx3      = jax.lax.top_k(-masked, k=3)
        top_dists    = jnp.where(jnp.isinf(masked[idx3]), 10.0, dists_edge[idx3])

        vecs         = flat_centers[idx3] - pos
        angles       = jnp.arctan2(vecs[:, 1], vecs[:, 0])
        orients      = (angles - theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        # Zero orientation when the slot has no real fire cell behind it
        orients      = jnp.where(jnp.isinf(masked[idx3]), 0.0, orients)

        return top_dists, orients

    # ------------------------------------------------------------------ #
    #  Fire dynamics                                                      #
    # ------------------------------------------------------------------ #

    def _propagate_fire(self, key, fire_grid, params):
        left  = jnp.roll(fire_grid, 1, axis=0).at[0, :].set(0.0)
        right = jnp.roll(fire_grid, -1, axis=0).at[-1, :].set(0.0)
        up    = jnp.roll(fire_grid, 1, axis=1).at[:, 0].set(0.0)
        down  = jnp.roll(fire_grid, -1, axis=1).at[:, -1].set(0.0)
        has_burning_nb = (left + right + up + down) > 0
        spreads = jax.random.uniform(key, (GRID_X, GRID_Y)) < params.fire_spread_prob
        new_cells = has_burning_nb & (fire_grid == 0) & spreads & (self.non_obstacle_mask > 0)
        return jnp.maximum(fire_grid, new_cells.astype(jnp.float32))

    def _spawn_fire(self, key):
        # O(1) table lookup – no while_loop
        n = self.valid_fire_cells.shape[0]
        flat_idx = self.valid_fire_cells[jax.random.randint(key, (), 0, n)]
        return (jnp.zeros((GRID_X, GRID_Y), dtype=jnp.float32)
                .at[flat_idx // GRID_Y, flat_idx % GRID_Y].set(1.0))

    # ------------------------------------------------------------------ #
    #  Gymnax API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        key, k1, k2, k3, k4, k5, k6, kfire = jax.random.split(key, 8)

        # O(1) table lookups – no while_loops, safe to vmap over thousands of envs
        n_r = self.valid_robot_spawns.shape[0]
        n_t = self.valid_task_spawns.shape[0]

        pos0   = self.valid_robot_spawns[jax.random.randint(k1, (), 0, n_r)]
        pos1   = self.valid_robot_spawns[jax.random.randint(k2, (), 0, n_r)]
        theta0 = jax.random.uniform(k3, minval=-jnp.pi, maxval=jnp.pi)
        theta1 = jax.random.uniform(k4, minval=-jnp.pi, maxval=jnp.pi)

        robot_pos   = jnp.stack([pos0, pos1])
        robot_theta = jnp.array([theta0, theta1])

        task0    = self.valid_task_spawns[jax.random.randint(k5, (), 0, n_t)]
        task1    = self.valid_task_spawns[jax.random.randint(k6, (), 0, n_t)]
        task_pos = jnp.stack([task0, task1])

        fire_grid = self._spawn_fire(kfire)
        task_status = jnp.zeros(NUM_TASKS, dtype=jnp.int32)

        # Initial visible distances for progress reward baseline
        pvtd0 = self._nearest_visible_task_dist(pos0, task_pos, task_status)
        pvtd1 = self._nearest_visible_task_dist(pos1, task_pos, task_status)

        state = EnvState(
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            robot_alive=jnp.ones(NUM_ROBOTS, dtype=jnp.float32),
            robot_carrying=jnp.zeros(NUM_ROBOTS, dtype=jnp.float32),
            robot_departed_exit=jnp.zeros(NUM_ROBOTS, dtype=jnp.float32),
            task_pos=task_pos,
            task_status=task_status,
            task_carrier=jnp.full(NUM_TASKS, -1, dtype=jnp.int32),
            fire_grid=fire_grid,
            global_safety_indicator=1.0,
            prev_visible_task_dist=jnp.array([pvtd0, pvtd1]),
            prev_visible_exit_dist=jnp.full(NUM_ROBOTS, jnp.nan),
            prev_joint_action=jnp.zeros(2 * NUM_ROBOTS, dtype=jnp.float32),
            prev_prev_joint_action=jnp.zeros(2 * NUM_ROBOTS, dtype=jnp.float32),
            step_count=0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        # Robot block (21 features per robot):
        #   x, y, theta, alive, carrying,
        #   task_d, task_orient,
        #   exit_d, exit_orient,
        #   fire1_d, fire1_o, fire2_d, fire2_o, fire3_d, fire3_o,
        #   obs1_d, obs1_o, obs2_d, obs2_o, obs3_d, obs3_o
        robot_feats = []
        for i in range(NUM_ROBOTS):
            p  = state.robot_pos[i]
            th = state.robot_theta[i]
            obs_d,  obs_o  = self._top3_obs_dist_orient(p, th)
            fire_d, fire_o = self._top3_fire_dist_orient(p, th, state.fire_grid)
            robot_feats += [
                p[0], p[1],
                th,
                state.robot_alive[i],
                state.robot_carrying[i],
                self._nearest_task_dist(p, state.task_pos, state.task_status),
                self._task_orientation(p, th, state.task_pos, state.task_status),
                self._nearest_exit_dist(p),
                self._exit_orientation(p, th),
                fire_d[0], fire_o[0],
                fire_d[1], fire_o[1],
                fire_d[2], fire_o[2],
                obs_d[0],  obs_o[0],
                obs_d[1],  obs_o[1],
                obs_d[2],  obs_o[2],
            ]

        # Task block: x, y, status, carried  (4 per task)
        task_feats = []
        for i in range(NUM_TASKS):
            task_feats += [
                state.task_pos[i, 0],
                state.task_pos[i, 1],
                state.task_status[i].astype(jnp.float32),
                (state.task_status[i] == 1).astype(jnp.float32),
            ]

        any_carried   = jnp.any(state.task_status == 1)
        any_delivered = jnp.any(state.task_status == 2)
        task_indicator = jnp.where(any_delivered, 2.0, jnp.where(any_carried, 1.0, 0.0))

        obs = jnp.concatenate([
            jnp.array(robot_feats, dtype=jnp.float32),
            jnp.array(task_feats, dtype=jnp.float32),
            jnp.array([state.global_safety_indicator, task_indicator]),
        ])
        return jnp.nan_to_num(obs, nan=0.0)

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        key, k_fire, k_fk0, k_fk1 = jax.random.split(key, 4)
        action = jnp.clip(jnp.asarray(action, dtype=jnp.float32),
                          jnp.array([params.v_min, params.w_min] * NUM_ROBOTS),
                          jnp.array([params.v_max, params.w_max] * NUM_ROBOTS))

        # ---- 1. Motion ---- #
        def move(pos, theta, act):
            v, w = act[0], act[1]
            new_pos = pos + jnp.array([v * jnp.cos(theta), v * jnp.sin(theta)]) * params.dt
            new_theta = (theta + w * params.dt + jnp.pi) % (2 * jnp.pi) - jnp.pi
            return new_pos, new_theta

        new_pos0, new_theta0 = move(state.robot_pos[0], state.robot_theta[0], action[0:2])
        new_pos1, new_theta1 = move(state.robot_pos[1], state.robot_theta[1], action[2:4])
        new_pos0 = jnp.where(state.robot_alive[0] > 0.5, new_pos0, state.robot_pos[0])
        new_theta0 = jnp.where(state.robot_alive[0] > 0.5, new_theta0, state.robot_theta[0])
        new_pos1 = jnp.where(state.robot_alive[1] > 0.5, new_pos1, state.robot_pos[1])
        new_theta1 = jnp.where(state.robot_alive[1] > 0.5, new_theta1, state.robot_theta[1])
        new_robot_pos = jnp.stack([new_pos0, new_pos1])
        new_robot_theta = jnp.array([new_theta0, new_theta1])

        # ---- 2. Departed-exit flag ---- #
        dep0 = jnp.where(self._nearest_exit_dist(new_pos0) > params.exit_departure_radius,
                         1.0, state.robot_departed_exit[0])
        dep1 = jnp.where(self._nearest_exit_dist(new_pos1) > params.exit_departure_radius,
                         1.0, state.robot_departed_exit[1])
        new_departed = jnp.array([dep0, dep1])

        # ---- 3. Fire propagation ---- #
        new_fire_grid = self._propagate_fire(k_fire, state.fire_grid, params)

        # ---- 4. Failure checks ---- #
        oob0  = ~self._in_bounds(new_pos0, params) & (state.robot_alive[0] > 0.5)
        oob1  = ~self._in_bounds(new_pos1, params) & (state.robot_alive[1] > 0.5)
        obs0  = (self._min_obs_dist(new_pos0) < params.robot_radius) & (state.robot_alive[0] > 0.5)
        obs1  = (self._min_obs_dist(new_pos1) < params.robot_radius) & (state.robot_alive[1] > 0.5)
        mut   = (jnp.linalg.norm(new_pos0 - new_pos1) < params.collision_distance) & \
                (state.robot_alive[0] > 0.5) & (state.robot_alive[1] > 0.5)

        fire_r = 0.5 * jnp.sqrt(2.0) * params.fire_cell_size + params.robot_radius
        fk0 = (self._nearest_fire_dist(new_pos0, state.fire_grid) <= fire_r) & \
              (jax.random.uniform(k_fk0) < params.fire_kill_prob) & (state.robot_alive[0] > 0.5)
        fk1 = (self._nearest_fire_dist(new_pos1, state.fire_grid) <= fire_r) & \
              (jax.random.uniform(k_fk1) < params.fire_kill_prob) & (state.robot_alive[1] > 0.5)

        dies0 = oob0 | obs0 | mut | fk0
        dies1 = oob1 | obs1 | mut | fk1
        new_alive0 = state.robot_alive[0] * (1.0 - dies0.astype(jnp.float32))
        new_alive1 = state.robot_alive[1] * (1.0 - dies1.astype(jnp.float32))
        new_alive = jnp.array([new_alive0, new_alive1])

        any_death = dies0 | dies1
        new_safety = state.global_safety_indicator * (1.0 - any_death.astype(jnp.float32))
        safety_events = dies0.astype(jnp.float32) + dies1.astype(jnp.float32)

        # ---- 5. Task contamination from fire ---- #
        def fire_contam(task_idx, ts):
            gx = jnp.clip(
                ((state.task_pos[task_idx, 0] - params.x_min) / params.fire_cell_size).astype(jnp.int32),
                0, GRID_X - 1)
            gy = jnp.clip(
                ((state.task_pos[task_idx, 1] - params.y_min) / params.fire_cell_size).astype(jnp.int32),
                0, GRID_Y - 1)
            return (ts[task_idx] == 0) & (new_fire_grid[gx, gy] > 0)

        ts = state.task_status
        ts = ts.at[0].set(jnp.where(fire_contam(0, ts), jnp.int32(3), ts[0]))
        ts = ts.at[1].set(jnp.where(fire_contam(1, ts), jnp.int32(3), ts[1]))

        # ---- 6. Drop carried tasks from dead robots ---- #
        def drop_on_death(ts, tc, robot_idx, died):
            match = (tc == robot_idx) & (ts == 1) & died
            return jnp.where(match, jnp.int32(3), ts)

        ts = drop_on_death(ts, state.task_carrier, 0, dies0)
        ts = drop_on_death(ts, state.task_carrier, 1, dies1)
        tc = state.task_carrier  # will update below

        # ---- 7. Task pickup (sequential: robot 0 then robot 1) ---- #
        def try_pickup(robot_idx, pos, alive, carrying, task_status, task_carrier):
            can = (alive > 0.5) & (carrying < 0.5)
            dists = jnp.linalg.norm(state.task_pos - pos, axis=1)
            masked = jnp.where(task_status == 0, dists, jnp.inf)
            best = jnp.argmin(masked)
            near_enough = masked[best] < params.pickup_radius
            does_pickup = can & near_enough
            mask_t = jnp.arange(NUM_TASKS) == best
            new_ts = jnp.where(does_pickup & mask_t, jnp.int32(1), task_status)
            new_tc = jnp.where(does_pickup & mask_t, jnp.int32(robot_idx), task_carrier)
            new_carry = jnp.where(does_pickup, 1.0, carrying)
            return new_ts, new_tc, new_carry, does_pickup.astype(jnp.float32)

        rc0 = state.robot_carrying[0]
        rc1 = state.robot_carrying[1]
        ts, tc, rc0, picked0 = try_pickup(0, new_pos0, new_alive0, rc0, ts, tc)
        ts, tc, rc1, picked1 = try_pickup(1, new_pos1, new_alive1, rc1, ts, tc)
        new_carrying = jnp.array([rc0, rc1])
        picked_count = picked0 + picked1

        # ---- 8. Update task positions (follow carrier) ---- #
        safe_tc = jnp.clip(tc, 0, NUM_ROBOTS - 1)
        new_task_pos = jnp.stack([
            jnp.where(ts[i] == 1, new_robot_pos[safe_tc[i]], state.task_pos[i])
            for i in range(NUM_TASKS)
        ])

        # ---- 9. Task delivery (sequential) ---- #
        def try_deliver(robot_idx, pos, alive, carrying, task_status, task_carrier):
            near_exit = self._nearest_exit_dist(pos) < params.delivery_radius
            can = (alive > 0.5) & (carrying > 0.5) & near_exit
            match = (task_carrier == robot_idx) & (task_status == 1) & can
            new_ts = jnp.where(match, jnp.int32(2), task_status)
            new_tc = jnp.where(match, jnp.int32(-1), task_carrier)
            new_carry = jnp.where(can, 0.0, carrying)
            return new_ts, new_tc, new_carry, can.astype(jnp.float32)

        ts, tc, rc0, deliv0 = try_deliver(0, new_pos0, new_alive0, new_carrying[0], ts, tc)
        ts, tc, rc1, deliv1 = try_deliver(1, new_pos1, new_alive1, new_carrying[1], ts, tc)
        new_carrying = jnp.array([rc0, rc1])
        delivered_count = deliv0 + deliv1

        # ---- 10. Progress reward ---- #
        def progress_i(pos, alive, carrying, prev_td, prev_ed, task_pos, task_status):
            cur_td = self._nearest_visible_task_dist(pos, task_pos, task_status)
            cur_ed = self._nearest_visible_exit_dist(pos)
            task_prog = jnp.where(
                (alive > 0.5) & (carrying < 0.5) & jnp.isfinite(prev_td) & jnp.isfinite(cur_td),
                jnp.maximum(prev_td - cur_td, 0.0), 0.0)
            exit_prog = jnp.where(
                (alive > 0.5) & (carrying > 0.5) & jnp.isfinite(prev_ed) & jnp.isfinite(cur_ed),
                jnp.maximum(prev_ed - cur_ed, 0.0), 0.0)
            new_ptd = jnp.where(alive > 0.5,
                                jnp.where(carrying < 0.5, cur_td, jnp.nan), jnp.nan)
            new_ped = jnp.where(alive > 0.5,
                                jnp.where(carrying > 0.5, cur_ed, jnp.nan), jnp.nan)
            return task_prog, exit_prog, new_ptd, new_ped

        tp0, ep0, new_ptd0, new_ped0 = progress_i(
            new_pos0, new_alive0, new_carrying[0],
            state.prev_visible_task_dist[0], state.prev_visible_exit_dist[0], new_task_pos, ts)
        tp1, ep1, new_ptd1, new_ped1 = progress_i(
            new_pos1, new_alive1, new_carrying[1],
            state.prev_visible_task_dist[1], state.prev_visible_exit_dist[1], new_task_pos, ts)

        progress_reward = (params.task_progress_reward_weight * (tp0 + tp1) +
                           params.exit_progress_reward_weight * (ep0 + ep1))

        # ---- 11. Smoothness penalty ---- #
        v_curr = action[0::2]; w_curr = action[1::2]
        v_prev = state.prev_joint_action[0::2]; w_prev = state.prev_joint_action[1::2]
        smooth_pen = -(
            params.smooth_action_weight * jnp.mean((v_curr - v_prev) ** 2) +
            params.turn_smooth_weight   * jnp.mean((w_curr - w_prev) ** 2) +
            params.jerk_weight * jnp.mean(
                (action - 2.0 * state.prev_joint_action + state.prev_prev_joint_action) ** 2) +
            params.direction_flip_weight * jnp.sum(
                ((v_curr * v_prev < 0.0) &
                 (jnp.minimum(jnp.abs(v_curr), jnp.abs(v_prev)) > 0.03)).astype(jnp.float32))
        )

        # ---- 12. Reward ---- #
        reward = (
            -3.0 * safety_events +
            params.pickup_reward   * picked_count +
            params.delivery_reward * delivered_count +
            progress_reward +
            smooth_pen +
            -0.01
        )
        # All-robot failure override
        all_fail = jnp.all(new_alive < 0.5) & (safety_events > 0)
        reward = jnp.where(all_fail, -20.0, reward)

        # ---- 13. Done ---- #
        all_dead       = jnp.all(new_alive < 0.5)
        no_active_tasks = jnp.all((ts == 2) | (ts == 3))  # all delivered or contaminated
        time_up        = state.step_count + 1 >= params.max_steps
        done = all_dead | no_active_tasks | time_up

        # ---- 14. New state ---- #
        new_state = EnvState(
            robot_pos=new_robot_pos,
            robot_theta=new_robot_theta,
            robot_alive=new_alive,
            robot_carrying=new_carrying,
            robot_departed_exit=new_departed,
            task_pos=new_task_pos,
            task_status=ts,
            task_carrier=tc,
            fire_grid=new_fire_grid,
            global_safety_indicator=new_safety,
            prev_visible_task_dist=jnp.array([new_ptd0, new_ptd1]),
            prev_visible_exit_dist=jnp.array([new_ped0, new_ped1]),
            prev_joint_action=action,
            prev_prev_joint_action=state.prev_joint_action,
            step_count=state.step_count + 1,
        )

        obs = self.get_obs(new_state, params)
        info = {
            "safety_indicator":   new_safety,
            "reward_progress":    progress_reward,
            "reward_pickup":      params.pickup_reward * picked_count,
            "reward_delivery":    params.delivery_reward * delivered_count,
            "smooth_penalty":     smooth_pen,
            "alive_robots":       jnp.sum(new_alive > 0.5),
            "delivered_tasks":    jnp.sum(ts == 2).astype(jnp.float32),
            "contaminated_tasks": jnp.sum(ts == 3).astype(jnp.float32),
        }
        return obs, new_state, reward, done, info

    # ------------------------------------------------------------------ #
    #  Gymnax boilerplate                                                 #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "SycaBotHazard-JAX-v0"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        lo = np.array([params.v_min, params.w_min] * NUM_ROBOTS, dtype=np.float32) \
             if params else np.full(2 * NUM_ROBOTS, -1.0, dtype=np.float32)
        hi = -lo
        return spaces.Box(low=lo, high=hi, shape=(2 * NUM_ROBOTS,), dtype=jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        obs_dim = NUM_ROBOTS * 21 + NUM_TASKS * 4 + 2
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(obs_dim,), dtype=jnp.float32)
