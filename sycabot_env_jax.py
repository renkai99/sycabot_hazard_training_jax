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
NUM_HAZARDS = 2

_X_MIN, _X_MAX = -1.55, 1.55
_Y_MIN, _Y_MAX = -3.10, 3.10
_CELL_SIZE = 0.08
GRID_X = int(np.ceil((_X_MAX - _X_MIN) / _CELL_SIZE))   # 39
GRID_Y = int(np.ceil((_Y_MAX - _Y_MIN) / _CELL_SIZE))   # 78
_MAX_INITIAL_FIRES = 10   # static upper bound; actual count set via EnvParams.num_initial_fires


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
    num_initial_fires: int = NUM_HAZARDS       # number of independent fire seeds at episode start
    pickup_reward: float = 6.0
    delivery_reward: float = 8.0
    smooth_action_weight: float = 0.006
    turn_smooth_weight: float = 0.001
    jerk_weight: float = 0.0005 
    direction_flip_weight: float = 0.007
    task_progress_reward_weight: float = 0.3
    exit_progress_reward_weight: float = 0.3
    death_penalty: float = 0.3
    cooperation_bonus_weight: float = 0.0
    all_fail_penalty: float = 1.0
    survival_reward_weight: float = 0.0
    obstacle_proximity_weight: float = 0.0
    obstacle_proximity_threshold: float = 0.20
    fire_proximity_weight: float = 0.0
    fire_proximity_threshold: float = 0.20
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

        # Valid robot spawn positions – per-exit lookup table so robots can be
        # assigned to distinct exits, preventing same-exit spawns that cause
        # immediate mutual-collision deaths.
        # Shape: (NUM_EXITS, K_max, 2), padded by repeating the first position.
        assert NUM_ROBOTS <= NUM_EXITS, "Need at least one exit per robot"
        jitter = np.linspace(-0.12, 0.12, 15)   # 15×15 = 225 candidates per exit
        spawns_per_exit = []
        for ex in exits_np:
            pts = []
            for dx in jitter:
                for dy in jitter:
                    pt = ex + np.array([dx, dy])
                    if not (_X_MIN <= pt[0] <= _X_MAX and _Y_MIN <= pt[1] <= _Y_MAX):
                        continue
                    if _min_obs_dist_np(pt) >= r * 2.0:
                        pts.append(pt.copy())
            spawns_per_exit.append(pts)
        n_per_exit = [len(pts) for pts in spawns_per_exit]
        K_max = max(n_per_exit)
        padded = np.zeros((NUM_EXITS, K_max, 2), dtype=np.float32)
        for i, pts in enumerate(spawns_per_exit):
            n = len(pts)
            padded[i, :n] = pts
            if n < K_max:                       # repeat first entry to fill padding
                padded[i, n:] = pts[0]
        self.robot_spawns_per_exit = jnp.array(padded)                     # (NUM_EXITS, K_max, 2)
        self.n_spawns_per_exit     = jnp.array(n_per_exit, dtype=jnp.int32)  # (NUM_EXITS,)

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

    def _spawn_fire(self, key, num_fires):
        # Sample _MAX_INITIAL_FIRES unique valid cells; activate only the first num_fires.
        # _MAX_INITIAL_FIRES is a Python constant so the shape is static for JAX.
        # num_fires is a traced int from EnvParams — used only in a dynamic comparison.
        n = self.valid_fire_cells.shape[0]
        indices   = jax.random.choice(key, n, shape=(_MAX_INITIAL_FIRES,), replace=False)
        flat_idxs = self.valid_fire_cells[indices]
        gx = flat_idxs // GRID_Y
        gy = flat_idxs % GRID_Y
        active = (jnp.arange(_MAX_INITIAL_FIRES) < num_fires).astype(jnp.float32)
        return jnp.zeros((GRID_X, GRID_Y), dtype=jnp.float32).at[gx, gy].set(active)

    # ------------------------------------------------------------------ #
    #  Gymnax API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        # Split into exactly 5 fixed base keys so that robot positions, headings,
        # and the fire seed are independent of NUM_TASKS (and NUM_ROBOTS).
        # Per-robot and per-task keys are then derived via fold_in(base, index),
        # which gives two important properties for Monte Carlo comparisons:
        #   1. robot_keys[r] and kfire are identical for any NUM_TASKS value.
        #   2. task_keys[t] is identical for any NUM_TASKS ≥ t+1, so adding
        #      more tasks simply appends new spawn positions (cumulative).
        perm_key, robot_base, theta_base, task_base, kfire = jax.random.split(key, 5)
        robot_keys = jax.vmap(lambda i: jax.random.fold_in(robot_base, i))(jnp.arange(NUM_ROBOTS))
        theta_keys = jax.vmap(lambda i: jax.random.fold_in(theta_base, i))(jnp.arange(NUM_ROBOTS))
        task_keys  = jax.vmap(lambda i: jax.random.fold_in(task_base,  i))(jnp.arange(NUM_TASKS))

        # Assign each robot a unique exit (no two robots share the same exit)
        exit_assign = jax.random.permutation(perm_key, NUM_EXITS)[:NUM_ROBOTS]  # (NUM_ROBOTS,)

        def sample_from_exit(exit_idx, spawn_key):
            n   = self.n_spawns_per_exit[exit_idx]
            idx = jax.random.randint(spawn_key, (), 0, n)
            return self.robot_spawns_per_exit[exit_idx, idx]

        n_t = self.valid_task_spawns.shape[0]

        robot_pos   = jax.vmap(sample_from_exit)(exit_assign, robot_keys)
        robot_theta = jax.vmap(lambda k: jax.random.uniform(k, minval=-jnp.pi, maxval=jnp.pi))(theta_keys)
        task_pos    = self.valid_task_spawns[
            jax.vmap(lambda k: jax.random.randint(k, (), 0, n_t))(task_keys)]

        fire_grid = self._spawn_fire(kfire, params.num_initial_fires)
        task_status = jnp.zeros(NUM_TASKS, dtype=jnp.int32)

        pvtd = jax.vmap(self._nearest_visible_task_dist, in_axes=(0, None, None))(
            robot_pos, task_pos, task_status)

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
            prev_visible_task_dist=pvtd,
            prev_visible_exit_dist=jnp.full(NUM_ROBOTS, jnp.nan),
            prev_joint_action=jnp.zeros(2 * NUM_ROBOTS, dtype=jnp.float32),
            prev_prev_joint_action=jnp.zeros(2 * NUM_ROBOTS, dtype=jnp.float32),
            step_count=0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        # Robot block (22 features per robot):
        #   x, y, sin(theta), cos(theta), alive, carrying,
        #   task_d, task_orient,
        #   exit_d, exit_orient,
        #   fire1_d, fire1_o, fire2_d, fire2_o, fire3_d, fire3_o,
        #   obs1_d, obs1_o, obs2_d, obs2_o, obs3_d, obs3_o
        # Task block (4 per task): x, y, status, carried
        # Global: safety_indicator, task_indicator
        # Action history: prev_joint_action (2*NUM_ROBOTS)
        robot_feats = []
        for i in range(NUM_ROBOTS):
            p  = state.robot_pos[i]
            th = state.robot_theta[i]
            obs_d,  obs_o  = self._top3_obs_dist_orient(p, th)
            fire_d, fire_o = self._top3_fire_dist_orient(p, th, state.fire_grid)
            robot_feats += [
                p[0], p[1],
                jnp.sin(th), jnp.cos(th),
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
            state.prev_joint_action,
        ])
        return jnp.nan_to_num(obs, nan=0.0)

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        fire_key, *fk_list = jax.random.split(key, 1 + NUM_ROBOTS)
        fk_keys = jnp.stack(fk_list)                                    # (NUM_ROBOTS, 2)
        action = jnp.clip(jnp.asarray(action, dtype=jnp.float32),
                          jnp.array([params.v_min, params.w_min] * NUM_ROBOTS),
                          jnp.array([params.v_max, params.w_max] * NUM_ROBOTS))
        actions_per_robot = action.reshape(NUM_ROBOTS, 2)

        # ---- 1. Motion ---- #
        def move(pos, theta, act):
            v, w = act[0], act[1]
            new_pos   = pos + jnp.array([v * jnp.cos(theta), v * jnp.sin(theta)]) * params.dt
            new_theta = (theta + w * params.dt + jnp.pi) % (2 * jnp.pi) - jnp.pi
            return new_pos, new_theta

        cand_pos, cand_theta = jax.vmap(move)(
            state.robot_pos, state.robot_theta, actions_per_robot)
        alive_mask = state.robot_alive > 0.5
        new_robot_pos   = jnp.where(alive_mask[:, None], cand_pos,   state.robot_pos)
        new_robot_theta = jnp.where(alive_mask,          cand_theta, state.robot_theta)

        # ---- 2. Departed-exit flag ---- #
        new_departed = jax.vmap(
            lambda p, d: jnp.where(self._nearest_exit_dist(p) > params.exit_departure_radius, 1.0, d)
        )(new_robot_pos, state.robot_departed_exit)

        # ---- 3. Fire propagation ---- #
        new_fire_grid = self._propagate_fire(fire_key, state.fire_grid, params)

        # ---- 4. Failure checks ---- #
        oob = jax.vmap(lambda p, a: ~self._in_bounds(p, params) & (a > 0.5))(
            new_robot_pos, state.robot_alive)
        obs_hit = jax.vmap(lambda p, a: (self._min_obs_dist(p) < params.robot_radius) & (a > 0.5))(
            new_robot_pos, state.robot_alive)

        # Mutual collision: robot i dies if within collision_distance of any other alive robot
        pair_dists = jnp.linalg.norm(
            new_robot_pos[:, None, :] - new_robot_pos[None, :, :], axis=-1)   # (N, N)
        both_alive = (state.robot_alive[:, None] > 0.5) & (state.robot_alive[None, :] > 0.5)
        not_self   = ~jnp.eye(NUM_ROBOTS, dtype=bool)
        mut = jnp.any((pair_dists < params.collision_distance) & not_self & both_alive, axis=1)

        fire_r = 0.5 * jnp.sqrt(2.0) * params.fire_cell_size + params.robot_radius
        fk = jax.vmap(lambda p, k, a:
            (self._nearest_fire_dist(p, state.fire_grid) <= fire_r) &
            (jax.random.uniform(k) < params.fire_kill_prob) & (a > 0.5)
        )(new_robot_pos, fk_keys, state.robot_alive)

        dies    = oob | obs_hit | mut | fk
        new_alive   = state.robot_alive * (1.0 - dies.astype(jnp.float32))
        any_death   = jnp.any(dies)
        new_safety  = state.global_safety_indicator * (1.0 - any_death.astype(jnp.float32))
        safety_events = jnp.sum(dies.astype(jnp.float32))

        # ---- 5. Task contamination from fire ---- #
        def fire_contam_one(task_pos_i, ts_i):
            gx = jnp.clip(
                ((task_pos_i[0] - params.x_min) / params.fire_cell_size).astype(jnp.int32),
                0, GRID_X - 1)
            gy = jnp.clip(
                ((task_pos_i[1] - params.y_min) / params.fire_cell_size).astype(jnp.int32),
                0, GRID_Y - 1)
            return (ts_i == 0) & (new_fire_grid[gx, gy] > 0)

        ts = state.task_status
        contam_mask = jax.vmap(fire_contam_one)(state.task_pos, ts)
        ts = jnp.where(contam_mask, jnp.int32(3), ts)

        # ---- 6. Drop carried tasks from dead robots ---- #
        for i in range(NUM_ROBOTS):
            match = (state.task_carrier == i) & (ts == 1) & dies[i]
            ts = jnp.where(match, jnp.int32(0), ts)
        tc = state.task_carrier

        # ---- 7. Task pickup (sequential) ---- #
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

        rc = state.robot_carrying
        picked_count = jnp.float32(0)
        for i in range(NUM_ROBOTS):
            ts, tc, rc_i, picked_i = try_pickup(i, new_robot_pos[i], new_alive[i], rc[i], ts, tc)
            rc = rc.at[i].set(rc_i)
            picked_count = picked_count + picked_i
        new_carrying = rc

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

        delivered_count = jnp.float32(0)
        for i in range(NUM_ROBOTS):
            ts, tc, rc_i, deliv_i = try_deliver(
                i, new_robot_pos[i], new_alive[i], new_carrying[i], ts, tc)
            new_carrying = new_carrying.at[i].set(rc_i)
            delivered_count = delivered_count + deliv_i

        # ---- 10. Progress reward ---- #
        def progress_i(pos, alive, carrying, prev_td, prev_ed):
            cur_td = self._nearest_visible_task_dist(pos, new_task_pos, ts)
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

        tp, ep, new_ptd, new_ped = jax.vmap(progress_i)(
            new_robot_pos, new_alive, new_carrying,
            state.prev_visible_task_dist, state.prev_visible_exit_dist)
        progress_reward = (params.task_progress_reward_weight * jnp.sum(tp) +
                           params.exit_progress_reward_weight * jnp.sum(ep))

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

        # ---- 12. Proximity penalties (obstacle + fire) ---- #
        def _proximity_pen(pos, alive):
            obs_dist  = self._min_obs_dist(pos)
            fire_dist = self._nearest_fire_dist(pos, new_fire_grid)
            obs_pen   = jnp.maximum(0.0, params.obstacle_proximity_threshold - obs_dist)
            fire_pen  = jnp.maximum(0.0, params.fire_proximity_threshold    - fire_dist)
            return alive * (params.obstacle_proximity_weight * obs_pen +
                            params.fire_proximity_weight     * fire_pen)

        proximity_penalty = -jnp.sum(
            jax.vmap(_proximity_pen)(new_robot_pos, new_alive)
        )

        # ---- 13. Reward ---- #
        survival_reward = params.survival_reward_weight * jnp.sum(new_alive)
        # Bonus for each robot that is simultaneously alive and carrying
        cooperation_bonus = params.cooperation_bonus_weight * jnp.sum(new_alive * new_carrying)
        reward = (
            -params.death_penalty * safety_events +
            params.pickup_reward   * picked_count +
            params.delivery_reward * delivered_count +
            progress_reward +
            smooth_pen +
            survival_reward +
            cooperation_bonus +
            proximity_penalty +
            -0.01
        )
        all_fail = jnp.all(new_alive < 0.5) & (safety_events > 0)
        reward = jnp.where(all_fail, -params.all_fail_penalty, reward)

        # ---- 14. Done ---- #
        all_dead        = jnp.all(new_alive < 0.5)
        no_active_tasks = jnp.all((ts == 2) | (ts == 3))
        time_up         = state.step_count + 1 >= params.max_steps
        done = all_dead | no_active_tasks | time_up

        # ---- 15. New state ---- #
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
            prev_visible_task_dist=new_ptd,
            prev_visible_exit_dist=new_ped,
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
            "proximity_penalty":  proximity_penalty,
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
        obs_dim = NUM_ROBOTS * 22 + NUM_TASKS * 4 + 2 + 2 * NUM_ROBOTS
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(obs_dim,), dtype=jnp.float32)
