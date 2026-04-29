"""Pygame renderer for the JAX SycaBot hazard environment.

Works with the decoupled (env, state, params) API used by SycaBotEnvJAX.
All JAX arrays are converted to NumPy before touching pygame.
"""

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class SycaBotRendererJAX:
    """Real-time pygame renderer.

    Usage::

        renderer = SycaBotRendererJAX()
        obs, state = env.reset_env(key, params)
        while True:
            renderer.render(env, state, params)
            action = ...
            obs, state, reward, done, info = env.step_env(key, state, action, params)
    """

    def __init__(self, screen_width: int = 500, screen_height: int = 1000):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for rendering.  pip install pygame")
        self.screen_width  = screen_width
        self.screen_height = screen_height
        self.window = None
        self.clock  = None

    # ------------------------------------------------------------------ #
    #  Coordinate helpers                                                  #
    # ------------------------------------------------------------------ #

    def _to_screen(self, x, y, params):
        sx = int((x - params.x_min) / (params.x_max - params.x_min) * self.screen_width)
        sy = int((params.y_max - y) / (params.y_max - params.y_min) * self.screen_height)
        return sx, sy

    def _cell_px(self, params):
        w = max(1, int(params.fire_cell_size / (params.x_max - params.x_min) * self.screen_width))
        h = max(1, int(params.fire_cell_size / (params.y_max - params.y_min) * self.screen_height))
        return w, h

    # ------------------------------------------------------------------ #
    #  Drawing helpers                                                     #
    # ------------------------------------------------------------------ #

    def _draw_fire_cell(self, center_px, cell_w, cell_h):
        px, py = center_px
        rect = pygame.Rect(px - cell_w // 2, py - cell_h // 2, cell_w, cell_h)
        pygame.draw.rect(self.window, (185, 35, 20), rect)
        rng = np.random
        for _ in range(16):
            rx = int(rng.randint(rect.left, rect.right + 1))
            ry = int(rng.randint(rect.top,  rect.bottom + 1))
            color  = (255, 70 + int(rng.randint(0, 40)), 0)
            radius = int(rng.randint(1, 3))
            pygame.draw.circle(self.window, color, (rx, ry), radius)

    def _draw_star(self, center, radius, color):
        cx, cy = center
        inner  = radius * 0.45
        points = []
        for i in range(10):
            angle = -np.pi / 2.0 + i * np.pi / 5.0
            r = radius if i % 2 == 0 else inner
            points.append((int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))))
        pygame.draw.polygon(self.window, color, points)

    def _draw_triangle(self, center, size, color):
        cx, cy = center
        pts = [
            (cx, cy - size),
            (cx - int(0.85 * size), cy + int(0.65 * size)),
            (cx + int(0.85 * size), cy + int(0.65 * size)),
        ]
        pygame.draw.polygon(self.window, color, pts)

    def _draw_arrow(self, center, theta, color, length=16, tip=6):
        cx, cy   = center
        head     = (int(cx + length * np.cos(theta)), int(cy - length * np.sin(theta)))
        left     = (int(head[0] - tip * np.cos(theta - np.pi / 6)),
                    int(head[1] + tip * np.sin(theta - np.pi / 6)))
        right    = (int(head[0] - tip * np.cos(theta + np.pi / 6)),
                    int(head[1] + tip * np.sin(theta + np.pi / 6)))
        pygame.draw.line(self.window, color, center, head, 3)
        pygame.draw.line(self.window, color, head, left,  3)
        pygame.draw.line(self.window, color, head, right, 3)

    # ------------------------------------------------------------------ #
    #  Public render method                                                #
    # ------------------------------------------------------------------ #

    def render(self, env, state, params, fps: int = 30):
        """Render one frame.  Pumps pygame events internally."""
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("SycaBot Hazard – JAX")
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Convert JAX arrays to NumPy once
        robot_pos     = np.array(state.robot_pos)
        robot_theta   = np.array(state.robot_theta)
        robot_alive   = np.array(state.robot_alive)
        robot_carrying = np.array(state.robot_carrying)
        task_pos      = np.array(state.task_pos)
        task_status   = np.array(state.task_status)
        fire_grid     = np.array(state.fire_grid)
        exits_np      = np.array(env.exits)
        obs_start_np  = np.array(env.obs_start)
        obs_end_np    = np.array(env.obs_end)

        p = params  # shorthand

        self.window.fill((245, 245, 245))

        # --- Fire grid --- #
        cell_w, cell_h = self._cell_px(p)
        from sycabot_env_jax import GRID_X, GRID_Y, _X_MIN, _Y_MIN, _CELL_SIZE
        for gx in range(GRID_X):
            for gy in range(GRID_Y):
                if fire_grid[gx, gy] <= 0:
                    continue
                cx = _X_MIN + (gx + 0.5) * _CELL_SIZE
                cy = _Y_MIN + (gy + 0.5) * _CELL_SIZE
                self._draw_fire_cell(self._to_screen(cx, cy, p), cell_w, cell_h)

        # --- Obstacles --- #
        for s, e in zip(obs_start_np, obs_end_np):
            pygame.draw.line(self.window, (20, 20, 20),
                             self._to_screen(s[0], s[1], p),
                             self._to_screen(e[0], e[1], p), 4)

        # --- Exits --- #
        for ep in exits_np:
            self._draw_triangle(self._to_screen(ep[0], ep[1], p), 10, (40, 180, 40))

        # --- Tasks --- #
        STATUS_COLORS = {0: (150, 40, 220), 1: (150, 40, 220),
                         2: (0, 200, 80),   3: (180, 60, 60)}
        for i in range(len(task_pos)):
            color = STATUS_COLORS.get(int(task_status[i]), (150, 40, 220))
            self._draw_star(self._to_screen(task_pos[i, 0], task_pos[i, 1], p), 9, color)

        # --- Robots --- #
        ROBOT_COLORS = [(50, 50, 220), (220, 140, 50)]  # blue, orange per robot
        for i in range(len(robot_pos)):
            x, y   = robot_pos[i]
            if np.isnan(x) or np.isnan(y):
                continue
            theta  = float(robot_theta[i])
            alive  = robot_alive[i] > 0.5
            carry  = robot_carrying[i] > 0.5
            center = self._to_screen(x, y, p)

            body_color = ROBOT_COLORS[i % len(ROBOT_COLORS)] if alive else (90, 90, 90)
            if alive and carry:
                body_color = tuple(min(255, c + 80) for c in body_color)  # lighter when carrying
            pygame.draw.circle(self.window, body_color, center, 8)
            if alive:
                self._draw_arrow(center, theta, (220, 30, 30))

        # --- HUD --- #
        if pygame.font.get_init() or (pygame.init() and True):
            try:
                font = pygame.font.SysFont("monospace", 14)
                step_surf = font.render(
                    f"Step {int(state.step_count):4d}  "
                    f"Safety {'OK' if state.global_safety_indicator > 0.5 else 'FAIL'}  "
                    f"Delivered {int(np.sum(task_status == 2))}/{len(task_pos)}",
                    True, (30, 30, 30))
                self.window.blit(step_surf, (6, 4))
            except Exception:
                pass

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False   # caller can use this to break

        pygame.display.flip()
        self.clock.tick(fps)
        return True

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock  = None
