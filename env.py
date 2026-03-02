"""
GridWorld environment for PAIRED.

A 13x13 navigable grid (inside a 15x15 border of walls).
The adversary places: agent start, goal, and up to MAX_BLOCKS obstacles.
The protagonist/antagonist navigate to the goal under partial observability.
"""

import numpy as np
from typing import Optional, Tuple

GRID_SIZE = 15          # total grid including walls
INNER_SIZE = 13         # navigable interior
MAX_BLOCKS = 50
MAX_STEPS = 250

# Tile types
EMPTY = 0
WALL = 1
GOAL = 2
AGENT = 3

# Actions
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
N_ACTIONS = 4

# Observation: 5x5 partial view, 3 channels (tile type one-hot-ish), + direction
OBS_VIEW = 5
OBS_CHANNELS = 3        # empty/wall, goal, agent
OBS_SHAPE = (OBS_CHANNELS, OBS_VIEW, OBS_VIEW)
OBS_DIM = OBS_CHANNELS * OBS_VIEW * OBS_VIEW + 1  # +1 for direction


class GridWorld:
    """
    Partially-observable grid navigation environment.
    Free parameters (theta): positions of agent start, goal, and obstacles.
    """

    def __init__(self, grid_size: int = GRID_SIZE, max_steps: int = MAX_STEPS):
        self.grid_size = grid_size
        self.inner_size = grid_size - 2  # navigable area
        self.max_steps = max_steps
        self.n_actions = N_ACTIONS

        # Will be set by adversary
        self.grid: Optional[np.ndarray] = None
        self.agent_start: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None

        # Runtime state
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.direction: int = 0
        self.steps: int = 0
        self.done: bool = False

    # ------------------------------------------------------------------
    # Environment construction (called by adversary)
    # ------------------------------------------------------------------

    def reset_layout(self):
        """Create a fresh grid with only border walls."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        # Border walls
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL
        self.agent_start = None
        self.goal_pos = None

    def place_object(self, action: int) -> bool:
        """
        Place the next object at grid position encoded by action.
        action ∈ [0, inner_size^2)  →  (row, col) in navigable interior.
        Returns True if placement succeeded (non-overlapping).
        """
        row = action // self.inner_size + 1  # +1 for border
        col = action % self.inner_size + 1
        pos = (row, col)

        if self.agent_start is None:
            self.agent_start = pos
            return True
        if self.goal_pos is None:
            if pos == self.agent_start:
                # Place goal randomly if collision
                free = self._free_positions(exclude={self.agent_start})
                if not free:
                    return False
                self.goal_pos = free[np.random.randint(len(free))]
            else:
                self.goal_pos = pos
            self.grid[self.goal_pos] = GOAL
            return True
        # Place obstacle
        if self.grid[pos] == EMPTY:
            self.grid[pos] = WALL
            return True
        return False  # already occupied — no-op (adversary wastes action)

    def finalize_layout(self):
        """Ensure agent start and goal exist; fill defaults if adversary skipped them."""
        if self.agent_start is None:
            free = self._free_positions()
            self.agent_start = free[np.random.randint(len(free))]
        if self.goal_pos is None:
            free = self._free_positions(exclude={self.agent_start})
            if not free:
                free = self._free_positions()
            self.goal_pos = free[np.random.randint(len(free))]
            self.grid[self.goal_pos] = GOAL

    def is_solvable(self) -> bool:
        """BFS check: is there a path from agent_start to goal_pos?"""
        if self.agent_start is None or self.goal_pos is None:
            return False
        visited = set()
        queue = [self.agent_start]
        visited.add(self.agent_start)
        while queue:
            r, c = queue.pop(0)
            if (r, c) == self.goal_pos:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and self.grid[nr, nc] != WALL:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def shortest_path_length(self) -> int:
        """BFS shortest path length; returns 0 if unsolvable."""
        if self.agent_start is None or self.goal_pos is None:
            return 0
        visited = {self.agent_start: 0}
        queue = [(self.agent_start, 0)]
        while queue:
            (r, c), dist = queue.pop(0)
            if (r, c) == self.goal_pos:
                return dist
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and self.grid[nr, nc] != WALL:
                    visited[(nr, nc)] = dist + 1
                    queue.append(((nr, nc), dist + 1))
        return 0

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset agent to start position for a new episode."""
        self.agent_pos = self.agent_start
        self.direction = np.random.randint(4)
        self.steps = 0
        self.done = False
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute one step. Returns (obs, reward, done)."""
        if self.done:
            return self._get_obs(), 0.0, True

        dr, dc = {LEFT: (0, -1), RIGHT: (0, 1), UP: (-1, 0), DOWN: (1, 0)}[action]
        self.direction = action
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        if self.grid[nr, nc] != WALL:
            self.agent_pos = (nr, nc)

        self.steps += 1
        reward = 0.0
        if self.agent_pos == self.goal_pos:
            reward = 1.0 - 0.9 * (self.steps / self.max_steps)
            self.done = True
        elif self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Returns a flat observation: 5x5x3 partial view (flattened) + direction.
        Channels: [0] walls/empty, [1] goal, [2] agent (always center).
        """
        r, c = self.agent_pos
        half = OBS_VIEW // 2
        view = np.zeros((OBS_CHANNELS, OBS_VIEW, OBS_VIEW), dtype=np.float32)

        for i in range(OBS_VIEW):
            for j in range(OBS_VIEW):
                gr = r - half + i
                gc = c - half + j
                if 0 <= gr < self.grid_size and 0 <= gc < self.grid_size:
                    tile = self.grid[gr, gc]
                    if tile == WALL:
                        view[0, i, j] = 1.0
                    elif tile == GOAL:
                        view[1, i, j] = 1.0
                else:
                    view[0, i, j] = 1.0  # out-of-bounds = wall

        # Agent is always at center
        view[2, half, half] = 1.0

        obs = np.concatenate([view.flatten(), [self.direction / 3.0]])
        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _free_positions(self, exclude: set = None) -> list:
        positions = []
        for r in range(1, self.grid_size - 1):
            for c in range(1, self.grid_size - 1):
                if self.grid[r, c] == EMPTY:
                    pos = (r, c)
                    if exclude is None or pos not in exclude:
                        positions.append(pos)
        return positions

    def render_ascii(self) -> str:
        """Simple ASCII render for debugging."""
        symbols = {EMPTY: '.', WALL: '#', GOAL: 'G', AGENT: '@'}
        rows = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if (r, c) == self.agent_pos:
                    row.append('@')
                else:
                    row.append(symbols[self.grid[r, c]])
            rows.append(''.join(row))
        return '\n'.join(rows)


# ------------------------------------------------------------------
# Adversary action space
# ------------------------------------------------------------------
ADV_ACTION_DIM = INNER_SIZE * INNER_SIZE   # 169 positions
ADV_OBS_DIM = GRID_SIZE * GRID_SIZE * 3 + 1 + 50  # grid image + timestep + noise z
