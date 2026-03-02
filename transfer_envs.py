"""
Hand-designed transfer environments for zero-shot evaluation.

Mirrors the test environments from the PAIRED paper:
  - Empty:     open grid, no obstacles
  - 50 Blocks: dense random obstacles
  - FourRooms: classic four-room layout
  - Maze:      simple maze with a long path
"""

import numpy as np
from env import GridWorld, GRID_SIZE, WALL, GOAL, EMPTY


def _make_env_from_grid(grid: np.ndarray, agent_start, goal_pos) -> GridWorld:
    env = GridWorld()
    env.grid = grid.copy()
    env.agent_start = agent_start
    env.goal_pos = goal_pos
    env.grid[goal_pos] = GOAL
    return env


# -----------------------------------------------------------------------
# Empty
# -----------------------------------------------------------------------

def make_empty_env() -> GridWorld:
    """Open grid, agent bottom-left, goal top-right."""
    env = GridWorld()
    env.reset_layout()
    env.agent_start = (GRID_SIZE - 2, 1)
    env.goal_pos = (1, GRID_SIZE - 2)
    env.grid[env.goal_pos] = GOAL
    return env


# -----------------------------------------------------------------------
# 50 Blocks (random dense)
# -----------------------------------------------------------------------

def make_50blocks_env(seed: int = 42) -> GridWorld:
    rng = np.random.RandomState(seed)
    env = GridWorld()
    env.reset_layout()
    env.agent_start = (GRID_SIZE - 2, 1)
    env.goal_pos = (1, GRID_SIZE - 2)
    env.grid[env.goal_pos] = GOAL

    placed = 0
    while placed < 50:
        r = rng.randint(1, GRID_SIZE - 1)
        c = rng.randint(1, GRID_SIZE - 1)
        if env.grid[r, c] == EMPTY and (r, c) != env.agent_start:
            env.grid[r, c] = WALL
            placed += 1
    return env


# -----------------------------------------------------------------------
# Four Rooms
# -----------------------------------------------------------------------

def make_four_rooms_env() -> GridWorld:
    """Classic four-rooms layout on a 15x15 grid."""
    env = GridWorld()
    env.reset_layout()
    g = env.grid
    mid = GRID_SIZE // 2  # 7

    # Horizontal wall
    for c in range(1, GRID_SIZE - 1):
        g[mid, c] = WALL
    # Vertical wall
    for r in range(1, GRID_SIZE - 1):
        g[r, mid] = WALL

    # Doorways (openings in walls)
    g[mid, 3] = EMPTY       # bottom-left → top-left
    g[mid, 11] = EMPTY      # bottom-right → top-right
    g[3, mid] = EMPTY       # left → right (top)
    g[11, mid] = EMPTY      # left → right (bottom)

    env.agent_start = (GRID_SIZE - 2, 1)
    env.goal_pos = (1, GRID_SIZE - 2)
    g[env.goal_pos] = GOAL
    return env


# -----------------------------------------------------------------------
# Maze
# -----------------------------------------------------------------------

def make_maze_env() -> GridWorld:
    """
    A hand-crafted maze requiring a winding path.
    Agent starts bottom-left, goal is top-right.
    """
    env = GridWorld()
    env.reset_layout()
    g = env.grid

    # Horizontal barriers with gaps
    for c in range(2, 13):
        g[3, c] = WALL
    g[3, 12] = EMPTY   # gap right

    for c in range(1, 11):
        g[6, c] = WALL
    g[6, 1] = EMPTY    # gap left

    for c in range(3, 14):
        g[9, c] = WALL
    g[9, 13] = EMPTY   # gap right

    for c in range(1, 12):
        g[12, c] = WALL
    g[12, 1] = EMPTY   # gap left

    env.agent_start = (13, 1)
    env.goal_pos = (1, 13)
    g[env.goal_pos] = GOAL
    return env


# -----------------------------------------------------------------------
# Labyrinth (spiral-like)
# -----------------------------------------------------------------------

def make_labyrinth_env() -> GridWorld:
    """Concentric rectangular walls with a single entry point each."""
    env = GridWorld()
    env.reset_layout()
    g = env.grid

    def rect_wall(r1, c1, r2, c2, gap_side, gap_pos):
        for c in range(c1, c2 + 1):
            g[r1, c] = WALL
            g[r2, c] = WALL
        for r in range(r1, r2 + 1):
            g[r, c1] = WALL
            g[r, c2] = WALL
        # Open gap
        if gap_side == 'top':
            g[r1, gap_pos] = EMPTY
        elif gap_side == 'bottom':
            g[r2, gap_pos] = EMPTY
        elif gap_side == 'left':
            g[gap_pos, c1] = EMPTY
        elif gap_side == 'right':
            g[gap_pos, c2] = EMPTY

    rect_wall(2, 2, 12, 12, 'bottom', 7)
    rect_wall(4, 4, 10, 10, 'top', 7)
    rect_wall(6, 6, 8, 8, 'bottom', 7)

    env.agent_start = (13, 7)
    env.goal_pos = (7, 7)
    g[env.goal_pos] = GOAL
    return env


TRANSFER_ENVS = {
    "empty": make_empty_env,
    "50_blocks": make_50blocks_env,
    "four_rooms": make_four_rooms_env,
    "maze": make_maze_env,
    "labyrinth": make_labyrinth_env,
}
