"""
PAIRED Double-Agent Demo
========================
Variant where the antagonist is trained alongside the protagonist (not fixed).
Explores convergence stability when both agents learn simultaneously.

Per Rajesh's request: "I would like to see another experiment with A also in
the training loop. Curious to know if convergence is stable or not. Would also
be nice to see graphs of regret vs iteration, to see how the different agents
are improving."

Run:
    python3 paired_demo_double_agent.py
"""

import numpy as np
import random
from collections import defaultdict
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── config ───────────────────────────────────────────────────────────────────
GRID           = 7
MAX_STEPS      = 60
ALPHA          = 0.5
GAMMA          = 0.95
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.9975

# No pre-train: both protagonist and antagonist start from scratch
ITERATIONS     = 1200
EVAL_EVERY     = 60
LOG_EVERY      = 1       # log every iteration for smooth plots

BUCKETS        = [2, 4, 6]
N_BUCKETS      = len(BUCKETS)
ADV_EXPLORE    = 5 * N_BUCKETS
ADV_EPSILON    = 0.15
ADV_EMA        = 0.25
DR_MIX         = 0.15

# Both agents use same epsilon decay (antagonist learns too)
ANT_EPSILON_DECAY = 0.9975


# ── GridWorld (same as paired_demo) ───────────────────────────────────────────

def make_grid(n_obstacles: int, seed=None):
    rng = np.random.RandomState(seed)
    forbidden = {(0, 0), (GRID - 1, GRID - 1)}
    candidates = [(r, c) for r in range(GRID) for c in range(GRID)
                  if (r, c) not in forbidden]
    n = min(n_obstacles, len(candidates))
    chosen = rng.choice(len(candidates), size=n, replace=False)
    return frozenset(candidates[i] for i in chosen)


def is_solvable(walls) -> bool:
    goal = (GRID - 1, GRID - 1)
    visited, queue = {(0, 0)}, [(0, 0)]
    while queue:
        r, c = queue.pop(0)
        if (r, c) == goal:
            return True
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            p = (nr, nc)
            if 0 <= nr < GRID and 0 <= nc < GRID and p not in walls and p not in visited:
                visited.add(p)
                queue.append(p)
    return False


def step(pos, action, walls):
    dr, dc = ((-1,0),(1,0),(0,-1),(0,1))[action]
    nr, nc = pos[0]+dr, pos[1]+dc
    if 0 <= nr < GRID and 0 <= nc < GRID and (nr, nc) not in walls:
        return (nr, nc)
    return pos


def run_episode(Q: dict, walls, epsilon: float) -> float:
    pos, goal = (0, 0), (GRID - 1, GRID - 1)
    total, gamma_t = 0.0, 1.0
    for _ in range(MAX_STEPS):
        if random.random() < epsilon:
            action = random.randrange(4)
        else:
            action = int(np.argmax([Q[(pos, a)] for a in range(4)]))
        nxt_pos = step(pos, action, walls)
        reward = 1.0 if nxt_pos == goal else -0.01
        total += gamma_t * reward
        gamma_t *= GAMMA
        best_next = max(Q[(nxt_pos, a)] for a in range(4))
        Q[(pos, action)] += ALPHA * (reward + GAMMA * best_next - Q[(pos, action)])
        pos = nxt_pos
        if pos == goal:
            break
    return total


def eval_episode(Q: dict, walls) -> float:
    pos, goal = (0, 0), (GRID - 1, GRID - 1)
    total, gamma_t = 0.0, 1.0
    for _ in range(MAX_STEPS):
        action = int(np.argmax([Q[(pos, a)] for a in range(4)]))
        nxt_pos = step(pos, action, walls)
        reward = 1.0 if nxt_pos == goal else -0.01
        total += gamma_t * reward
        gamma_t *= GAMMA
        pos = nxt_pos
        if pos == goal:
            break
    return total


# ── Adversary ─────────────────────────────────────────────────────────────────

class RegretAdversary:
    def __init__(self):
        self.counts = np.zeros(N_BUCKETS)
        self.values = np.zeros(N_BUCKETS)
        self.t = 0

    def choose(self) -> int:
        self.t += 1
        if self.t <= ADV_EXPLORE:
            return (self.t - 1) % N_BUCKETS
        if random.random() < ADV_EPSILON:
            return random.randrange(N_BUCKETS)
        return int(np.argmax(self.values))

    def update(self, arm: int, regret: float):
        self.counts[arm] += 1
        self.values[arm] += ADV_EMA * (regret - self.values[arm])

    def distribution(self) -> np.ndarray:
        return self.counts / (self.counts.sum() + 1e-9)

    def highest_regret_bucket(self) -> int:
        return int(np.argmax(self.values))


# ── Training (antagonist in the loop) ─────────────────────────────────────────

def train_paired_double_agent():
    protagonist = defaultdict(float)
    antagonist  = defaultdict(float)   # no pre-train; starts from scratch
    adversary   = RegretAdversary()

    epsilon_prot = EPSILON_START
    epsilon_ant  = EPSILON_START

    # Logging for plots
    history = {
        "iter": [],
        "regret": [],
        "prot_return": [],
        "ant_return": [],
        "mean_regret": [],
        "highest_bucket": [],
    }

    print("=" * 62)
    print("  PAIRED Double-Agent Demo  (A trained alongside P)")
    print("=" * 62)
    print(f"  Grid: {GRID}×{GRID}   Buckets: {BUCKETS}")
    print(f"  Antagonist NOT pre-trained — both agents learn from scratch")
    print(f"  Iterations: {ITERATIONS}")
    print()

    for it in range(1, ITERATIONS + 1):
        bucket = random.randrange(N_BUCKETS) if random.random() < DR_MIX else adversary.choose()
        n_obs = BUCKETS[bucket]
        walls = make_grid(n_obs, seed=it + 5000)

        if not is_solvable(walls):
            adversary.update(bucket, 0.0)
            continue

        # Both agents play and update (antagonist in the loop)
        prot_return = run_episode(protagonist, walls, epsilon_prot)
        ant_return  = run_episode(antagonist,  walls, epsilon_ant)

        regret = max(0.0, ant_return - prot_return)
        adversary.update(bucket, regret)

        epsilon_prot = max(EPSILON_END, epsilon_prot * EPSILON_DECAY)
        epsilon_ant  = max(EPSILON_END, epsilon_ant * ANT_EPSILON_DECAY)

        # Log every iteration for smooth plots
        if it % LOG_EVERY == 0:
            history["iter"].append(it)
            history["regret"].append(regret)
            history["prot_return"].append(prot_return)
            history["ant_return"].append(ant_return)
            history["mean_regret"].append(np.mean(adversary.values))
            history["highest_bucket"].append(BUCKETS[adversary.highest_regret_bucket()])

        if it % EVAL_EVERY == 0:
            best = adversary.highest_regret_bucket()
            mean_r = np.mean(history["regret"][-EVAL_EVERY:]) if len(history["regret"]) >= EVAL_EVERY else 0.0
            print(f"[iter {it:>4}]  regret={mean_r:.3f}  prot={np.mean(history['prot_return'][-EVAL_EVERY:]):.3f}  ant={np.mean(history['ant_return'][-EVAL_EVERY:]):.3f}  → b{BUCKETS[best]}")

    return protagonist, antagonist, adversary, history


def plot_results(history: dict, out_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not installed; skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    iters = np.array(history["iter"])
    regret = np.array(history["regret"])
    prot = np.array(history["prot_return"])
    ant = np.array(history["ant_return"])

    # Smooth with rolling mean for readability
    window = max(1, len(iters) // 100)
    def smooth(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Regret vs iteration
    ax = axes[0, 0]
    ax.plot(iters, regret, alpha=0.3, color="gray")
    if len(iters) >= window:
        s = smooth(regret)
        ax.plot(iters[window-1:], s, color="C0", linewidth=2, label="regret (smoothed)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret")
    ax.set_title("Regret vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Protagonist vs Antagonist return
    ax = axes[0, 1]
    ax.plot(iters, prot, alpha=0.3, color="C0")
    ax.plot(iters, ant, alpha=0.3, color="C1")
    if len(iters) >= window:
        ax.plot(iters[window-1:], smooth(prot), color="C0", linewidth=2, label="protagonist")
        ax.plot(iters[window-1:], smooth(ant), color="C1", linewidth=2, label="antagonist")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return")
    ax.set_title("Agent Returns vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Mean regret over buckets vs iteration
    ax = axes[1, 0]
    mean_regret = np.array(history["mean_regret"])
    ax.plot(iters, mean_regret, color="C2", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Regret (over buckets)")
    ax.set_title("Adversary Mean Regret vs Iteration")
    ax.grid(True, alpha=0.3)

    # 4. Highest-regret bucket over time (curriculum)
    ax = axes[1, 1]
    highest = np.array(history["highest_bucket"])
    if len(iters) >= window:
        # Mode over window
        modes = []
        for i in range(window - 1, len(highest)):
            vals = highest[i - window + 1 : i + 1]
            modes.append(np.bincount(vals.astype(int)).argmax())
        ax.plot(iters[window-1:], modes, color="C3", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Highest-regret bucket (# obstacles)")
    ax.set_title("Curriculum: Highest-Regret Bucket vs Iteration")
    ax.set_yticks(BUCKETS)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "paired_double_agent_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved plots to {path}")


def final_eval(Q: dict, n_trials: int = 50, seed_offset: int = 9000) -> dict:
    results = {}
    for n_obs in BUCKETS:
        successes = total_solvable = 0
        for trial in range(n_trials):
            walls = make_grid(n_obs, seed=seed_offset + trial)
            if not is_solvable(walls):
                continue
            total_solvable += 1
            if eval_episode(Q, walls) > 0:
                successes += 1
        results[n_obs] = successes / total_solvable if total_solvable else 0.0
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    protagonist, antagonist, adversary, history = train_paired_double_agent()

    # Plots
    plot_results(history, Path("results/plots"))

    # Eval
    print("\n  Final success rates (protagonist, 50 trials):")
    results = final_eval(protagonist, n_trials=50, seed_offset=9000)
    for n_obs in BUCKETS:
        print(f"    b{n_obs}: {results[n_obs]:.1%}")

    print("\n  Final adversary distribution:")
    dist = adversary.distribution()
    for n_obs, p in zip(BUCKETS, dist):
        bar = "█" * int(p * 40)
        print(f"    {n_obs} obs  {p:.2f}  {bar}")


if __name__ == "__main__":
    main()
