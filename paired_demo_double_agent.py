"""
PAIRED Double-Agent Demo
========================
Variant where the antagonist is trained alongside the protagonist (not fixed).
Explores convergence stability when both agents learn simultaneously.

Per Rajesh's request: "I would like to see another experiment with A also in
the training loop. Curious to know if convergence is stable or not. Would also
be nice to see graphs of regret vs iteration, to see how the different agents
are improving."

Regret definition (PAIRED paper):
    REGRET = antagonist_return - protagonist_return
    (Higher regret = antagonist did better than protagonist on that episode)

Run:
    python paired_demo_double_agent.py [--seed 42] [--num-seeds 1]
"""

import argparse
import numpy as np
import random
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass


# ── Config (single source of truth) ──────────────────────────────────────────

@dataclass
class Config:
    # Grid
    grid_size: int = 7
    max_steps: int = 60

    # Q-learning
    alpha: float = 0.5
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9975

    # Training
    iterations: int = 1200
    buckets: tuple = (2, 4, 6)
    dr_mix: float = 0.15
    ant_epsilon_decay: float = 0.9975

    # Adversary (bandit over buckets)
    adv_explore: int = 15  # 5 * n_buckets
    adv_epsilon: float = 0.15
    adv_ema: float = 0.25

    # Logging
    eval_every: int = 60
    log_every: int = 1

    # Evaluation (separate from training)
    eval_episodes_per_bucket: int = 100  # Use 100 for clean 1% steps
    eval_seed_offset: int = 9000

    # Reproducibility
    seed: int = 42
    num_seeds: int = 1

    @property
    def n_buckets(self) -> int:
        return len(self.buckets)

    def __post_init__(self):
        self.adv_explore = 5 * self.n_buckets


# Reward setup (documented for interpretability)
#   +1.0 for reaching goal, -0.01 per step
#   Return = discounted sum of rewards. Higher return is better.
REWARD_GOAL = 1.0
REWARD_STEP = -0.01


# ── Single source for regret ─────────────────────────────────────────────────
# REGRET = antagonist_return - protagonist_return  (computed exactly once per episode)


def compute_regret(ant_return: float, prot_return: float) -> float:
    """Regret = U(πA) - U(πP). Higher = antagonist did better."""
    return ant_return - prot_return


# ── GridWorld ────────────────────────────────────────────────────────────────

def make_grid(n_obstacles: int, grid_size: int, rng: np.random.RandomState) -> frozenset:
    forbidden = {(0, 0), (grid_size - 1, grid_size - 1)}
    candidates = [(r, c) for r in range(grid_size) for c in range(grid_size)
                  if (r, c) not in forbidden]
    n = min(n_obstacles, len(candidates))
    chosen = rng.choice(len(candidates), size=n, replace=False)
    return frozenset(candidates[i] for i in chosen)


def is_solvable(walls: frozenset, grid_size: int) -> bool:
    goal = (grid_size - 1, grid_size - 1)
    visited, queue = {(0, 0)}, [(0, 0)]
    while queue:
        r, c = queue.pop(0)
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            p = (nr, nc)
            if 0 <= nr < grid_size and 0 <= nc < grid_size and p not in walls and p not in visited:
                visited.add(p)
                queue.append(p)
    return False


def step(pos, action, walls, grid_size):
    dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
    nr, nc = pos[0] + dr, pos[1] + dc
    if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in walls:
        return (nr, nc)
    return pos


# ── Episodes ─────────────────────────────────────────────────────────────────

def run_episode(Q: dict, walls: frozenset, epsilon: float, cfg: Config, rng) -> float:
    """Run one episode, update Q (training). Returns discounted return."""
    pos, goal = (0, 0), (cfg.grid_size - 1, cfg.grid_size - 1)
    total, gamma_t = 0.0, 1.0
    for _ in range(cfg.max_steps):
        if rng.random() < epsilon:
            action = rng.randint(0, 4)
        else:
            action = int(np.argmax([Q[(pos, a)] for a in range(4)]))
        nxt_pos = step(pos, action, walls, cfg.grid_size)
        reward = REWARD_GOAL if nxt_pos == goal else REWARD_STEP
        total += gamma_t * reward
        gamma_t *= cfg.gamma
        best_next = max(Q[(nxt_pos, a)] for a in range(4))
        Q[(pos, action)] += cfg.alpha * (reward + cfg.gamma * best_next - Q[(pos, action)])
        pos = nxt_pos
        if pos == goal:
            break
    return total


def eval_episode(Q: dict, walls: frozenset, cfg: Config) -> tuple[float, bool]:
    """Evaluate one episode, no updates. Returns (return, reached_goal)."""
    pos, goal = (0, 0), (cfg.grid_size - 1, cfg.grid_size - 1)
    total, gamma_t = 0.0, 1.0
    for _ in range(cfg.max_steps):
        action = int(np.argmax([Q[(pos, a)] for a in range(4)]))
        nxt_pos = step(pos, action, walls, cfg.grid_size)
        reward = REWARD_GOAL if nxt_pos == goal else REWARD_STEP
        total += gamma_t * reward
        gamma_t *= cfg.gamma
        pos = nxt_pos
        if pos == goal:
            return total, True
    return total, False


# ── Adversary ────────────────────────────────────────────────────────────────

class RegretAdversary:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.counts = np.zeros(cfg.n_buckets)
        self.values = np.zeros(cfg.n_buckets)
        self.t = 0

    def choose(self, rng) -> int:
        self.t += 1
        if self.t <= self.cfg.adv_explore:
            return (self.t - 1) % self.cfg.n_buckets
        if rng.random() < self.cfg.adv_epsilon:
            return rng.randint(0, self.cfg.n_buckets)
        return int(np.argmax(self.values))

    def update(self, arm: int, regret: float):
        self.counts[arm] += 1
        self.values[arm] += self.cfg.adv_ema * (regret - self.values[arm])

    def distribution(self) -> np.ndarray:
        return self.counts / (self.counts.sum() + 1e-9)

    def highest_regret_bucket(self) -> int:
        return int(np.argmax(self.values))


# ── Training step ────────────────────────────────────────────────────────────

def train_one_iteration(protagonist, antagonist, adversary: RegretAdversary,
                        epsilon_prot: float, epsilon_ant: float, cfg: Config, rng) -> dict:
    """
    One training iteration: sample bucket, both agents play, compute regret,
    update adversary. Returns {prot_return, ant_return, regret, bucket}.
    """
    bucket = rng.randint(0, cfg.n_buckets) if rng.random() < cfg.dr_mix else adversary.choose(rng)
    n_obs = cfg.buckets[bucket]
    walls = make_grid(n_obs, cfg.grid_size, rng)

    if not is_solvable(walls, cfg.grid_size):
        adversary.update(bucket, 0.0)
        return {"prot_return": 0.0, "ant_return": 0.0, "regret": 0.0, "bucket": bucket}

    prot_return = run_episode(protagonist, walls, epsilon_prot, cfg, rng)
    ant_return = run_episode(antagonist, walls, epsilon_ant, cfg, rng)

    # Single place: REGRET = antagonist_return - protagonist_return
    regret = compute_regret(ant_return, prot_return)
    adversary.update(bucket, regret)

    return {"prot_return": prot_return, "ant_return": ant_return, "regret": regret, "bucket": bucket}


# ── Evaluation (separate from training) ───────────────────────────────────────

def build_fixed_eval_set(cfg: Config, rng: np.random.RandomState) -> dict[int, list]:
    """Pre-generate fixed set of solvable mazes per bucket for evaluation."""
    eval_set = {n_obs: [] for n_obs in cfg.buckets}
    for n_obs in cfg.buckets:
        tried = 0
        while len(eval_set[n_obs]) < cfg.eval_episodes_per_bucket and tried < 500:
            walls = make_grid(n_obs, cfg.grid_size, rng)
            if is_solvable(walls, cfg.grid_size):
                eval_set[n_obs].append(walls)
            tried += 1
    return eval_set


def evaluate_policy(Q: dict, eval_set: dict[int, list], cfg: Config) -> dict:
    """
    Evaluate policy on fixed eval set. Freeze Q (no updates).
    Returns {n_obs: {"success_rate": float, "successes": int, "total": int}}.
    """
    results = {}
    for n_obs in cfg.buckets:
        successes = 0
        for walls in eval_set[n_obs]:
            _, reached = eval_episode(Q, walls, cfg)
            if reached:
                successes += 1
        total = len(eval_set[n_obs])
        results[n_obs] = {
            "success_rate": successes / total if total else 0.0,
            "successes": successes,
            "total": total,
        }
    return results


# ── Training loop ─────────────────────────────────────────────────────────────

def train_paired_double_agent(cfg: Config) -> tuple[dict, dict, RegretAdversary, dict]:
    protagonist = defaultdict(float)
    antagonist = defaultdict(float)
    adversary = RegretAdversary(cfg)
    rng = np.random.RandomState(cfg.seed)
    random.seed(cfg.seed)

    epsilon_prot = cfg.epsilon_start
    epsilon_ant = cfg.epsilon_start

    history = {
        "iter": [],
        "regret": [],
        "prot_return": [],
        "ant_return": [],
        "mean_regret": [],
        "highest_bucket": [],
    }

    window = cfg.eval_every

    for it in range(1, cfg.iterations + 1):
        metrics = train_one_iteration(protagonist, antagonist, adversary,
                                      epsilon_prot, epsilon_ant, cfg, rng)

        epsilon_prot = max(cfg.epsilon_end, epsilon_prot * cfg.epsilon_decay)
        epsilon_ant = max(cfg.epsilon_end, epsilon_ant * cfg.ant_epsilon_decay)

        if it % cfg.log_every == 0:
            history["iter"].append(it)
            history["regret"].append(metrics["regret"])
            history["prot_return"].append(metrics["prot_return"])
            history["ant_return"].append(metrics["ant_return"])
            history["mean_regret"].append(np.mean(adversary.values))
            history["highest_bucket"].append(cfg.buckets[adversary.highest_regret_bucket()])

        if it % cfg.eval_every == 0:
            n = min(window, len(history["regret"]))
            mean_regret = np.mean(history["regret"][-n:])
            mean_prot = np.mean(history["prot_return"][-n:])
            mean_ant = np.mean(history["ant_return"][-n:])
            top_bucket = cfg.buckets[adversary.highest_regret_bucket()]
            print(f"[iter {it:>4}]  mean_regret(last {n})={mean_regret:+.3f}  "
                  f"mean_return_P={mean_prot:+.3f}  mean_return_A={mean_ant:+.3f}  "
                  f"top_bucket=b{top_bucket}")

    return protagonist, antagonist, adversary, history


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_metrics(history: dict, cfg: Config, out_dir: Path):
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

    window = max(1, len(iters) // 100)
    def smooth(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(iters, regret, alpha=0.3, color="gray")
    if len(iters) >= window:
        s = smooth(regret)
        ax.plot(iters[window - 1:], s, color="C0", linewidth=2, label="regret (smoothed)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret (= return_A − return_P)")
    ax.set_title("Regret vs Iteration (training)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(iters, prot, alpha=0.3, color="C0")
    ax.plot(iters, ant, alpha=0.3, color="C1")
    if len(iters) >= window:
        ax.plot(iters[window - 1:], smooth(prot), color="C0", linewidth=2, label="protagonist")
        ax.plot(iters[window - 1:], smooth(ant), color="C1", linewidth=2, label="antagonist")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return (higher is better)")
    ax.set_title("Agent Returns vs Iteration (training)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    mean_regret = np.array(history["mean_regret"])
    ax.plot(iters, mean_regret, color="C2", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Regret (over buckets)")
    ax.set_title("Adversary Mean Regret vs Iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    highest = np.array(history["highest_bucket"])
    if len(iters) >= window:
        modes = []
        for i in range(window - 1, len(highest)):
            vals = highest[i - window + 1 : i + 1]
            modes.append(int(np.bincount(vals.astype(int)).argmax()))
        ax.plot(iters[window - 1:], modes, color="C3", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Highest-regret bucket (# obstacles)")
    ax.set_title("Curriculum: Highest-Regret Bucket vs Iteration")
    ax.set_yticks(cfg.buckets)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "paired_demo_double_agent_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved plots to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PAIRED double-agent demo (A trained alongside P)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-seeds", type=int, default=1, help="Number of seeds (for stability checks)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Eval episodes per bucket (100 gives clean 1%% steps)")
    args = parser.parse_args()

    cfg = Config(seed=args.seed, num_seeds=args.num_seeds,
                 eval_episodes_per_bucket=args.eval_episodes)

    # Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 70)
    print("  PAIRED Double-Agent Demo  (A trained alongside P)")
    print("=" * 70)
    print(f"  Seed: {cfg.seed}")
    print(f"  Reward: +{REWARD_GOAL} goal, {REWARD_STEP} per step. Higher return is better.")
    print(f"  Regret = antagonist_return − protagonist_return")
    print(f"  Grid: {cfg.grid_size}×{cfg.grid_size}   Buckets: {cfg.buckets}")
    print(f"  Antagonist NOT pre-trained — both agents learn from scratch")
    print(f"  Iterations: {cfg.iterations}")
    print()

    protagonist, antagonist, adversary, history = train_paired_double_agent(cfg)

    # Plots (training metrics)
    plot_metrics(history, cfg, Path("results/plots"))

    # Evaluation (separate path: freeze policies, fixed eval set)
    print("\n  Evaluation (frozen policies, fixed eval set):")
    rng_eval = np.random.RandomState(cfg.eval_seed_offset)
    eval_set = build_fixed_eval_set(cfg, rng_eval)
    results = evaluate_policy(protagonist, eval_set, cfg)

    total_eps = sum(r["total"] for r in results.values())
    print(f"  Final success rates (protagonist, {total_eps} eval episodes total):")
    for n_obs in cfg.buckets:
        r = results[n_obs]
        pct = 100 * r["successes"] / r["total"] if r["total"] else 0
        print(f"    b{n_obs}: {r['successes']}/{r['total']} = {pct:.1f}%")

    print("\n  Final adversary distribution:")
    dist = adversary.distribution()
    for n_obs, p in zip(cfg.buckets, dist):
        bar = "█" * int(p * 40)
        print(f"    {n_obs} obs  {p:.2f}  {bar}")


if __name__ == "__main__":
    main()
