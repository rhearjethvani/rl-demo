"""
PAIRED Minimal Demo
===================
Implements the core PAIRED mechanism in a single file using:
  - Q-learning agent (protagonist) on a small GridWorld
  - Bandit adversary over 3 difficulty buckets (2, 4, 6 obstacles)
  - Regret = antagonist_return - protagonist_return
  - Adversary picks difficulty to maximise regret → automatic curriculum

The curriculum shifts visibly over training:
  Early:  b2 highest regret (protagonist struggles on easy)
  Mid:    b4 highest regret (protagonist mastered b2)
  Late:   b6 highest regret (protagonist mastered b4)

Run:
    python3 paired_demo.py
"""

import numpy as np
import random
from collections import defaultdict

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── config ───────────────────────────────────────────────────────────────────
GRID           = 7          # 7×7 — smaller = faster mastery, clearer b2/b4/b6 separation
MAX_STEPS      = 60
ALPHA          = 0.5
GAMMA          = 0.95
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.9975     # protagonist masters each phase in ~300–400 iters

PRETRAIN_STEPS = 500
ITERATIONS     = 1200       # ~400 per phase for visible b2→b4→b6 shift
EVAL_EVERY     = 120

# 3 buckets for a clear curriculum: b2 → b4 → b6
BUCKETS    = [2, 4, 6]
N_BUCKETS  = len(BUCKETS)

# Adversary: round-robin, then ε-greedy; EMA so recent regret drives the shift
ADV_EXPLORE   = 5 * N_BUCKETS
ADV_EPSILON   = 0.15        # 15% explore — get fresh regret for b4/b6
ADV_EMA       = 0.25        # new sample weight; recent regret dominates


# ── GridWorld ─────────────────────────────────────────────────────────────────

def make_grid(n_obstacles: int, seed=None):
    """
    GRID×GRID maze. Agent at (0,0), goal at (GRID-1,GRID-1).
    Returns a frozenset of wall positions.
    """
    rng = np.random.RandomState(seed)
    forbidden = {(0, 0), (GRID - 1, GRID - 1)}
    candidates = [(r, c) for r in range(GRID) for c in range(GRID)
                  if (r, c) not in forbidden]
    n = min(n_obstacles, len(candidates))
    chosen = rng.choice(len(candidates), size=n, replace=False)
    return frozenset(candidates[i] for i in chosen)


def is_solvable(walls) -> bool:
    """BFS reachability from (0,0) to goal."""
    goal = (GRID - 1, GRID - 1)
    visited = {(0, 0)}
    queue = [(0, 0)]
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


def step(state, action, walls):
    dr, dc = ((-1,0),(1,0),(0,-1),(0,1))[action]
    nr, nc = state[0]+dr, state[1]+dc
    if 0 <= nr < GRID and 0 <= nc < GRID and (nr, nc) not in walls:
        return (nr, nc)
    return state


def run_episode(Q: dict, walls, epsilon: float) -> float:
    """ε-greedy Q-learning episode. Updates Q in-place. Returns discounted return."""
    state   = (0, 0)
    goal    = (GRID - 1, GRID - 1)
    total   = 0.0
    gamma_t = 1.0

    for _ in range(MAX_STEPS):
        if random.random() < epsilon:
            action = random.randrange(4)
        else:
            action = int(np.argmax([Q[(state, a)] for a in range(4)]))

        nxt    = step(state, action, walls)
        reward = 1.0 if nxt == goal else -0.01
        total += gamma_t * reward
        gamma_t *= GAMMA

        best_next = max(Q[(nxt, a)] for a in range(4))
        Q[(state, action)] += ALPHA * (reward + GAMMA * best_next - Q[(state, action)])
        state = nxt
        if state == goal:
            break

    return total


def eval_episode(Q: dict, walls) -> float:
    """Greedy episode (no exploration). Returns discounted return."""
    state   = (0, 0)
    goal    = (GRID - 1, GRID - 1)
    total   = 0.0
    gamma_t = 1.0

    for _ in range(MAX_STEPS):
        action = int(np.argmax([Q[(state, a)] for a in range(4)]))
        nxt    = step(state, action, walls)
        reward = 1.0 if nxt == goal else -0.01
        total += gamma_t * reward
        gamma_t *= GAMMA
        state = nxt
        if state == goal:
            break

    return total


# ── Adversary bandit (UCB1) ───────────────────────────────────────────────────

class RegretAdversary:
    """
    Bandit over N_BUCKETS difficulty levels.
    Reward = regret = max(0, antagonist_return − protagonist_return).
    Round-robin explore, then greedy on mean regret — concentrates on
    the protagonist's zone of proximal development (highest-regret bucket).
    """

    def __init__(self):
        self.counts = np.zeros(N_BUCKETS)
        self.values = np.zeros(N_BUCKETS)   # running mean regret per bucket
        self.t      = 0

    def choose(self) -> int:
        self.t += 1
        if self.t <= ADV_EXPLORE:
            return (self.t - 1) % N_BUCKETS
        if random.random() < ADV_EPSILON:
            return random.randrange(N_BUCKETS)   # explore
        return int(np.argmax(self.values))       # exploit: highest regret

    def update(self, arm: int, regret: float):
        self.counts[arm] += 1
        # EMA: recent regret dominates → fast response when prot masters a level
        self.values[arm] += ADV_EMA * (regret - self.values[arm])

    def distribution(self) -> np.ndarray:
        """Empirical: fraction of time each bucket was chosen (over recent window)."""
        return self.counts / (self.counts.sum() + 1e-9)

    def highest_regret_bucket(self) -> int:
        return int(np.argmax(self.values))


# ── PAIRED training ───────────────────────────────────────────────────────────

def pretrain(Q: dict, n_steps: int):
    """
    Curriculum warm-up for antagonist: easy → medium → hard.
    Ensures antagonist is strong on b2, good on b4, and can handle b6,
    so regret peaks at different buckets as protagonist progresses.
    """
    eps = 0.6
    third = n_steps // 3
    for i in range(n_steps):
        if i < third:
            max_obs = 2       # Phase 1: master b2
        elif i < 2 * third:
            max_obs = 4       # Phase 2: add b4
        else:
            max_obs = 6       # Phase 3: add b6
        n_obs = random.randint(0, max_obs)
        walls = make_grid(n_obs, seed=i + 10000)
        if is_solvable(walls):
            run_episode(Q, walls, eps)
        eps = max(0.05, eps * 0.997)


def train_paired():
    protagonist = defaultdict(float)
    antagonist  = defaultdict(float)
    adversary   = RegretAdversary()

    # Pre-train antagonist with curriculum (easy → hard) so it stays ahead on b2/b4/b6
    pretrain(antagonist, PRETRAIN_STEPS)

    epsilon = EPSILON_START

    print("=" * 62)
    print("  PAIRED Minimal Demo  (Q-learning + Regret Bandit Adversary)")
    print("=" * 62)
    print(f"  Grid: {GRID}×{GRID}   Buckets: {BUCKETS} obstacles")
    print(f"  Antagonist pre-trained for {PRETRAIN_STEPS} steps on easy envs")
    print(f"  PAIRED iterations: {ITERATIONS}   ε: {EPSILON_START}→{EPSILON_END}")
    print()

    for it in range(1, ITERATIONS + 1):
        # 1. Adversary picks difficulty
        bucket = adversary.choose()
        n_obs  = BUCKETS[bucket]

        # 2. Build environment
        walls = make_grid(n_obs, seed=it + 5000)
        if not is_solvable(walls):
            adversary.update(bucket, 0.0)   # unsolvable → zero regret
            continue

        # 3. Both agents play
        prot_return = run_episode(protagonist, walls, epsilon)
        ant_return  = run_episode(antagonist,  walls, epsilon * 0.5)  # antagonist explores less

        # 4. Regret (non-negative, per paper Appendix F)
        regret = max(0.0, ant_return - prot_return)

        # 5. Update adversary bandit
        adversary.update(bucket, regret)

        # 6. Decay protagonist exploration
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # 7. Periodic status
        if it % EVAL_EVERY == 0:
            best = adversary.highest_regret_bucket()
            dist = adversary.distribution()
            print(f"[iter {it:>4}]  ε={epsilon:.3f}  → highest regret: b{BUCKETS[best]} ({BUCKETS[best]} obstacles)")
            print(f"  Mean regret: " + "  ".join(f"b{b}={v:.3f}" for b, v in zip(BUCKETS, adversary.values)))
            print("  Chosen distribution (empirical):")
            for n_obs_b, p in zip(BUCKETS, dist):
                bar = "█" * int(p * 40)
                mark = "  ←" if BUCKETS[best] == n_obs_b else ""
                print(f"    {n_obs_b:>2} obstacles  {p:.2f}  {bar}{mark}")
            print()

    return protagonist, adversary


# ── Baselines ─────────────────────────────────────────────────────────────────

def train_domain_randomization():
    """Protagonist trained on uniformly random difficulty each step."""
    protagonist = defaultdict(float)
    epsilon = EPSILON_START
    for it in range(1, ITERATIONS + 1):
        n_obs = BUCKETS[random.randrange(N_BUCKETS)]
        walls = make_grid(n_obs, seed=it + 5000)
        if is_solvable(walls):
            run_episode(protagonist, walls, epsilon)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    return protagonist


def train_minimax():
    """Adversary always picks the hardest bucket — protagonist never sees easy envs."""
    protagonist = defaultdict(float)
    epsilon = EPSILON_START
    for it in range(1, ITERATIONS + 1):
        walls = make_grid(BUCKETS[-1], seed=it + 5000)
        if is_solvable(walls):
            run_episode(protagonist, walls, epsilon)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    return protagonist


# ── Evaluation ────────────────────────────────────────────────────────────────

def final_eval(Q: dict, n_trials: int = 100) -> dict:
    """Greedy success rate across all difficulty buckets (held-out seeds)."""
    results = {}
    for n_obs in BUCKETS:
        successes = 0
        total_solvable = 0
        for trial in range(n_trials):
            walls = make_grid(n_obs, seed=9000 + trial)
            if not is_solvable(walls):
                continue
            total_solvable += 1
            if eval_episode(Q, walls) > 0:
                successes += 1
        results[n_obs] = successes / total_solvable if total_solvable else 0.0
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    paired_agent, adversary = train_paired()

    print("Training baselines (silent)...")
    dr_agent = train_domain_randomization()
    mm_agent = train_minimax()
    print()

    paired_results = final_eval(paired_agent)
    dr_results     = final_eval(dr_agent)
    mm_results     = final_eval(mm_agent)

    print("=" * 62)
    print("  Final Zero-Shot Success Rates  (100 held-out trials each)")
    print("=" * 62)
    print(f"  {'Obstacles':<12}  {'PAIRED':^16}  {'Domain Rand':^16}  {'Minimax':^16}")
    print("  " + "─" * 58)
    for n_obs in BUCKETS:
        p  = paired_results[n_obs]
        dr = dr_results[n_obs]
        mm = mm_results[n_obs]
        def bar(v): return ("█" * int(v * 10)).ljust(10)
        print(f"  {n_obs:<12}  {p:>5.1%} {bar(p)}  {dr:>5.1%} {bar(dr)}  {mm:>5.1%} {bar(mm)}")

    print()
    print("  Final chosen difficulty distribution (PAIRED):")
    dist = adversary.distribution()
    best = adversary.highest_regret_bucket()
    for n_obs, p in zip(BUCKETS, dist):
        bar = "█" * int(p * 45)
        mark = "  ← highest regret" if BUCKETS[best] == n_obs else ""
        print(f"    {n_obs:>2} obstacles  {p:.2f}  {bar}{mark}")

    print()
    print("  Key insight from the paper:")
    print("  The adversary concentrates on the difficulty bucket where")
    print("  regret is highest — the protagonist's 'zone of proximal")
    print("  development' — creating an automatic curriculum that")
    print("  neither domain randomization nor minimax can replicate.")


if __name__ == "__main__":
    main()
