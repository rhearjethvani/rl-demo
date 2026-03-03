# PAIRED RL Demo

A lightweight implementation of **Protagonist Antagonist Induced Regret Environment Design (PAIRED)** from:

> Dennis et al., *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design*, NeurIPS 2020. ([arXiv:2012.02096](https://arxiv.org/abs/2012.02096))

---

## Quick Start: Minimal Demo (Recommended)

The minimal demo isolates the PAIRED mechanism in a single file—no PyTorch, just NumPy:

```bash
pip install numpy   # if needed
python paired_demo.py
```

- **Q-learning** protagonist on a 7×7 GridWorld
- **Bandit adversary** over 3 difficulty buckets (2, 4, 6 obstacles)
- **Regret** = antagonist_return − protagonist_return
- Prints **chosen difficulty distribution**, **success rates** (PAIRED vs DR vs Minimax), and a **sanity check** (Q-table size/sum)

The curriculum shifts visibly over training:

| Phase | Highest-regret bucket |
|-------|------------------------|
| **Early** | b2 (2 obstacles) |
| **Mid** | b4 (4 obstacles) |
| **Late** | b6 (6 obstacles) |

This matches the paper’s algorithmic idea: the adversary concentrates on the protagonist’s “zone of proximal development.”

---

## Full PPO Implementation

A deep RL version using PyTorch + PPO is also available:

```bash
pip install -r requirements.txt
python train.py --method all --iterations 300
```

---

## What is PAIRED?

PAIRED is a method for **Unsupervised Environment Design (UED)**: instead of hand-crafting a training distribution, an adversary *learns* to generate environments that challenge the protagonist agent.

The key insight is using **minimax regret** as the adversary's objective:

```
REGRET(θ) = U_θ(πA) − U_θ(πP)
```

Where:
- `πP` = **Protagonist** — the agent we want to train (minimises regret)
- `πA` = **Antagonist** — a second agent allied with the adversary (maximises regret)
- `˜Λ` = **Adversary** — generates environment parameters θ to maximise regret

This prevents the adversary from creating unsolvable environments: if the environment is impossible, both agents fail equally and regret = 0, giving the adversary no reward.

### Algorithm 1 (PAIRED)

```
Randomly initialise Protagonist πP, Antagonist πA, Adversary ˜Λ

while not converged:
    θ ~ ˜Λ                                    # adversary generates environment
    τP ~ πP in M_θ,  U_θ(πP) = Σ r_t γ^t     # protagonist plays
    τA ~ πA in M_θ,  U_θ(πA) = Σ r_t γ^t     # antagonist plays
    REGRET = U_θ(πA) − U_θ(πP)               # compute regret

    Update πP  with reward = −REGRET          # protagonist minimises regret
    Update πA  with reward = +REGRET          # antagonist maximises regret
    Update ˜Λ  with reward = +REGRET          # adversary maximises regret
```

---

## Project Structure

```
rl-demo/
├── paired_demo.py    # Minimal demo: Q-learning + bandit adversary (run this first)
├── train.py          # Full PPO training (paired, DR, minimax)
├── env.py            # GridWorld environment (UPOMDP)
├── models.py         # AgentNet + AdversaryNet
├── ppo.py            # PPO trainer
├── paired.py         # PAIRED training loop (Algorithm 1)
├── baselines.py      # Domain Randomization + Minimax baselines
├── transfer_envs.py  # Hand-designed transfer environments
├── evaluate.py       # Evaluation utilities
└── requirements.txt  # For PPO version (torch, gymnasium, etc.); minimal demo needs only numpy
```

---

## Minimal Demo Details

| Component | Implementation |
|-----------|----------------|
| Protagonist | Q-learning (state = position only; generalizes across maze layouts) |
| Antagonist | Q-learning, pre-trained on easy envs |
| Adversary | ε-greedy bandit with EMA over mean regret |
| Environment | 7×7 grid, 3 difficulty buckets [2, 4, 6 obstacles], 60 steps/episode |
| Training | 1200 iterations, 500 antagonist pre-train steps |
| Eval | 100 held-out trials per bucket (seeds 9000–9099) |

The antagonist is pre-trained so it provides a meaningful regret signal from the start. The adversary uses EMA (α=0.25) so recent regret drives the curriculum shift. A 15% domain-randomization mix (`DR_MIX`) samples a random difficulty bucket instead of the adversary’s choice to improve generalization to held-out mazes.

After training, a sanity check prints Q-table size and value sum (non-zero confirms the agent was trained).

### Expected Results (minimal demo)

| Method | b2 (2 obs) | b4 (4 obs) | b6 (6 obs) |
|--------|------------|------------|------------|
| **PAIRED** | ~50% | ~23% | ~12% |
| Domain Rand. | ~60% | ~20% | ~10% |
| Minimax | 0% | 0% | 0% |

Minimax gets 0% because it only trains on the hardest bucket (b6) and never sees easier environments. PAIRED and Domain Randomization both achieve non-zero success; PAIRED additionally shows the curriculum shift in the difficulty distribution.

---

## Full PPO Environment

A **15×15 GridWorld** (13×13 navigable interior, border walls). 250 steps per episode.

The adversary's free parameters **θ** are:
1. Agent start position
2. Goal position
3. Up to 50 obstacle positions

The protagonist/antagonist observe a **5×5 partial view** (3 channels: walls, goal, empty) plus their current direction.

---

## Setup (PPO Version)

```bash
pip install -r requirements.txt
```

---

## PPO Training

```bash
# Train all three methods and compare
python train.py --method all --iterations 300

# Train PAIRED only
python train.py --method paired --iterations 500

# Baselines
python train.py --method domain_randomization --iterations 500
python train.py --method minimax --iterations 500
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | `all` | `paired`, `domain_randomization`, `minimax`, or `all` |
| `--iterations` | `300` | Number of training iterations |
| `--n_episodes` | `8` | Episodes per PPO update |
| `--seed` | `42` | Random seed |
| `--device` | `cpu` | `cpu` or `cuda` |

---

## PPO Outputs

Results are saved to `results/`:

- `results/<method>_metrics.npy` — training metrics
- `results/<method>_protagonist.pt` — saved weights
- `results/plots/training_curves.png` — solved path length + maze transfer over training
- `results/plots/transfer_comparison.png` — final zero-shot transfer bar chart
- `results/plots/solvable_fraction.png` — fraction of solvable envs generated over training

---

## Transfer Environments (PPO)

Zero-shot transfer is evaluated on five hand-designed environments (10 trials each):

| Environment | Description |
|-------------|-------------|
| Empty | Open grid, no obstacles |
| 50 Blocks | Dense random obstacles |
| Four Rooms | Classic four-room layout |
| Maze | Horizontal barriers, winding path |
| Labyrinth | Concentric rectangular walls |

---

## Key Design Choices

- **Non-negative regret**: `REGRET = max(0, REGRET)` — stabilises training
- **Protagonist/antagonist use environment reward** (not regret) — per paper Appendix E.2
- **Minimal demo**: EMA on adversary bandit for responsive curriculum shift; 15% DR mix for generalization to held-out mazes
- **State representation**: Position only (not walls) — allows generalization across layouts; including walls would be Markovian but break zero-shot transfer
- **PPO version**: All agents trained with PPO, matching the paper
