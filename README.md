# PAIRED RL Demo

A lightweight implementation of **Protagonist Antagonist Induced Regret Environment Design (PAIRED)** from:

> Dennis et al., *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design*, NeurIPS 2020. ([arXiv:2012.02096](https://arxiv.org/abs/2012.02096))

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
├── env.py            # GridWorld environment (UPOMDP)
├── models.py         # AgentNet (protagonist/antagonist) + AdversaryNet
├── ppo.py            # PPO trainer (shared across all three agents)
├── paired.py         # PAIRED training loop (Algorithm 1)
├── baselines.py      # Domain Randomization + Minimax Adversarial baselines
├── transfer_envs.py  # Hand-designed transfer environments (Empty, FourRooms, Maze, ...)
├── evaluate.py       # Evaluation utilities (success rate, solved path length)
├── train.py          # Main training script
└── requirements.txt
```

---

## Environment

A **15×15 partially-observable GridWorld** (13×13 navigable interior).

The adversary's free parameters **θ** are:
1. Agent start position
2. Goal position
3. Up to 50 obstacle positions

The protagonist/antagonist observe a **5×5 partial view** (3 channels: walls, goal, empty) plus their current direction. They must navigate to the green goal square.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

```bash
# Train all three methods and compare (recommended)
python train.py --method all --iterations 300

# Train PAIRED only
python train.py --method paired --iterations 500

# Train domain randomization baseline
python train.py --method domain_randomization --iterations 500

# Train minimax adversarial baseline
python train.py --method minimax --iterations 500
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--method` | `all` | `paired`, `domain_randomization`, `minimax`, or `all` |
| `--iterations` | `300` | Number of training iterations |
| `--n_episodes` | `8` | Episodes per PPO update |
| `--seed` | `42` | Random seed |
| `--device` | `cpu` | `cpu` or `cuda` |

---

## Outputs

Results are saved to `results/`:

- `results/<method>_metrics.npy` — training metrics dict
- `results/<method>_protagonist.pt` — saved protagonist weights
- `results/plots/training_curves.png` — solved path length + maze transfer over training
- `results/plots/transfer_comparison.png` — final zero-shot transfer bar chart
- `results/plots/solvable_fraction.png` — fraction of solvable environments generated

---

## Transfer Environments

Zero-shot transfer is evaluated on five hand-designed environments never seen during training:

| Environment | Description |
|---|---|
| **Empty** | Open grid, no obstacles |
| **50 Blocks** | Dense random obstacles |
| **Four Rooms** | Classic four-room layout with doorways |
| **Maze** | Horizontal barriers requiring winding path |
| **Labyrinth** | Concentric rectangular walls |

---

## Expected Results

After ~300–500 iterations:

| Method | Empty | 50 Blocks | Four Rooms | Maze | Labyrinth |
|---|---|---|---|---|---|
| **PAIRED** | High | Medium | Medium | Low–Medium | Low–Medium |
| Domain Rand. | High | Low | Low | Very Low | Very Low |
| Minimax | High | Very Low | Very Low | ~0 | ~0 |

PAIRED is the only method that generates a **curriculum of increasing complexity**, training the protagonist on progressively harder environments and achieving meaningful zero-shot transfer to maze-like structures.

---

## Key Design Choices

- **Non-negative regret**: `REGRET = max(0, REGRET)` — stabilises training by removing noise when protagonist outperforms antagonist
- **Regret distributed across adversary steps**: each adversary placement action receives `regret / n_steps` reward
- **Protagonist/antagonist use environment reward** (not regret) — per Appendix E.2, this is more stable
- **All agents trained with PPO** — same optimizer, same hyperparameters, matching the paper
