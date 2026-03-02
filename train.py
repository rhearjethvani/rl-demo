"""
Main training script for PAIRED and baselines.

Usage:
    python train.py --method paired --iterations 500
    python train.py --method domain_randomization --iterations 500
    python train.py --method minimax --iterations 500
    python train.py --method all --iterations 500   # train all three and compare

Results are saved to results/<method>_metrics.npy and plots to results/plots/.
"""

import argparse
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from models import AgentNet, AdversaryNet
from ppo import PPOTrainer
from paired import PAIREDTrainer
from baselines import DomainRandomizationTrainer, MinimaxAdversaryTrainer
from evaluate import evaluate_all_transfer, solved_path_length
from baselines import build_random_env


def make_agent(device):
    return AgentNet().to(device)


def make_adversary(device):
    return AdversaryNet().to(device)


def make_ppo(model, lr=1e-4, entropy_coef=0.0, device="cpu"):
    return PPOTrainer(model, lr=lr, entropy_coef=entropy_coef, device=device)


def train_paired(args, device):
    print("\n=== Training PAIRED ===")
    protagonist = make_agent(device)
    antagonist = make_agent(device)
    adversary = make_adversary(device)

    trainer = PAIREDTrainer(
        protagonist=protagonist,
        antagonist=antagonist,
        adversary=adversary,
        ppo_protagonist=make_ppo(protagonist, device=device),
        ppo_antagonist=make_ppo(antagonist, device=device),
        ppo_adversary=make_ppo(adversary, lr=1e-4, entropy_coef=0.0, device=device),
        n_episodes_per_update=args.n_episodes,
        device=device,
        nonneg_regret=True,
    )

    metrics = run_training(trainer, protagonist, "paired", args, device)
    return protagonist, metrics


def train_domain_randomization(args, device):
    print("\n=== Training Domain Randomization ===")
    protagonist = make_agent(device)

    trainer = DomainRandomizationTrainer(
        protagonist=protagonist,
        ppo_protagonist=make_ppo(protagonist, device=device),
        n_episodes_per_update=args.n_episodes,
        device=device,
    )

    metrics = run_training(trainer, protagonist, "domain_randomization", args, device)
    return protagonist, metrics


def train_minimax(args, device):
    print("\n=== Training Minimax Adversarial ===")
    protagonist = make_agent(device)
    adversary = make_adversary(device)

    trainer = MinimaxAdversaryTrainer(
        protagonist=protagonist,
        adversary=adversary,
        ppo_protagonist=make_ppo(protagonist, device=device),
        ppo_adversary=make_ppo(adversary, device=device),
        n_episodes_per_update=args.n_episodes,
        device=device,
    )

    metrics = run_training(trainer, protagonist, "minimax", args, device)
    return protagonist, metrics


def run_training(trainer, protagonist, method_name, args, device):
    """Generic training loop with periodic evaluation."""
    os.makedirs("results", exist_ok=True)

    metrics = {
        "mean_path_length": [],
        "solvable_frac": [],
        "solved_path_length": [],
        "transfer_empty": [],
        "transfer_50_blocks": [],
        "transfer_four_rooms": [],
        "transfer_maze": [],
        "transfer_labyrinth": [],
    }
    if method_name == "paired":
        metrics["mean_regret"] = []

    eval_interval = max(1, args.iterations // 20)

    pbar = tqdm(range(1, args.iterations + 1), desc=method_name)
    for it in pbar:
        step_metrics = trainer.train_step()

        metrics["mean_path_length"].append(step_metrics.get("mean_path_length", 0.0))
        metrics["solvable_frac"].append(step_metrics.get("solvable_frac", 0.0))
        if method_name == "paired":
            metrics["mean_regret"].append(step_metrics.get("mean_regret", 0.0))

        if it % eval_interval == 0 or it == args.iterations:
            # Evaluate solved path length on random envs
            spl = solved_path_length(protagonist, build_random_env, n_trials=20, device=device)
            metrics["solved_path_length"].append(spl)

            # Transfer evaluation
            transfer = evaluate_all_transfer(protagonist, n_trials=10, device=device)
            metrics["transfer_empty"].append(transfer.get("empty", 0.0))
            metrics["transfer_50_blocks"].append(transfer.get("50_blocks", 0.0))
            metrics["transfer_four_rooms"].append(transfer.get("four_rooms", 0.0))
            metrics["transfer_maze"].append(transfer.get("maze", 0.0))
            metrics["transfer_labyrinth"].append(transfer.get("labyrinth", 0.0))

            pbar.set_postfix({
                "path_len": f"{step_metrics.get('mean_path_length', 0):.1f}",
                "solved_pl": f"{spl:.1f}",
                "maze_sr": f"{transfer.get('maze', 0):.2f}",
            })

    np.save(f"results/{method_name}_metrics.npy", metrics)
    torch.save(protagonist.state_dict(), f"results/{method_name}_protagonist.pt")
    print(f"  Saved results to results/{method_name}_metrics.npy")
    return metrics


def plot_comparison(all_metrics: dict, eval_interval: int, iterations: int):
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    os.makedirs("results/plots", exist_ok=True)
    eval_steps = list(range(eval_interval, iterations + 1, eval_interval))
    if iterations not in eval_steps:
        eval_steps.append(iterations)

    colors = {"paired": "#2196F3", "domain_randomization": "#FF9800", "minimax": "#F44336"}
    labels = {"paired": "PAIRED", "domain_randomization": "Domain Rand.", "minimax": "Minimax"}

    # --- Solved path length (emergent complexity) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for method, metrics in all_metrics.items():
        spl = metrics.get("solved_path_length", [])
        steps = eval_steps[:len(spl)]
        ax.plot(steps, spl, label=labels[method], color=colors[method], linewidth=2)
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Solved Path Length")
    ax.set_title("Emergent Complexity\n(Solved Path Length)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Transfer performance on maze ---
    ax = axes[1]
    for method, metrics in all_metrics.items():
        maze_sr = metrics.get("transfer_maze", [])
        steps = eval_steps[:len(maze_sr)]
        ax.plot(steps, maze_sr, label=labels[method], color=colors[method], linewidth=2)
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Success Rate")
    ax.set_title("Zero-Shot Transfer\n(Maze Environment)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("results/plots/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved training curves to results/plots/training_curves.png")

    # --- Transfer bar chart (final performance) ---
    transfer_keys = ["empty", "50_blocks", "four_rooms", "maze", "labyrinth"]
    transfer_labels = ["Empty", "50 Blocks", "Four Rooms", "Maze", "Labyrinth"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(transfer_keys))
    width = 0.25
    offsets = {"paired": -width, "domain_randomization": 0, "minimax": width}

    for method, metrics in all_metrics.items():
        vals = [
            (metrics.get(f"transfer_{k}", [0.0]) or [0.0])[-1]
            for k in transfer_keys
        ]
        ax.bar(x + offsets[method], vals, width, label=labels[method], color=colors[method], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(transfer_labels)
    ax.set_ylabel("Success Rate")
    ax.set_title("Zero-Shot Transfer Performance (Final)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("results/plots/transfer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved transfer comparison to results/plots/transfer_comparison.png")

    # --- Solvable fraction over training ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, metrics in all_metrics.items():
        sf = metrics.get("solvable_frac", [])
        steps = list(range(1, len(sf) + 1))
        # Smooth with rolling mean
        window = max(1, len(sf) // 20)
        smoothed = np.convolve(sf, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(sf) + 1), smoothed, label=labels[method],
                color=colors[method], linewidth=2)
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Fraction of Solvable Environments")
    ax.set_title("Solvable Environment Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/plots/solvable_fraction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved solvable fraction plot to results/plots/solvable_fraction.png")


def main():
    parser = argparse.ArgumentParser(description="PAIRED RL Demo")
    parser.add_argument(
        "--method",
        choices=["paired", "domain_randomization", "minimax", "all"],
        default="all",
        help="Training method to run",
    )
    parser.add_argument("--iterations", type=int, default=300,
                        help="Number of training iterations")
    parser.add_argument("--n_episodes", type=int, default=8,
                        help="Episodes per training update")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    print(f"PAIRED Demo — method={args.method}, iterations={args.iterations}, device={device}")
    start = time.time()

    all_metrics = {}

    if args.method in ("paired", "all"):
        _, metrics = train_paired(args, device)
        all_metrics["paired"] = metrics

    if args.method in ("domain_randomization", "all"):
        _, metrics = train_domain_randomization(args, device)
        all_metrics["domain_randomization"] = metrics

    if args.method in ("minimax", "all"):
        _, metrics = train_minimax(args, device)
        all_metrics["minimax"] = metrics

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")

    if len(all_metrics) > 1:
        eval_interval = max(1, args.iterations // 20)
        plot_comparison(all_metrics, eval_interval, args.iterations)

    # Print final transfer results
    print("\n=== Final Zero-Shot Transfer Results ===")
    transfer_keys = ["empty", "50_blocks", "four_rooms", "maze", "labyrinth"]
    header = f"{'Method':<25}" + "".join(f"{k:<14}" for k in transfer_keys)
    print(header)
    print("-" * len(header))
    for method, metrics in all_metrics.items():
        row = f"{method:<25}"
        for k in transfer_keys:
            val = (metrics.get(f"transfer_{k}", [0.0]) or [0.0])[-1]
            row += f"{val:<14.3f}"
        print(row)


if __name__ == "__main__":
    main()
