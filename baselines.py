"""
Baseline UED methods for comparison with PAIRED.

1. DomainRandomization: places agent, goal, and obstacles uniformly at random.
2. MinimaxAdversary: adversary minimises protagonist reward (no antagonist).
"""

import numpy as np
import torch
from typing import List, Dict

from env import GridWorld, MAX_BLOCKS, INNER_SIZE, GRID_SIZE
from models import AgentNet, AdversaryNet
from ppo import PPOTrainer, RolloutBuffer
from paired import run_episode, build_env_with_adversary, _grid_to_adv_obs


# -----------------------------------------------------------------------
# Domain Randomization
# -----------------------------------------------------------------------

def build_random_env(max_blocks: int = MAX_BLOCKS) -> GridWorld:
    """Build a fully random environment (domain randomization)."""
    env = GridWorld()
    env.reset_layout()
    inner = INNER_SIZE
    n_positions = inner * inner

    # Place agent
    env.place_object(np.random.randint(n_positions))
    # Place goal
    env.place_object(np.random.randint(n_positions))
    # Place random number of blocks
    n_blocks = np.random.randint(0, max_blocks + 1)
    for _ in range(n_blocks):
        env.place_object(np.random.randint(n_positions))

    env.finalize_layout()
    return env


class DomainRandomizationTrainer:
    """Train protagonist with domain randomization."""

    def __init__(
        self,
        protagonist: AgentNet,
        ppo_protagonist: PPOTrainer,
        n_episodes_per_update: int = 8,
        device: str = "cpu",
    ):
        self.protagonist = protagonist
        self.ppo_protagonist = ppo_protagonist
        self.n_eps = n_episodes_per_update
        self.device = device

    def train_step(self) -> Dict:
        prot_bufs = []
        path_lengths = []
        solvable_count = 0

        for _ in range(self.n_eps):
            env = build_random_env()
            if env.is_solvable():
                solvable_count += 1
            path_lengths.append(env.shortest_path_length())
            prot_buf, _ = run_episode(self.protagonist, env, self.device)
            prot_bufs.append(prot_buf)

        prot_stats = self.ppo_protagonist.update(prot_bufs)
        return {
            "mean_path_length": float(np.mean(path_lengths)),
            "solvable_frac": solvable_count / self.n_eps,
            "prot_policy_loss": prot_stats.get("policy_loss", 0.0),
        }


# -----------------------------------------------------------------------
# Minimax Adversarial
# -----------------------------------------------------------------------

class MinimaxAdversaryTrainer:
    """
    Train protagonist + adversary where adversary minimises protagonist reward.
    No antagonist — adversary reward = −protagonist_return.
    """

    def __init__(
        self,
        protagonist: AgentNet,
        adversary: AdversaryNet,
        ppo_protagonist: PPOTrainer,
        ppo_adversary: PPOTrainer,
        n_episodes_per_update: int = 8,
        device: str = "cpu",
    ):
        self.protagonist = protagonist
        self.adversary = adversary
        self.ppo_protagonist = ppo_protagonist
        self.ppo_adversary = ppo_adversary
        self.n_eps = n_episodes_per_update
        self.device = device

    def train_step(self) -> Dict:
        prot_bufs, adv_bufs = [], []
        path_lengths = []
        solvable_count = 0

        for _ in range(self.n_eps):
            env, adv_buf = build_env_with_adversary(self.adversary, self.device)[:2]

            if env.is_solvable():
                solvable_count += 1
            path_lengths.append(env.shortest_path_length())

            prot_buf, prot_return = run_episode(self.protagonist, env, self.device)

            # Adversary reward = −protagonist_return (minimax)
            n_adv_steps = len(adv_buf)
            for i in range(n_adv_steps):
                adv_buf.rewards[i] = -prot_return / n_adv_steps

            prot_bufs.append(prot_buf)
            adv_bufs.append(adv_buf)

        prot_stats = self.ppo_protagonist.update(prot_bufs)
        adv_stats = self.ppo_adversary.update(adv_bufs)

        return {
            "mean_path_length": float(np.mean(path_lengths)),
            "solvable_frac": solvable_count / self.n_eps,
            "prot_policy_loss": prot_stats.get("policy_loss", 0.0),
            "adv_policy_loss": adv_stats.get("policy_loss", 0.0),
        }
