"""
Evaluation utilities.

evaluate_agent: run N trials of an agent in a given environment and return
                success rate and mean return.
evaluate_all_transfer: evaluate across all transfer environments.
"""

import numpy as np
import torch
from typing import Dict, Tuple

from env import GridWorld
from models import AgentNet
from paired import run_episode
from transfer_envs import TRANSFER_ENVS


def evaluate_agent(
    agent: AgentNet,
    env_factory,
    n_trials: int = 10,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Run n_trials episodes of agent in env_factory().
    Returns (success_rate, mean_return).
    """
    successes = 0
    returns = []
    for _ in range(n_trials):
        env = env_factory()
        _, total_return = run_episode(agent, env, device)
        returns.append(total_return)
        if total_return > 0:
            successes += 1
    return successes / n_trials, float(np.mean(returns))


def evaluate_all_transfer(
    agent: AgentNet,
    n_trials: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate agent on all transfer environments. Returns success rates."""
    results = {}
    for name, factory in TRANSFER_ENVS.items():
        success_rate, mean_return = evaluate_agent(agent, factory, n_trials, device)
        results[name] = success_rate
    return results


def solved_path_length(
    agent: AgentNet,
    env_factory,
    n_trials: int = 20,
    device: str = "cpu",
) -> float:
    """
    Measure the mean shortest-path-length of mazes the agent successfully solves.
    Used to track emergent complexity (Figure 2d in paper).
    """
    solved_lengths = []
    for _ in range(n_trials):
        env = env_factory()
        if not env.is_solvable():
            continue
        spl = env.shortest_path_length()
        _, total_return = run_episode(agent, env, device)
        if total_return > 0:
            solved_lengths.append(spl)
    return float(np.mean(solved_lengths)) if solved_lengths else 0.0
