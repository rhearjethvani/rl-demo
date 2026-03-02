"""
PAIRED training loop.

Implements Algorithm 1 from "Emergent Complexity and Zero-shot Transfer
via Unsupervised Environment Design" (Dennis et al., NeurIPS 2020).

Three agents are trained simultaneously:
  - Adversary  (˜Λ): generates environment parameters θ to maximise regret
  - Protagonist (πP): navigates the environment, minimises regret
  - Antagonist  (πA): navigates the environment, maximises regret (allied with adversary)

Regret ≈ max_τA U(τA) − E_τP[U(τP)]

The adversary receives REGRET as reward.
The antagonist receives REGRET as reward.
The protagonist receives −REGRET as reward (i.e., normal env reward minus antagonist bonus).
"""

import numpy as np
import torch
from typing import List, Tuple, Dict

from env import GridWorld, GRID_SIZE, INNER_SIZE, MAX_BLOCKS, ADV_OBS_DIM, ADV_ACTION_DIM
from models import AgentNet, AdversaryNet
from ppo import PPOTrainer, RolloutBuffer


# -----------------------------------------------------------------------
# Adversary rollout: build the environment
# -----------------------------------------------------------------------

def build_env_with_adversary(
    adversary: AdversaryNet,
    device: str = "cpu",
    n_objects: int = MAX_BLOCKS + 2,  # agent + goal + blocks
) -> Tuple[GridWorld, RolloutBuffer, List[float]]:
    """
    Let the adversary place objects one by one.
    Returns (env, adv_buffer_placeholder, adv_obs_list).
    The adversary's reward will be filled in later (= regret).
    """
    env = GridWorld()
    env.reset_layout()

    adv_buffer = RolloutBuffer()
    z = np.random.randn(50).astype(np.float32)  # noise vector

    for t in range(n_objects):
        # Build adversary observation: grid image + timestep + z
        grid_img = _grid_to_adv_obs(env, t, z)
        obs_t = torch.FloatTensor(grid_img).unsqueeze(0).to(device)

        with torch.no_grad():
            action, log_prob, value = adversary.act(obs_t)

        a = action.item()
        lp = log_prob.item()
        v = value.item()

        env.place_object(a)
        adv_buffer.add(grid_img, a, lp, 0.0, v, False)  # reward filled later

    env.finalize_layout()
    return env, adv_buffer


def _grid_to_adv_obs(env: GridWorld, t: int, z: np.ndarray) -> np.ndarray:
    """Flatten grid state + timestep + noise into adversary observation."""
    grid = env.grid  # (15, 15)
    # 3 channels: wall, goal, empty
    channels = np.zeros((3, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    channels[0] = (grid == 1).astype(np.float32)  # wall
    channels[1] = (grid == 2).astype(np.float32)  # goal
    channels[2] = (grid == 0).astype(np.float32)  # empty
    flat = channels.flatten()
    t_norm = np.array([t / (MAX_BLOCKS + 2)], dtype=np.float32)
    return np.concatenate([flat, t_norm, z])


# -----------------------------------------------------------------------
# Agent rollout: play one episode
# -----------------------------------------------------------------------

def run_episode(
    agent: AgentNet,
    env: GridWorld,
    device: str = "cpu",
) -> Tuple[RolloutBuffer, float]:
    """
    Run one episode of agent in env.
    Returns (buffer, total_return).
    """
    buf = RolloutBuffer()
    obs = env.reset()
    total_return = 0.0

    while True:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, value = agent.act(obs_t)

        a = action.item()
        lp = log_prob.item()
        v = value.item()

        next_obs, reward, done = env.step(a)
        buf.add(obs, a, lp, reward, v, done)
        total_return += reward
        obs = next_obs

        if done:
            break

    return buf, total_return


# -----------------------------------------------------------------------
# PAIRED training
# -----------------------------------------------------------------------

class PAIREDTrainer:
    """
    Orchestrates the three-agent PAIRED training loop.
    """

    def __init__(
        self,
        protagonist: AgentNet,
        antagonist: AgentNet,
        adversary: AdversaryNet,
        ppo_protagonist: PPOTrainer,
        ppo_antagonist: PPOTrainer,
        ppo_adversary: PPOTrainer,
        n_episodes_per_update: int = 8,
        device: str = "cpu",
        nonneg_regret: bool = True,
    ):
        self.protagonist = protagonist
        self.antagonist = antagonist
        self.adversary = adversary
        self.ppo_protagonist = ppo_protagonist
        self.ppo_antagonist = ppo_antagonist
        self.ppo_adversary = ppo_adversary
        self.n_eps = n_episodes_per_update
        self.device = device
        self.nonneg_regret = nonneg_regret

    def train_step(self) -> Dict:
        """
        One PAIRED training step (Algorithm 1).
        Generates n_eps environments, collects trajectories, computes regret,
        assigns rewards, and runs PPO updates for all three agents.
        """
        prot_bufs, ant_bufs, adv_bufs = [], [], []
        regrets = []
        path_lengths = []
        solvable_count = 0

        for _ in range(self.n_eps):
            # 1. Adversary generates environment
            env, adv_buf, = build_env_with_adversary(self.adversary, self.device)[:2]

            solvable = env.is_solvable()
            if solvable:
                solvable_count += 1
            path_lengths.append(env.shortest_path_length())

            # 2. Protagonist and antagonist play in the same environment
            prot_buf, prot_return = run_episode(self.protagonist, env, self.device)
            ant_buf, ant_return = run_episode(self.antagonist, env, self.device)

            # 3. Regret = antagonist_return − protagonist_return
            regret = ant_return - prot_return
            if self.nonneg_regret:
                regret = max(0.0, regret)
            regrets.append(regret)

            # 4. Assign rewards
            # Protagonist: normal env reward (already in buffer)
            # Antagonist: normal env reward (already in buffer)
            # Adversary: regret signal distributed across its steps
            n_adv_steps = len(adv_buf)
            for i in range(n_adv_steps):
                adv_buf.rewards[i] = regret / n_adv_steps

            prot_bufs.append(prot_buf)
            ant_bufs.append(ant_buf)
            adv_bufs.append(adv_buf)

        # 5. PPO updates
        prot_stats = self.ppo_protagonist.update(prot_bufs)
        ant_stats = self.ppo_antagonist.update(ant_bufs)
        adv_stats = self.ppo_adversary.update(adv_bufs)

        return {
            "mean_regret": float(np.mean(regrets)),
            "mean_path_length": float(np.mean(path_lengths)),
            "solvable_frac": solvable_count / self.n_eps,
            "prot_policy_loss": prot_stats.get("policy_loss", 0.0),
            "ant_policy_loss": ant_stats.get("policy_loss", 0.0),
            "adv_policy_loss": adv_stats.get("policy_loss", 0.0),
        }
