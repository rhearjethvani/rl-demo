"""
PPO (Proximal Policy Optimization) trainer.

A single PPOTrainer can be used for the protagonist, antagonist, or adversary.
Rollout collection and update logic are separated so the PAIRED loop can
orchestrate them across all three agents.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class RolloutBuffer:
    """Stores one episode's worth of transitions for PPO."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


class PPOTrainer:
    """
    PPO update logic.  Operates on a batch of rollout buffers.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.0,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        device: str = "cpu",
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.device = device

    def compute_returns_and_advantages(
        self, rewards: List[float], values: List[float], dones: List[bool], last_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            next_done = dones[t]
            delta = rewards[t] + self.gamma * next_val * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        return returns, advantages

    def update(self, buffers: List[RolloutBuffer]) -> dict:
        """
        Run PPO update on a list of rollout buffers (one per episode).
        Returns a dict of loss statistics.
        """
        all_obs, all_actions, all_log_probs, all_returns, all_advantages = [], [], [], [], []

        for buf in buffers:
            if len(buf) == 0:
                continue
            returns, advantages = self.compute_returns_and_advantages(
                buf.rewards, buf.values, buf.dones
            )
            all_obs.extend(buf.obs)
            all_actions.extend(buf.actions)
            all_log_probs.extend(buf.log_probs)
            all_returns.extend(returns.tolist())
            all_advantages.extend(advantages.tolist())

        if not all_obs:
            return {}

        obs_t = torch.FloatTensor(np.array(all_obs)).to(self.device)
        act_t = torch.LongTensor(all_actions).to(self.device)
        old_lp_t = torch.FloatTensor(all_log_probs).to(self.device)
        ret_t = torch.FloatTensor(all_returns).to(self.device)
        adv_t = torch.FloatTensor(all_advantages).to(self.device)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n = len(all_obs)

        for _ in range(self.n_epochs):
            # Shuffle
            idx = torch.randperm(n)
            new_lp, values, entropy = self.model.evaluate(obs_t[idx], act_t[idx])
            ratio = torch.exp(new_lp - old_lp_t[idx])
            adv = adv_t[idx]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (values - ret_t[idx]).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += (-entropy_loss.item())

        for k in stats:
            stats[k] /= self.n_epochs
        return stats
