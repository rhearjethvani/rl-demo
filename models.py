"""
Neural network models for PAIRED.

- AgentNet: protagonist / antagonist policy (Conv + FC + policy/value heads)
- AdversaryNet: environment-generating adversary policy (FC + policy/value heads)

All policies are trained with PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from env import OBS_DIM, N_ACTIONS, ADV_ACTION_DIM, ADV_OBS_DIM, OBS_CHANNELS, OBS_VIEW


class AgentNet(nn.Module):
    """
    Policy network for protagonist / antagonist.
    Input: flat observation (OBS_DIM,)
    Output: action logits (N_ACTIONS,) and value estimate (1,)
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(OBS_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 16 * OBS_VIEW * OBS_VIEW
        self.fc = nn.Sequential(
            nn.Linear(conv_out + 1, hidden),  # +1 for direction
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def _parse_obs(self, obs: torch.Tensor):
        """Split flat obs into (image, direction)."""
        img_flat = obs[:, :-1]
        direction = obs[:, -1:]
        img = img_flat.view(-1, OBS_CHANNELS, OBS_VIEW, OBS_VIEW)
        return img, direction

    def forward(self, obs: torch.Tensor):
        img, direction = self._parse_obs(obs)
        x = self.conv(img)
        x = torch.cat([x, direction], dim=-1)
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor):
        """Sample action and return (action, log_prob, value)."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate stored actions for PPO update."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy


class AdversaryNet(nn.Module):
    """
    Policy network for the environment-generating adversary.
    Input: flat adversary observation (ADV_OBS_DIM,)
    Output: action logits over grid positions (ADV_ACTION_DIM,) and value (1,)
    """

    def __init__(
        self,
        obs_dim: int = ADV_OBS_DIM,
        action_dim: int = ADV_ACTION_DIM,
        hidden: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy
