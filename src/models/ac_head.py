"""
Actor-Critic 头: h_t → (action_logits, value)
Phase 1 训练，Phase 2 冻结。
"""
import torch
import torch.nn as nn


class ActorCriticHead(nn.Module):
    """
    两个独立 MLP:
      - Actor:  h_t → action_logits (7 维, BabyAI 动作空间)
      - Critic: h_t → value (1 维)
    """
    def __init__(self, input_dim=64, n_actions=7, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h):
        """
        h: (B, input_dim)
        returns: action_logits (B, n_actions), value (B,)
        """
        return self.actor(h), self.critic(h).squeeze(-1)
