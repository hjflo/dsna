"""
Alpha 控制器: α_raw = -b + entropy + γ·novelty
b, γ 可学习。novelty 来自 TaskMLP 的 novelty_logit (无历史缓存)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaController(nn.Module):
    """
    α = σ(-b + entropy(skill_logits) + γ·σ(novelty_logit))
    
    学习动态:
      - ReLU(α_raw) 推动 b↑ → 默认 α 更低 → 偏向 System 1
      - PPO loss 在需要 GW 时通过梯度推动 entropy/novelty↑
    """
    def __init__(self, b_init=1.5, gamma_init=1.0):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(b_init))
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def compute_entropy(self, skill_logits):
        """二元熵 (每技能独立 Bernoulli)"""
        probs = torch.sigmoid(skill_logits)  # (B, S)
        eps = 1e-8
        entropy = -(probs * torch.log(probs + eps) +
                    (1 - probs) * torch.log(1 - probs + eps))
        return entropy.mean(dim=-1)  # (B,)

    def forward(self, skill_logits, novelty_logit):
        """
        skill_logits:  (B, S)
        novelty_logit: (B, 1)
        returns: alpha (B,), alpha_raw (B,), entropy (B,), novelty (B,)
        """
        entropy = self.compute_entropy(skill_logits)              # (B,)
        novelty = torch.sigmoid(novelty_logit).squeeze(-1)        # (B,)

        alpha_raw = -self.b + entropy + self.gamma * novelty      # (B,)
        alpha = torch.sigmoid(alpha_raw)                           # (B,)
        return alpha, alpha_raw, entropy, novelty
