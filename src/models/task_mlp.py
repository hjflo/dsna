"""
TaskMLP: 拼接 S 路技能 GRU 状态 → (skill_logits, novelty_logit)
Fast/Slow 双速架构 + Fast Head 每 episode 重置
"""
import torch
import torch.nn as nn


class TaskMLP(nn.Module):
    """
    拼接 S 路技能 GRU 状态，输出 skill_logits 和 novelty_logit。
    
    slow_base: 跨任务共享，Reptile 元学习训练
    fast:      每 episode 重置，快速适配当前情境
    """
    def __init__(self, n_skills=8, skill_gru_dim=64, shared_dim=256, fast_dim=64, input_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = n_skills * skill_gru_dim  # v1: S × 64 = 512
        # v2: 512 + 32(task_emb) = 544

        # 慢速基座
        self.slow_base = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
        )

        # 快速头部 (每 episode 重置为零)
        self.fast = nn.Sequential(
            nn.Linear(shared_dim, fast_dim),
            nn.ReLU(),
        )
        self.skill_head = nn.Linear(fast_dim, n_skills)
        self.novelty_head = nn.Linear(fast_dim, 1)

        self._reset_fast_head()

    def _reset_fast_head(self):
        """将 fast_head 重置为近零 → 高熵+中低novelty → 初始依赖 GW"""
        for layer in self.fast:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        nn.init.zeros_(self.skill_head.weight)
        nn.init.zeros_(self.skill_head.bias)
        nn.init.zeros_(self.novelty_head.weight)
        nn.init.zeros_(self.novelty_head.bias)

    def reset_episode(self):
        """每个 episode 开始时重置 fast_head"""
        self._reset_fast_head()

    def get_slow_params(self):
        """返回慢速基座参数 (用于 Reptile 元更新)"""
        return list(self.slow_base.parameters())

    def get_fast_params(self):
        """返回快速头部参数 (用于内循环更新)"""
        return (list(self.fast.parameters()) +
                list(self.skill_head.parameters()) +
                list(self.novelty_head.parameters()))

    def clone_slow_params(self):
        """克隆慢速基座参数 (用于 Reptile 元更新前保存)"""
        return {n: p.clone() for n, p in self.slow_base.named_parameters()}

    def forward(self, x):
        """
        x: (B, input_dim) — 可以是 (S,B,D) 自动展平, 或预拼接的 (B, input_dim)
        returns: skill_logits (B, S), novelty_logit (B, 1)
        """
        if x.dim() == 3:
            # (S, B, D) → (B, S*D)
            S, B, D = x.shape
            x = x.permute(1, 0, 2).reshape(B, S * D)

        f_slow = self.slow_base(x)              # (B, shared_dim)
        f_fast = self.fast(f_slow)               # (B, fast_dim)

        skill_logits = self.skill_head(f_fast)   # (B, S)
        novelty_logit = self.novelty_head(f_fast)  # (B, 1)
        return skill_logits, novelty_logit
