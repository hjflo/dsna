"""
System 1: 活跃技能 Multi-Head Attention 成对交互
注意力参数在 Phase 2 中可训练。
"""
import torch
import torch.nn as nn


class System1_PairwiseSkillComm(nn.Module):
    """
    对活跃技能的 GRU 状态做 Multi-Head Attention。
    非活跃技能通过 key_padding_mask 被排除。
    """
    def __init__(self, skill_dim=64, n_heads=4):
        super().__init__()
        assert skill_dim % n_heads == 0, "skill_dim must be divisible by n_heads"
        self.skill_dim = skill_dim
        self.n_heads = n_heads

        self.W_q = nn.Linear(skill_dim, skill_dim, bias=False)
        self.W_k = nn.Linear(skill_dim, skill_dim, bias=False)
        self.W_v = nn.Linear(skill_dim, skill_dim, bias=False)
        self.W_o = nn.Linear(skill_dim, skill_dim, bias=False)

    def forward(self, skill_h, task_label):
        """
        skill_h:    (S, B, D)  所有技能 GRU 状态
        task_label: (B, S)     技能激活向量 (≈0/1)
        returns:    h_s1 (B, D)
        """
        S, B, D = skill_h.shape

        # 筛选活跃技能: 非活跃的用 mask 排除
        active_mask = (task_label > 0.5).bool()  # (B, S)

        # 转换为 (B, S, D) 做多头注意力
        skill_bt = skill_h.permute(1, 0, 2)  # (B, S, D)

        Q = self.W_q(skill_bt).view(B, S, self.n_heads, D // self.n_heads).transpose(1, 2)
        K = self.W_k(skill_bt).view(B, S, self.n_heads, D // self.n_heads).transpose(1, 2)
        V = self.W_v(skill_bt).view(B, S, self.n_heads, D // self.n_heads).transpose(1, 2)
        # Q,K,V: (B, n_heads, S, d_head)

        # 构建 mask: True = 忽略
        key_mask = ~active_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        key_mask = key_mask.expand(-1, self.n_heads, S, -1)  # (B, n_heads, S_query, S)

        scale = (D // self.n_heads) ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_scores = attn_scores.masked_fill(key_mask, -float('inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)  # 全mask时→0

        out = torch.matmul(attn_weights, V)  # (B, n_heads, S, d_head)
        out = out.transpose(1, 2).reshape(B, S, D)  # (B, S, D)
        out = self.W_o(out)

        # 仅活跃技能聚合
        active_count = active_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()  # (B, 1)
        active_mask_float = active_mask.float().unsqueeze(-1)  # (B, S, 1)
        h_s1 = (out * active_mask_float).sum(dim=1) / active_count  # (B, D)

        return h_s1
