"""
System 2: 共享全局工作空间 (对齐 Goyal et al., ICLR 2022)
- N_iter 次迭代写入
- S 路独立状态输出
- 可学习加权聚合 → h_s2
- RMC 风格门控记忆更新
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class System2_GlobalWorkspace(nn.Module):
    """
    对齐 Goyal et al. (2022) Section 2.1:
    - N_iter 次迭代写入 (原论文: "apply attention multiple times to distill information")
    - S 路独立状态输出 (原论文: 每个专家保持独立 trajectory)
    - 可学习加权聚合 → h_s2
    - RMC 风格门控记忆更新
    """
    def __init__(self, n_specialists=8, specialist_dim=64, n_slots=4,
                 top_k=2, n_write_iters=2):
        super().__init__()
        self.n_specialists = n_specialists
        self.n_slots = n_slots
        self.top_k = top_k
        self.n_write_iters = n_write_iters
        D = specialist_dim

        # 工作空间记忆槽位 (可学习初始值)
        self.memory_init = nn.Parameter(torch.randn(n_slots, D) * 0.02)
        self.register_buffer('memory', self.memory_init.clone())

        # 写入: Q 来自记忆, K/V 来自专家
        self.W_q_write = nn.Linear(D, D, bias=False)
        self.W_k_write = nn.Linear(D, D, bias=False)
        self.W_v_write = nn.Linear(D, D, bias=False)

        # 广播读取: Q 来自专家, K/V 来自记忆
        self.W_q_read = nn.Linear(D, D, bias=False)
        self.W_k_read = nn.Linear(D, D, bias=False)
        self.W_v_read = nn.Linear(D, D, bias=False)

        # RMC 风格门控更新
        self.gate = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(D)

        # 可学习聚合权重
        self.agg_weight = nn.Parameter(torch.zeros(n_specialists))

    def reset_episode(self):
        self.memory = self.memory_init.clone()

    def forward(self, specialist_states):
        """
        specialist_states: (S, B, D) — S 路技能 GRU 的隐藏状态
        returns:
          h_s2:      (B, D) — 加权聚合的全局表征
          h_updated: (S, B, D) — S 路各自独立更新后的状态
        """
        S, B, D = specialist_states.shape
        M = self.memory  # (n_slots, D) — 无 batch 维度，跨 batch 共享

        # ======== Step 1: N_iter 次迭代写入 ========
        for _ in range(self.n_write_iters):
            # Q: (n_slots, D), K: (S, B, D), V: (S, B, D)
            Q = self.W_q_write(M)                        # (n_slots, D)
            K = self.W_k_write(specialist_states)         # (S, B, D)
            V = self.W_v_write(specialist_states)         # (S, B, D)

            # scores: (n_slots, S, B) — 每个 batch item 独立计算
            Q_exp = Q.unsqueeze(-1)  # (n_slots, D, 1)
            scores = torch.einsum('sbd,ndk->nsb', K, Q_exp) / math.sqrt(D)

            # Top-k 硬竞争: 每槽位每 batch 选 k 个专家
            _, topk_idx = torch.topk(scores, self.top_k, dim=1)  # (n_slots, k, B)
            mask = torch.zeros_like(scores).scatter_(1, topk_idx, 1.0)
            weights = F.softmax(scores.masked_fill(mask == 0, -float('inf')), dim=1)

            # V_write: (n_slots, B, D)
            V_write = torch.einsum('nsb,sbd->nbd', weights, V)

            # 门控更新: 将 batch 平均后的 V_write 用于更新 M (保持 M 无 batch)
            V_write_mean = V_write.mean(dim=1)  # (n_slots, D)
            gate_val = self.gate(torch.cat([M, V_write_mean], dim=-1))
            M = (1 - gate_val) * M + gate_val * V_write_mean

        # 保存更新后的记忆
        self.memory = M.detach() + (M - M.detach())

        # ======== Step 2: 广播 — S 路各自独立读取 ========
        Q_read = self.W_q_read(specialist_states)           # (S, B, D)
        K_read = self.W_k_read(M).unsqueeze(1).expand(-1, B, -1)  # (n_slots, B, D)
        V_read = self.W_v_read(M).unsqueeze(1).expand(-1, B, -1)  # (n_slots, B, D)

        attn = F.softmax(
            torch.einsum('sbd,nbd->snb', Q_read, K_read) / math.sqrt(D), dim=1)
        broadcast_info = torch.einsum('snb,nbd->sbd', attn, V_read)
        h_updated = self.layer_norm(specialist_states + broadcast_info)  # (S, B, D)

        # ======== Step 3: 可学习加权聚合 ========
        w = F.softmax(self.agg_weight, dim=0)               # (S,)
        h_s2 = torch.einsum('s,sbd->bd', w, h_updated)      # (B, D)

        return h_s2, h_updated
