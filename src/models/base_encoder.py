"""
DSNA Encoder: S 路独立技能 GRU + 共享 CNN+FiLM + 共享指令 GRU
Phase 1 训练，Phase 2 冻结。
"""
import torch
import torch.nn as nn


class CNNWithFiLM(nn.Module):
    """CNN + FiLM: 用指令嵌入对视觉特征做特征级条件调制"""
    def __init__(self, vision_dim=128, instr_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, vision_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        # FiLM: 从指令生成 scale 和 shift
        self.film_scale = nn.Linear(instr_dim, vision_dim)
        self.film_shift = nn.Linear(instr_dim, vision_dim)

    def forward(self, obs, instr_emb):
        """
        obs: (B, 7, 7, 3) -> permute to (B, 3, 7, 7)
        instr_emb: (B, instr_dim)
        returns: (B, vision_dim)
        """
        x = obs.float().permute(0, 3, 1, 2)  # (B, 3, 7, 7)
        v = self.cnn(x)

        # FiLM modulation
        scale = self.film_scale(instr_emb)
        shift = self.film_shift(instr_emb)
        v = v * (1 + scale) + shift
        return v


class SkillEncoder(nn.Module):
    """
    S 路独立技能 GRU 编码器。无内部状态存储——调用者负责状态管理。
    每路 GRU 从共享的视觉+指令特征中提取技能相关信息。
    """
    def __init__(self, n_skills=8, skill_gru_dim=64, vision_dim=128, instr_dim=128, vocab_size=1000):
        super().__init__()
        self.n_skills = n_skills
        self.skill_gru_dim = skill_gru_dim
        self.instr_dim = instr_dim

        self.cnn_film = CNNWithFiLM(vision_dim, instr_dim)
        self.instr_embed = nn.Embedding(vocab_size, instr_dim)
        self.instr_gru = nn.GRU(instr_dim, instr_dim, batch_first=True)

        # S 路独立技能 GRU
        input_dim = vision_dim + instr_dim
        self.skill_grus = nn.ModuleList([
            nn.GRUCell(input_dim, skill_gru_dim) for _ in range(n_skills)
        ])

    def init_skill_states(self, batch_size, device):
        """返回初始化的技能 GRU 状态 (调用者负责管理)"""
        return torch.zeros(self.n_skills, batch_size, self.skill_gru_dim, device=device)

    def init_instr_state(self):
        return None

    def encode_instruction(self, instr_tokens, instr_hidden=None):
        """
        instr_tokens: (B, seq_len) Long tensor
        instr_hidden: 可选的上一步隐藏状态
        returns: instr_emb (B, instr_dim), new_hidden
        """
        emb = self.instr_embed(instr_tokens)
        _, new_hidden = self.instr_gru(emb, instr_hidden)
        return new_hidden.squeeze(0), new_hidden

    def forward(self, obs, instr_emb, skill_states):
        """
        obs: (B, 7, 7, 3)
        instr_emb: (B, instr_dim)
        skill_states: (S, B, skill_gru_dim) — 上一步的技能 GRU 状态
        returns: new_skill_states (S, B, skill_gru_dim)
        """
        v_feat = self.cnn_film(obs, instr_emb)
        combined = torch.cat([v_feat, instr_emb], dim=-1)

        new_states = []
        for i, gru in enumerate(self.skill_grus):
            h_i = gru(combined, skill_states[i])
            new_states.append(h_i)
        return torch.stack(new_states, dim=0)
