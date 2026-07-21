"""
DSNA v2 主模型: 共享 Encoder 权重 + 各自独立 GRU 状态
S1: Encoder→TaskMLP→MHA→AC1
S2: Encoder(detach)→GW→AC2
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_encoder import SkillEncoder
from .task_mlp import TaskMLP
from .system1_pairwise import System1_PairwiseSkillComm
from .system2_workspace import System2_GlobalWorkspace
from .ac_head import ActorCriticHead


class DSNAModelV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config['model']
        self.n_skills = cfg['n_skills']
        self.skill_dim = cfg['skill_gru_dim']

        # === 共享 Encoder (S1训练, S2只读) ===
        self.encoder = SkillEncoder(
            n_skills=cfg['n_skills'],
            skill_gru_dim=cfg['skill_gru_dim'],
            vision_dim=cfg['vision_dim'],
            instr_dim=cfg['instr_dim'],
            vocab_size=cfg.get('vocab_size', 1000),
        )
        # S2 独立 Encoder 实例 (共享权重, 独立GRU状态)
        self.encoder_s2 = copy.deepcopy(self.encoder)

        # === S1 组件 ===
        task_input_dim = cfg['n_skills'] * cfg['skill_gru_dim'] + cfg.get('task_emb_dim', 32)
        self.task_mlp = TaskMLP(
            n_skills=cfg['n_skills'],
            skill_gru_dim=cfg['skill_gru_dim'],
            shared_dim=cfg.get('task_mlp_shared_dim', 256),
            fast_dim=cfg.get('task_mlp_fast_dim', 64),
            input_dim=task_input_dim,
        )
        self.s1_mha = System1_PairwiseSkillComm(
            skill_dim=cfg['skill_gru_dim'],
            n_heads=cfg.get('s1_n_heads', 4),
        )
        ac_input_dim = cfg['skill_gru_dim'] + cfg.get('task_emb_dim', 32)
        self.ac_s1 = ActorCriticHead(
            input_dim=ac_input_dim,
            n_actions=7,
            hidden=cfg.get('ac_hidden', 128),
        )

        # === S2 组件 ===
        self.s2_gw = System2_GlobalWorkspace(
            n_specialists=cfg['n_skills'],
            specialist_dim=cfg['skill_gru_dim'],
            n_slots=cfg['gw_n_slots'],
            top_k=cfg['gw_top_k'],
            n_write_iters=cfg.get('gw_n_write_iters', 1),
        )
        self.ac_s2 = ActorCriticHead(
            input_dim=ac_input_dim,
            n_actions=7,
            hidden=cfg.get('ac_hidden', 128),
        )

        # === Task Embedding ===
        self.task_emb = nn.Embedding(cfg.get('n_levels', 8), cfg.get('task_emb_dim', 32))
        self.task_emb_dim = cfg.get('task_emb_dim', 32)

    def sync_encoder_s2(self):
        """S2 Encoder 权重同步 (detach, S2不训练Encoder)"""
        self.encoder_s2.load_state_dict(self.encoder.state_dict())

    def reset_s1_states(self, batch_size, device='cpu'):
        return self.encoder.init_skill_states(batch_size, device), None

    def reset_s2_states(self, batch_size, device='cpu'):
        return self.encoder_s2.init_skill_states(batch_size, device), None

    def forward_s1(self, obs, instr_tokens, task_id, skill_states, instr_hidden):
        """
        S1 前向 (有梯度到 Encoder+Skill+TaskMLP+MHA+AC1)
        返回: action_logits, value, info_dict, new_states
        """
        B = obs.shape[0]
        device = obs.device
        e_task = self.task_emb(torch.full((B,), task_id, device=device, dtype=torch.long))

        # Encoder
        instr_emb, instr_hidden = self.encoder.encode_instruction(instr_tokens, instr_hidden)
        skill_states = self.encoder(obs, instr_emb, skill_states)

        # TaskMLP
        h_flat = skill_states.permute(1, 0, 2).reshape(B, -1)
        skill_logits, novelty_logit = self.task_mlp(torch.cat([h_flat, e_task], dim=-1))
        task_label = gumbel_sigmoid(skill_logits, hard=False) if self.training else (skill_logits > 0).float()

        # MHA
        h_s1 = self.s1_mha(skill_states, task_label)

        # AC Head
        a, v = self.ac_s1(torch.cat([h_s1, e_task], dim=-1))

        info = {'skill_logits': skill_logits, 'task_label': task_label, 'h_s1': h_s1}
        return a, v, info, skill_states, instr_hidden

    def forward_s2(self, obs, instr_tokens, task_id, skill_states, instr_hidden):
        """
        S2 前向 (梯度不回传 Encoder — encoder_s2权重在optimizer外)
        返回: action_logits, value, info_dict, new_states
        """
        B = obs.shape[0]
        device = obs.device
        e_task = self.task_emb(torch.full((B,), task_id, device=device, dtype=torch.long))

        # Encoder (encoder_s2, 独立GRU状态)
        instr_emb, instr_hidden = self.encoder_s2.encode_instruction(instr_tokens, instr_hidden)
        skill_states = self.encoder_s2(obs, instr_emb, skill_states)

        # GW (全8路竞争)
        h_s2, _ = self.s2_gw(skill_states)

        # AC Head
        a, v = self.ac_s2(torch.cat([h_s2, e_task], dim=-1))

        info = {'h_s2': h_s2}
        return a, v, info, skill_states, instr_hidden


def gumbel_sigmoid(logits, tau=1.0, hard=False):
    """Manual gumbel-sigmoid (compatible with all PyTorch versions)"""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    y = torch.sigmoid((logits + gumbel_noise) / tau)
    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard - y.detach() + y
    return y
