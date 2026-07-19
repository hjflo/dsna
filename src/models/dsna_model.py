"""
DSNA 主模型: 组合 Encoder + TaskMLP + Alpha + S1 + S2 + AC Head
支持 Phase 1 (GW-only) 和 Phase 2 (full dual-system) 两种模式。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_encoder import SkillEncoder
from .task_mlp import TaskMLP
from .alpha_controller import AlphaController
from .system1_pairwise import System1_PairwiseSkillComm
from .system2_workspace import System2_GlobalWorkspace
from .ac_head import ActorCriticHead


class DSNAModel(nn.Module):
    """
    Dual-System Neural Architecture.
    
    Phase 1 (mode='gw_only'):
      Encoder → GW → AC Head → action/value
      训练后全部冻结。
    
    Phase 2 (mode='dual'):
      Encoder(frozen) → TaskMLP + S1(trainable) + S2(frozen) → α fusion → AC Head(frozen)
    """
    def __init__(self, config, mode='gw_only'):
        super().__init__()
        cfg = config['model']
        self.mode = mode
        self.n_skills = cfg['n_skills']
        self.skill_dim = cfg['skill_gru_dim']

        # 共享组件
        self.encoder = SkillEncoder(
            n_skills=cfg['n_skills'],
            skill_gru_dim=cfg['skill_gru_dim'],
            vision_dim=cfg['vision_dim'],
            instr_dim=cfg['instr_dim'],
            vocab_size=cfg.get('vocab_size', 1000),
        )
        self.gw = System2_GlobalWorkspace(
            n_specialists=cfg['n_skills'],
            specialist_dim=cfg['skill_gru_dim'],
            n_slots=cfg['gw_n_slots'],
            top_k=cfg['gw_top_k'],
            n_write_iters=cfg['gw_n_write_iters'],
        )
        self.ac_head = ActorCriticHead(
            input_dim=cfg['skill_gru_dim'],
            n_actions=7,
            hidden=cfg.get('ac_hidden', 128),
        )

        # Phase 2 专用组件
        if mode == 'dual':
            self.task_mlp = TaskMLP(
                n_skills=cfg['n_skills'],
                skill_gru_dim=cfg['skill_gru_dim'],
                shared_dim=cfg.get('task_mlp_shared_dim', 256),
                fast_dim=cfg.get('task_mlp_fast_dim', 64),
            )
            self.system1 = System1_PairwiseSkillComm(
                skill_dim=cfg['skill_gru_dim'],
                n_heads=cfg.get('s1_n_heads', 4),
            )
            self.alpha_ctrl = AlphaController(
                b_init=config.get('alpha', {}).get('b_init', 1.5),
                gamma_init=config.get('alpha', {}).get('gamma_init', 1.0),
            )
            self.gumbel_tau = config.get('alpha', {}).get('gumbel_tau', 1.0)

    def reset_episode(self, batch_size):
        self.encoder.reset_episode(batch_size)
        self.gw.reset_episode()
        if self.mode == 'dual':
            self.task_mlp.reset_episode()

    def freeze_for_phase2(self):
        """Phase 2 前冻结 Encoder, GW, AC Head"""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.gw.parameters():
            p.requires_grad = False
        for p in self.ac_head.parameters():
            p.requires_grad = False

    def get_phase2_trainable_params(self):
        """返回 Phase 2 可训练参数: TaskMLP + S1 + b/γ"""
        params = []
        params += list(self.task_mlp.parameters())
        params += list(self.system1.parameters())
        params += list(self.alpha_ctrl.parameters())
        return params

    def forward_phase1(self, obs, instr_emb):
        """Phase 1: GW-only forward"""
        skill_h = self.encoder(obs, instr_emb)          # (S, B, 64)
        h_s2, _ = self.gw(skill_h)                       # (B, 64)
        action, value = self.ac_head(h_s2)               # (B,7), (B,)
        return action, value, h_s2

    def forward_phase2(self, obs, instr_emb, training=True):
        """Phase 2: dual-system forward with alpha gating"""
        B = obs.shape[0]

        with torch.no_grad():
            skill_h = self.encoder(obs, instr_emb)       # (S, B, 64)

        # TaskMLP → skill_logits, novelty_logit
        skill_logits, novelty_logit = self.task_mlp(skill_h)

        # GumbelSigmoid → task_label
        if training:
            # Manual gumbel_sigmoid (compatible with older PyTorch)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(skill_logits) + 1e-8) + 1e-8)
            task_label = torch.sigmoid((skill_logits + gumbel_noise) / self.gumbel_tau)
        else:
            task_label = (skill_logits > 0).float()

        # Alpha
        alpha, alpha_raw, entropy, novelty = self.alpha_ctrl(skill_logits, novelty_logit)

        # System 1 (trainable in Phase 2)
        h_s1 = self.system1(skill_h, task_label)          # (B, 64)

        # System 2 (frozen)
        with torch.no_grad():
            h_s2, _ = self.gw(skill_h)                    # (B, 64)

        # Alpha 融合
        alpha_exp = alpha.unsqueeze(-1)                    # (B, 1)
        h_t = (1 - alpha_exp) * h_s1 + alpha_exp * h_s2    # (B, 64)

        # AC Head (frozen, gradient passes through)
        action, value = self.ac_head(h_t)                  # (B,7), (B,)

        info = {
            'alpha': alpha,
            'alpha_raw': alpha_raw,
            'entropy': entropy,
            'novelty': novelty,
            'task_label': task_label,
            'h_s1': h_s1,
            'h_s2': h_s2,
        }
        return action, value, info

    def forward(self, obs, instr_emb, training=True):
        if self.mode == 'gw_only':
            action, value, _ = self.forward_phase1(obs, instr_emb)
            return action, value
        else:
            action, value, info = self.forward_phase2(obs, instr_emb, training)
            return action, value, info
