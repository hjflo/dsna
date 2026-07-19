"""
Unit tests for DSNA model components.
Run: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
import yaml


# Load test config
TEST_CONFIG = {
    'model': {
        'n_skills': 4,
        'skill_gru_dim': 64,
        'vision_dim': 128,
        'instr_dim': 128,
        'gw_n_slots': 2,
        'gw_top_k': 1,
        'gw_n_write_iters': 2,
        'task_mlp_shared_dim': 128,
        'task_mlp_fast_dim': 32,
        's1_n_heads': 2,
        'ac_hidden': 64,
    },
    'alpha': {'b_init': 1.5, 'gamma_init': 1.0, 'gumbel_tau': 1.0},
}


class TestEncoder:
    def test_forward_shape(self):
        from src.models.base_encoder import SkillEncoder
        enc = SkillEncoder(n_skills=4, skill_gru_dim=64, vision_dim=128, instr_dim=128)
        B = 2
        enc.reset_episode(B)
        obs = torch.randn(B, 7, 7, 3)
        instr_emb = torch.randn(B, 128)
        out = enc(obs, instr_emb)
        assert out.shape == (4, B, 64)

    def test_reset_episode(self):
        from src.models.base_encoder import SkillEncoder
        enc = SkillEncoder(n_skills=4, skill_gru_dim=64, vision_dim=128, instr_dim=128, vocab_size=200)
        enc.reset_episode(3)
        assert enc.skill_h.shape == (4, 3, 64)
        assert (enc.skill_h == 0).all()
    
    def test_encode_instruction(self):
        from src.models.base_encoder import SkillEncoder
        enc = SkillEncoder()
        instr = torch.randint(0, 100, (2, 16))
        emb = enc.encode_instruction(instr)
        assert emb.shape == (2, 128)


class TestTaskMLP:
    def test_forward_shape(self):
        from src.models.task_mlp import TaskMLP
        mlp = TaskMLP(n_skills=4, skill_gru_dim=64, shared_dim=128, fast_dim=32)
        skill_h = torch.randn(4, 2, 64)
        logits, novelty = mlp(skill_h)
        assert logits.shape == (2, 4)
        assert novelty.shape == (2, 1)

    def test_reset_episode(self):
        from src.models.task_mlp import TaskMLP
        mlp = TaskMLP(n_skills=4, skill_gru_dim=64, shared_dim=128, fast_dim=32)
        # Manually set weights to non-zero
        with torch.no_grad():
            mlp.skill_head.weight.fill_(1.0)
        mlp.reset_episode()
        # After reset, skill_head weight should be zeros
        assert (mlp.skill_head.weight == 0).all()


class TestAlphaController:
    def test_alpha_range(self):
        from src.models.alpha_controller import AlphaController
        ctrl = AlphaController(b_init=1.5, gamma_init=1.0)
        logits = torch.randn(8, 4)
        novelty = torch.randn(8, 1)
        alpha, raw, ent, nov = ctrl(logits, novelty)
        assert (alpha >= 0).all() and (alpha <= 1).all()
        assert (ent >= 0).all()

    def test_high_entropy_high_alpha(self):
        from src.models.alpha_controller import AlphaController
        ctrl = AlphaController(b_init=0.0, gamma_init=0.0)  # b=0, γ=0
        logits = torch.zeros(1, 4)  # p=0.5 → max entropy
        novelty = torch.zeros(1, 1)
        alpha, _, _, _ = ctrl(logits, novelty)
        assert alpha.item() > 0.5  # high entropy → high alpha


class TestSystem1:
    def test_forward_shape(self):
        from src.models.system1_pairwise import System1_PairwiseSkillComm
        s1 = System1_PairwiseSkillComm(skill_dim=64, n_heads=2)
        skill_h = torch.randn(4, 2, 64)
        task_label = torch.tensor([[1., 0., 1., 0.], [0., 1., 1., 0.]])
        h_s1 = s1(skill_h, task_label)
        assert h_s1.shape == (2, 64)

    def test_all_inactive_handled(self):
        from src.models.system1_pairwise import System1_PairwiseSkillComm
        s1 = System1_PairwiseSkillComm(skill_dim=64, n_heads=2)
        skill_h = torch.randn(4, 2, 64)
        task_label = torch.zeros(2, 4)  # no active skills
        h_s1 = s1(skill_h, task_label)
        assert h_s1.shape == (2, 64)
        assert not torch.isnan(h_s1).any()


class TestSystem2:
    def test_forward_shape(self):
        from src.models.system2_workspace import System2_GlobalWorkspace
        gw = System2_GlobalWorkspace(n_specialists=4, specialist_dim=64, n_slots=2,
                                     top_k=1, n_write_iters=2)
        x = torch.randn(4, 2, 64)
        h_s2, h_updated = gw(x)
        assert h_s2.shape == (2, 64)
        assert h_updated.shape == (4, 2, 64)
        assert not torch.isnan(h_s2).any()

    def test_reset_episode(self):
        from src.models.system2_workspace import System2_GlobalWorkspace
        gw = System2_GlobalWorkspace(n_specialists=4, specialist_dim=64, n_slots=2)
        gw.reset_episode()
        init_mem = gw.memory.clone()
        x = torch.randn(4, 2, 64)
        _ = gw(x)
        assert not torch.allclose(gw.memory, init_mem)


class TestDSNAModel:
    def test_phase1_forward(self):
        from src.models.dsna_model import DSNAModel
        model = DSNAModel(TEST_CONFIG, mode='gw_only')
        B = 2
        model.reset_episode(B)
        obs = torch.randn(B, 7, 7, 3)
        instr_tokens = torch.randint(0, 100, (B, 16))
        instr_emb = model.encoder.encode_instruction(instr_tokens)
        action, value = model(obs, instr_emb)
        assert action.shape == (B, 7)
        assert value.shape == (B,)

    def test_phase2_forward(self):
        from src.models.dsna_model import DSNAModel
        model = DSNAModel(TEST_CONFIG, mode='dual')
        B = 2
        model.reset_episode(B)
        obs = torch.randn(B, 7, 7, 3)
        instr_tokens = torch.randint(0, 100, (B, 16))
        instr_emb = model.encoder.encode_instruction(instr_tokens)
        action, value, info = model(obs, instr_emb)
        assert action.shape == (B, 7)
        assert value.shape == (B,)
        assert 'alpha' in info
        assert info['alpha'].shape == (B,)

    def test_freeze_phase2(self):
        from src.models.dsna_model import DSNAModel
        model = DSNAModel(TEST_CONFIG, mode='dual')
        model.freeze_for_phase2()
        assert not any(p.requires_grad for p in model.encoder.parameters())
        assert not any(p.requires_grad for p in model.gw.parameters())
        assert not any(p.requires_grad for p in model.ac_head.parameters())
        # TaskMLP + S1 + alpha should still be trainable
        assert any(p.requires_grad for p in model.task_mlp.parameters())
        assert any(p.requires_grad for p in model.system1.parameters())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
