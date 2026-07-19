#!/usr/bin/env python3
"""
Phase 2: Reptile 元学习 TaskMLP + System 1 + Alpha 参数
GW (冻结) 作为教师提供 System 2 表征。
"""
import os
import sys
import yaml
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import DSNAModel
from src.env.multitask_wrapper import MultiTaskBabyAIEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--phase1_ckpt', required=True, help='Path to Phase 1 checkpoint')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_dir', default='outputs/phase2')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def collect_trajectory(model, env, n_envs, rollout_steps, device, training=True):
    """收集一个 rollout 的轨迹数据"""
    all_obs, all_instr, all_actions, all_logps = [], [], [], []
    all_rewards, all_values, all_dones = [], [], []
    all_info = defaultdict(list)

    obs_list, instr_list = env.reset(n_envs)
    model.reset_episode(n_envs)

    for _ in range(rollout_steps):
        obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)
        instr_t = torch.tensor(np.stack(instr_list), dtype=torch.long).to(device)

        with torch.no_grad():
            instr_emb = model.encoder.encode_instruction(instr_t)

        action_logits, value, info = model(obs_t, instr_emb, training=training)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, rewards, dones, _ = zip(*[env.step(a.item()) for a in action])

        all_obs.append(obs_t.cpu())
        all_instr.append(instr_t.cpu())
        all_actions.append(action.cpu())
        all_logps.append(log_prob.cpu())
        all_rewards.append(torch.tensor(rewards))
        all_values.append(value.detach().cpu())
        all_dones.append(torch.tensor(dones, dtype=torch.float32))

        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                all_info[k].append(v.detach().cpu())

        obs_list = next_obs

    cat = lambda x: torch.cat(x, dim=0)
    return (cat(all_obs), cat(all_instr), cat(all_actions), cat(all_logps),
            cat(all_rewards), cat(all_values), cat(all_dones), all_info)


def ppo_update_phase2(model, ppo_opt, batch_data, config, device):
    """Phase 2 PPO 更新 (仅更新可训练参数)"""
    obs_b, instr_b, act_b, old_lp_b, adv_b, ret_b, val_b = [d.to(device) for d in batch_data[:7]]
    info_b = batch_data[7]

    p = config['ppo']

    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

    # Forward (需要重新计算以获得梯度)
    model.reset_episode(obs_b.shape[0])
    # Simplified: process sequentially
    total_loss = 0
    for i in range(obs_b.shape[0]):
        o = obs_b[i:i+1]
        instr = instr_b[i:i+1]
        with torch.no_grad():
            instr_emb = model.encoder.encode_instruction(instr)
        _, _, info = model(o, instr_emb, training=True)

        # PPO loss on single step
        # (simplified - in production, batch this properly)
        alpha_loss = p['entropy_coef'] * (-info['entropy'].mean())

        # Alpha penalty
        lambda_alpha = config['loss']['lambda_alpha']
        alpha_penalty = lambda_alpha * F.relu(info['alpha_raw']).mean()

        total_loss += alpha_loss + alpha_penalty

    total_loss /= obs_b.shape[0]

    ppo_opt.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.get_phase2_trainable_params(), p['max_grad_norm'])
    ppo_opt.step()

    return {'loss': total_loss.item(), 'alpha_mean': info['alpha'].mean().item()}


def reptile_inner_loop(model, level, env, config, device, K_inner=5):
    """
    Reptile 内循环: 在单个关卡上 K 步快速适配 fast_head。
    使用 Adam 优化器，每任务重新初始化。
    """
    t_cfg = config['training']['phase2']
    inner_opt = torch.optim.Adam(
        model.task_mlp.get_fast_params(),
        lr=t_cfg['lr_inner']
    )

    alpha_history = []

    for k in range(K_inner):
        obs_b, instr_b, actions_b, _, rewards_b, values_b, dones_b, info = \
            collect_trajectory(model, env, n_envs=4, rollout_steps=20, device=device)

        # 计算 PPO 损失 + alpha 惩罚
        advantages = torch.zeros_like(rewards_b)  # simplified
        returns = rewards_b  # simplified

        action_logits, value, info = model(obs_b.to(device), instr_b.to(device), training=True)

        log_probs = F.log_softmax(action_logits, dim=-1)
        new_logp = log_probs.gather(1, actions_b.unsqueeze(-1).to(device)).squeeze(-1)

        ratio = torch.exp(new_logp - new_logp.detach())
        policy_loss = -(ratio * advantages.to(device)).mean()
        value_loss = F.mse_loss(value, returns.to(device))

        lambda_alpha = config['loss']['lambda_alpha']
        alpha_penalty = lambda_alpha * F.relu(info['alpha_raw']).mean()

        inner_loss = policy_loss + 0.5 * value_loss + alpha_penalty

        inner_opt.zero_grad()
        inner_loss.backward()
        inner_opt.step()

        alpha_history.append(info['alpha'].mean().item())

    return alpha_history


def train_phase2(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载 Phase 1 预训练权重
    ckpt = torch.load(args.phase1_ckpt, map_location=device)

    # 创建 Phase 2 模型
    model = DSNAModel(config, mode='dual').to(device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.freeze_for_phase2()

    env = MultiTaskBabyAIEnv(config['env']['levels'])
    levels = config['env']['levels']

    # Phase 2 可训练参数
    trainable = model.get_phase2_trainable_params()
    outer_opt = torch.optim.Adam(trainable, lr=config['training']['phase2']['lr_outer'])

    t_cfg = config['training']['phase2']
    total_meta_iters = t_cfg['total_meta_iters']
    K_inner = t_cfg['K_inner']
    epsilon_meta = t_cfg['epsilon_meta']
    tasks_per_batch = t_cfg.get('tasks_per_batch', 4)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'phase2_checkpoint.pt')

    for meta_iter in range(total_meta_iters):
        # 采样一批任务
        task_indices = np.random.choice(len(levels), size=tasks_per_batch, replace=False)

        meta_metrics = {'alpha_start': [], 'alpha_end': [], 'meta_loss': 0}

        for task_idx in task_indices:
            level = levels[task_idx]

            # 保存 slow_base 参数
            slow_params_init = model.task_mlp.clone_slow_params()

            # 重置 fast_head
            model.task_mlp.reset_episode()

            # === 内循环: K 步快速适配 ===
            alpha_history = reptile_inner_loop(model, level, env, config, device, K_inner)

            meta_metrics['alpha_start'].append(alpha_history[0])
            meta_metrics['alpha_end'].append(alpha_history[-1])

            # === 外循环: Reptile 更新 slow_base ===
            for name, param in model.task_mlp.named_parameters():
                if 'slow_base' in name:
                    param.data += epsilon_meta * (param.data - slow_params_init[name])

        # === 端到端训练 (防遗忘) ===
        batch_data = collect_trajectory(model, env, n_envs=8, rollout_steps=20, device=device)
        metrics = ppo_update_phase2(model, outer_opt, batch_data, config, device)
        meta_metrics['meta_loss'] = metrics['loss']

        # 保存
        if meta_iter % 500 == 0:
            torch.save({
                'model_state': model.state_dict(),
                'meta_iter': meta_iter,
                'config': config,
            }, save_path)
            print(f"[Phase 2] Meta iter {meta_iter:5d} | α: {meta_metrics['alpha_start'][0]:.3f}→"
                  f"{meta_metrics['alpha_end'][0]:.3f} | loss={meta_metrics['meta_loss']:.4f}")

    torch.save({
        'model_state': model.state_dict(),
        'meta_iter': total_meta_iters,
        'config': config,
    }, save_path)
    print(f"[Phase 2] Done! Model saved to {save_path}")


if __name__ == '__main__':
    args = parse_args()
    train_phase2(args)
