#!/usr/bin/env python3
"""
DSNA v2: 联合训练 + 三阶段课程
S1/S2 各自独立采样, on-policy PPO.
"""
import sys, os, time, yaml, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.env.multitask_wrapper import MultiTaskBabyAIEnv, LEVEL_MAP
from src.models.dsna_model_v2 import DSNAModelV2, gumbel_sigmoid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/default_v2.yaml')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--save_every', type=int, default=5000)
    return p.parse_args()


def sample_curriculum(episode, cfg):
    """根据当前 episode 返回阶段索引和可用关卡列表"""
    curr = cfg['curriculum']
    if episode < curr['stage1_end']:
        stage, levels = 1, curr['stages'][1]
    elif episode < curr['stage2_end']:
        stage, levels = 2, curr['stages'][2]
    else:
        stage, levels = 3, curr['stages'][3]
    return stage, levels


def collect_rollout(model, env, task_id, n_envs, rollout_steps, device, system='s1'):
    """收集 S1 或 S2 的 rollout"""
    levels = env.levels
    idx = np.random.choice(len(levels))
    level = levels[idx]

    obs_list, instr_list = env.reset(n_envs)
    # 强制使用指定关卡 (简化: 用随机采样)
    if system == 's1':
        states, ihidden = model.reset_s1_states(n_envs, device)
    else:
        states, ihidden = model.reset_s2_states(n_envs, device)

    rollout = []
    for _ in range(rollout_steps):
        obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)
        instr_t = torch.tensor(np.stack(instr_list), dtype=torch.long).to(device)

        if system == 's1':
            a, v, info, states, ihidden = model.forward_s1(obs_t, instr_t, task_id, states, ihidden)
        else:
            a, v, info, states, ihidden = model.forward_s2(obs_t, instr_t, task_id, states, ihidden)

        dist = torch.distributions.Categorical(logits=a)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, rewards, dones, _ = env.step(action.tolist())
        rollout.append((obs_t.cpu(), instr_t.cpu(), action.cpu(), log_prob.cpu(),
                        v.cpu(), torch.tensor(rewards), torch.tensor(dones, dtype=torch.float32)))
        obs_list = next_obs
        if any(dones):
            obs_list, instr_list = env.reset(n_envs)
            if system == 's1':
                states, ihidden = model.reset_s1_states(n_envs, device)
            else:
                states, ihidden = model.reset_s2_states(n_envs, device)

    return rollout, info


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """计算 GAE advantages"""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t] if t+1 < len(values) else rewards[t] - values[t]
        gae = delta + gamma * gae_lambda * (1-dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def ppo_update(model, rollout, system, optimizer, cfg, device):
    """PPO 更新 (4 epoch, 每epoch从初始GRU状态重跑)"""
    ppo_cfg = cfg['ppo']
    n_envs = cfg['training']['n_envs']

    # 提取 rollout 数据
    obs_b = torch.cat([r[0] for r in rollout], dim=0).to(device)
    instr_b = torch.cat([r[1] for r in rollout], dim=0).to(device)
    act_b = torch.cat([r[2] for r in rollout], dim=0).to(device)
    old_logp_b = torch.cat([r[3] for r in rollout], dim=0).to(device)
    val_b = torch.cat([r[4] for r in rollout], dim=0).to(device)
    rew_b = torch.cat([r[5] for r in rollout], dim=0).to(device)
    dones_b = torch.cat([r[6] for r in rollout], dim=0).to(device)

    # GAE
    advantages, returns = compute_gae(rew_b, val_b, dones_b, ppo_cfg['gamma'], ppo_cfg['gae_lambda'])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages, returns = advantages.to(device), returns.to(device)

    # 找到初始 GRU 状态 (rollout 第一步前的状态)
    init_states, init_ihidden = model.reset_s1_states(n_envs, device) if system == 's1' else model.reset_s2_states(n_envs, device)
    task_id = 0  # 简化: 使用 task_id=0

    total_samples = obs_b.shape[0]
    indices = np.random.permutation(total_samples)

    for _ in range(ppo_cfg.get('ppo_epochs', 4)):
        states = init_states.clone().detach() if isinstance(init_states, torch.Tensor) else init_states
        ihidden = init_ihidden

        for start in range(0, total_samples, cfg['training'].get('batch_size', 256)):
            end = min(start + cfg['training'].get('batch_size', 256), total_samples)
            idx = indices[start:end]

            o_b = obs_b[idx]
            i_b = instr_b[idx]
            a_b = act_b[idx]
            olp_b = old_logp_b[idx]
            adv_b = advantages[idx]
            ret_b = returns[idx]

            # 重新前向 (从初始状态开始, 需要处理序列)
            # 简化: 单步处理
            if system == 's1':
                al, v, info, states, ihidden = model.forward_s1(o_b, i_b, task_id, states, ihidden)
            else:
                al, v, info, states, ihidden = model.forward_s2(o_b, i_b, task_id, states, ihidden)

            log_probs = F.log_softmax(al, dim=-1)
            new_logp = log_probs.gather(1, a_b.unsqueeze(-1)).squeeze(-1)

            ratio = torch.exp(new_logp - olp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1-ppo_cfg['clip_eps'], 1+ppo_cfg['clip_eps']) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(v, ret_b)
            entropy_bonus = ppo_cfg['entropy_coef'] * log_probs.exp().mul(log_probs).sum(-1).mean()

            loss = policy_loss + ppo_cfg['value_loss_coef'] * value_loss + entropy_bonus

            if system == 's1':
                # S1 sparse loss
                if 'skill_logits' in info:
                    probs = torch.sigmoid(info['skill_logits'])
                    ent = -(probs * torch.log(probs+1e-8) + (1-probs)*torch.log(1-probs+1e-8)).mean()
                    loss = loss + cfg['loss']['lambda_sparse'] * F.relu(-cfg['loss']['b_sparse'] + ent)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg['max_grad_norm'])
            optimizer.step()

            # detach 状态
            if isinstance(states, torch.Tensor):
                states = states.detach()
            if isinstance(ihidden, torch.Tensor):
                ihidden = ihidden.detach()

    return loss.item()


def evaluate(model, env, level, n_episodes, device, system='s1'):
    """评估指定系统在指定关卡的 success rate"""
    successes = 0
    n_envs = 4

    for _ in range(n_episodes // n_envs):
        obs_list, instr_list = env.reset(n_envs)
        if system == 's1':
            states, ihidden = model.reset_s1_states(n_envs, device)
        else:
            model.sync_encoder_s2()
            states, ihidden = model.reset_s2_states(n_envs, device)

        for _ in range(128):
            obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)
            instr_t = torch.tensor(np.stack(instr_list), dtype=torch.long).to(device)

            with torch.no_grad():
                if system == 's1':
                    a, v, info, states, ihidden = model.forward_s1(obs_t, instr_t, 0, states, ihidden)
                else:
                    a, v, info, states, ihidden = model.forward_s2(obs_t, instr_t, 0, states, ihidden)

            action = a.argmax(-1)
            next_obs, rewards, dones, _ = env.step(action.tolist())

            for i, (r, d) in enumerate(zip(rewards, dones)):
                if r > 0:
                    successes += 1

            obs_list = next_obs
            if all(dones):
                break

    return successes / n_episodes


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = MultiTaskBabyAIEnv(cfg['curriculum']['stages'][3], cfg['env']['max_steps_per_episode'])
    model = DSNAModelV2(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    t_cfg = cfg['training']

    os.makedirs('outputs/v2', exist_ok=True)
    log_path = 'outputs/v2/log.csv'
    ckpt_path = 'outputs/v2/checkpoint.pt'

    with open(log_path, 'w') as f:
        f.write('episode,stage,s1_sr,s2_sr,loss,entropy\n')

    t0 = time.time()
    for ep in range(t_cfg['total_episodes']):
        stage, levels = sample_curriculum(ep, cfg)
        task_id = stage - 1  # 0,1,2

        # sync S2 encoder before S1 sampling
        model.sync_encoder_s2()

        # === S1 采样 + 更新 ===
        rollout_s1, info_s1 = collect_rollout(model, env, task_id, t_cfg['n_envs'],
                                               t_cfg['rollout_steps'], device, 's1')
        loss_s1 = ppo_update(model, rollout_s1, 's1', opt, cfg, device)

        # === S2 采样 + 更新 ===
        model.sync_encoder_s2()
        rollout_s2, info_s2 = collect_rollout(model, env, task_id, t_cfg['n_envs'],
                                               t_cfg['rollout_steps'], device, 's2')
        loss_s2 = ppo_update(model, rollout_s2, 's2', opt, cfg, device)

        # === 评估 ===
        if ep > 0 and ep % cfg['eval']['interval'] == 0:
            s1_srs, s2_srs = [], []
            for level in levels:
                s1_sr = evaluate(model, env, level, cfg['eval']['episodes'], device, 's1')
                s2_sr = evaluate(model, env, level, cfg['eval']['episodes'], device, 's2')
                s1_srs.append(s1_sr)
                s2_srs.append(s2_sr)

            entropy_val = info_s1.get('skill_logits', torch.zeros(1)).sigmoid()
            ent = -(entropy_val * torch.log(entropy_val+1e-8) + (1-entropy_val)*torch.log(1-entropy_val+1e-8)).mean().item()

            print(f'Ep {ep:5d} | S{stage} | S1_sr={np.mean(s1_srs):.3f} | S2_sr={np.mean(s2_srs):.3f} | '
                  f'loss={loss_s1:.3f}/{loss_s2:.3f} | ent={ent:.3f} | {ep/(time.time()-t0):.1f} eps/s')

            with open(log_path, 'a') as f:
                f.write(f'{ep},{stage},{np.mean(s1_srs):.4f},{np.mean(s2_srs):.4f},{loss_s1:.4f},{ent:.4f}\n')

        if ep > 0 and ep % args.save_every == 0:
            torch.save({'model_state': model.state_dict(), 'episode': ep, 'config': cfg}, ckpt_path)

    torch.save({'model_state': model.state_dict(), 'episode': t_cfg['total_episodes'], 'config': cfg}, ckpt_path)
    print(f'\n✅ V2 训练完成! Saved to {ckpt_path}')


if __name__ == '__main__':
    main()
