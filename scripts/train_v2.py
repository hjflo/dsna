#!/usr/bin/env python3
"""
DSNA v2: GPU优化版 — 联合训练 + 三阶段课程
- 异步 CPU→GPU 传输 (pin_memory + non_blocking)
- 混合精度 (AMP)
- 批量环境步进
- 预分配 GPU 张量
"""
import sys, os, time, yaml, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.env.multitask_wrapper import MultiTaskBabyAIEnv
from src.models.dsna_model_v2 import DSNAModelV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/default_v2.yaml')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--save_every', type=int, default=5000)
    p.add_argument('--mixed_precision', action='store_true', default=True,
                   help='Use FP16 mixed precision')
    return p.parse_args()


def sample_curriculum(episode, cfg):
    curr = cfg['curriculum']
    if episode < curr['stage1_end']:       return 1, curr['stages'][1]
    elif episode < curr['stage2_end']:     return 2, curr['stages'][2]
    else:                                   return 3, curr['stages'][3]


@torch.no_grad()
def collect_rollout(model, env, task_id, n_envs, rollout_steps, device, system='s1'):
    """收集 rollout — 批量环境步进, GPU 直接存储"""
    B = n_envs
    obs_list, instr_list = env.reset(B)
    states, ihidden = (model.reset_s1_states(B, device) if system == 's1' 
                       else model.reset_s2_states(B, device))
    
    # 预分配 GPU 存储 (避免每次 cat)
    all_obs = torch.zeros(rollout_steps, B, 7, 7, 3, dtype=torch.float32, device=device)
    all_instr = torch.zeros(rollout_steps, B, 32, dtype=torch.long, device=device)
    all_act = torch.zeros(rollout_steps, B, dtype=torch.long, device=device)
    all_logp = torch.zeros(rollout_steps, B, device=device)
    all_val = torch.zeros(rollout_steps, B, device=device)
    all_rew = torch.zeros(rollout_steps, B, device=device)
    all_done = torch.zeros(rollout_steps, B, device=device)
    all_skill_logits = torch.zeros(rollout_steps, B, model.n_skills, device=device)
    
    for t in range(rollout_steps):
        # 异步传输到 GPU
        obs_t = torch.from_numpy(np.stack(obs_list)).to(device=device, dtype=torch.float32, non_blocking=True)
        instr_t = torch.from_numpy(np.stack(instr_list)).to(device=device, dtype=torch.long, non_blocking=True)
        
        if system == 's1':
            a, v, info, states, ihidden = model.forward_s1(obs_t, instr_t, task_id, states, ihidden)
        else:
            a, v, info, states, ihidden = model.forward_s2(obs_t, instr_t, task_id, states, ihidden)
        
        dist = torch.distributions.Categorical(logits=a)
        action = dist.sample()
        
        next_obs, rewards, dones, _ = env.step(action.cpu().tolist())
        
        all_obs[t] = obs_t
        all_instr[t] = instr_t
        all_act[t] = action
        all_logp[t] = dist.log_prob(action)
        all_val[t] = v
        all_rew[t] = torch.tensor(rewards, dtype=torch.float32, device=device)
        all_done[t] = torch.tensor(dones, dtype=torch.float32, device=device)
        if system == 's1' and 'skill_logits' in info:
            all_skill_logits[t] = info['skill_logits']
        
        obs_list = next_obs
        if any(dones):
            obs_list, instr_list = env.reset(B)
            states, ihidden = (model.reset_s1_states(B, device) if system == 's1'
                              else model.reset_s2_states(B, device))
    
    # 展平为 (T*B, ...) 用于 PPO
    flat = lambda x: x.reshape(-1, *x.shape[2:])
    return (flat(all_obs), flat(all_instr), flat(all_act), flat(all_logp),
            flat(all_val), flat(all_rew), flat(all_done), flat(all_skill_logits))


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """GPU 上的向量化 GAE"""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t+1] if t+1 < T else 0.0
        delta = rewards[t] + gamma * next_val * (1-dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1-dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def ppo_update(model, rollout_data, system, optimizer, scaler, cfg, device, use_amp):
    """PPO 更新 — 混合精度 + 大batch"""
    obs_b, instr_b, act_b, old_logp_b, val_b, rew_b, dones_b, skill_b = rollout_data
    ppo_cfg, t_cfg = cfg['ppo'], cfg['training']
    total = obs_b.shape[0]
    
    # GAE
    adv_b, ret_b = compute_gae(rew_b, val_b, dones_b, ppo_cfg['gamma'], ppo_cfg['gae_lambda'])
    
    for epoch in range(ppo_cfg.get('ppo_epochs', 4)):
        perm = torch.randperm(total, device=device)
        for start in range(0, total, t_cfg.get('batch_size', 512)):
            idx = perm[start:start + t_cfg.get('batch_size', 512)]
            o, i, a, olp, adv, ret = obs_b[idx], instr_b[idx], act_b[idx], old_logp_b[idx], adv_b[idx], ret_b[idx]
            
            with autocast(device_type=device, enabled=use_amp):
                # 重新前向
                states, ihidden = model.reset_s1_states(len(idx), device) if system == 's1' else model.reset_s2_states(len(idx), device)
                if system == 's1':
                    al, v, info, _, _ = model.forward_s1(o, i, 0, states, ihidden)
                else:
                    al, v, info, _, _ = model.forward_s2(o, i, 0, states, ihidden)
                
                logp = F.log_softmax(al, dim=-1)
                new_logp = logp.gather(1, a.unsqueeze(-1)).squeeze(-1)
                
                ratio = torch.exp(new_logp - olp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-ppo_cfg['clip_eps'], 1+ppo_cfg['clip_eps']) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(v, ret)
                ent = ppo_cfg['entropy_coef'] * logp.exp().mul(logp).sum(-1).mean()
                loss = policy_loss + ppo_cfg['value_loss_coef'] * value_loss + ent
                
                if system == 's1':
                    sk = skill_b[idx]
                    probs = torch.sigmoid(sk)
                    ent_sp = -(probs*torch.log(probs+1e-8)+(1-probs)*torch.log(1-probs+1e-8)).mean()
                    loss = loss + cfg['loss']['lambda_sparse'] * F.relu(-cfg['loss']['b_sparse'] + ent_sp)
            
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg['max_grad_norm'])
                optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, env, levels, n_episodes, device, system='s1'):
    """GPU 批量评估"""
    n_envs = min(16, n_episodes)
    total_success, total_episodes = 0, 0
    
    for level in levels:
        successes = 0
        for _ in range(max(1, n_episodes // n_envs)):
            obs_list, instr_list = env.reset(n_envs)
            states, ihidden = (model.reset_s1_states(n_envs, device) if system == 's1'
                              else model.reset_s2_states(n_envs, device))
            if system == 's2':
                model.sync_encoder_s2()
            
            for _ in range(128):
                obs_t = torch.from_numpy(np.stack(obs_list)).to(device, dtype=torch.float32, non_blocking=True)
                instr_t = torch.from_numpy(np.stack(instr_list)).to(device, dtype=torch.long, non_blocking=True)
                
                if system == 's1':
                    a, v, info, states, ihidden = model.forward_s1(obs_t, instr_t, 0, states, ihidden)
                else:
                    a, v, info, states, ihidden = model.forward_s2(obs_t, instr_t, 0, states, ihidden)
                
                actions = a.argmax(-1).cpu().tolist()
                next_obs, rewards, dones, _ = env.step(actions)
                successes += sum(1 for r in rewards if r > 0)
                obs_list = next_obs
                if all(dones): break
        
        total_success += successes
        total_episodes += n_episodes
    
    return total_success / max(total_episodes, 1)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(args.device)
    use_amp = args.mixed_precision and device.type == 'cuda'
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    
    all_levels = cfg['curriculum']['stages'][3]
    env = MultiTaskBabyAIEnv(all_levels, cfg['env']['max_steps_per_episode'])
    
    model = DSNAModelV2(cfg).to(device)
    if use_amp:
        model = torch.compile(model, mode='reduce-overhead')  # CUDA graph optimization
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'])
    scaler = GradScaler(device_type=device.type, enabled=use_amp)
    t_cfg = cfg['training']
    
    os.makedirs('outputs/v2', exist_ok=True)
    log_path, ckpt_path = 'outputs/v2/log.csv', 'outputs/v2/checkpoint.pt'
    with open(log_path, 'w') as f:
        f.write('episode,stage,s1_sr,s2_sr,loss,eps_per_sec\n')
    
    t0 = time.time()
    for ep in range(t_cfg['total_episodes']):
        stage, levels = sample_curriculum(ep, cfg)
        task_id = stage - 1
        
        # S2 权重同步 (S1 更新前)
        model.sync_encoder_s2()
        
        # === S1 采样 + PPO ===
        data_s1 = collect_rollout(model, env, task_id, t_cfg['n_envs'],
                                   t_cfg['rollout_steps'], device, 's1')
        loss_s1 = ppo_update(model, data_s1, 's1', opt, scaler, cfg, device, use_amp)
        
        # === S2 采样 + PPO ===
        model.sync_encoder_s2()
        data_s2 = collect_rollout(model, env, task_id, t_cfg['n_envs'],
                                   t_cfg['rollout_steps'], device, 's2')
        loss_s2 = ppo_update(model, data_s2, 's2', opt, scaler, cfg, device, use_amp)
        
        # === 评估 ===
        if ep > 0 and ep % cfg['eval']['interval'] == 0:
            s1_sr = evaluate(model, env, levels, cfg['eval']['episodes'], device, 's1')
            s2_sr = evaluate(model, env, levels, cfg['eval']['episodes'], device, 's2')
            elapsed = time.time() - t0
            actual_eps = (ep + 1) * t_cfg['n_envs'] * t_cfg['rollout_steps'] * 2
            eps_per_sec = actual_eps / elapsed if elapsed > 0 else 0
            
            print(f'Ep {ep:5d} | S{stage} | S1={s1_sr:.3f} S2={s2_sr:.3f} | '
                  f'L={loss_s1:.3f}/{loss_s2:.3f} | {eps_per_sec:.0f} eps/s')
            with open(log_path, 'a') as f:
                f.write(f'{ep},{stage},{s1_sr:.4f},{s2_sr:.4f},{loss_s1:.4f},{eps_per_sec:.1f}\n')
        
        if ep > 0 and ep % args.save_every == 0:
            torch.save({'model_state': model.state_dict(), 'episode': ep, 'config': cfg}, ckpt_path)
    
    torch.save({'model_state': model.state_dict(), 'episode': t_cfg['total_episodes'], 'config': cfg}, ckpt_path)
    print(f'\n✅ V2 GPU训练完成! {time.time()-t0:.0f}s | Saved to {ckpt_path}')


if __name__ == '__main__':
    main()
