#!/usr/bin/env python3
"""
Phase 1: GW + Encoder + AC Head 预训练
在所有 BabyAI 关卡上 PPO 训练，完成后保存冻结权重。
"""
import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import gym
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import DSNAModel
from src.env.multitask_wrapper import MultiTaskBabyAIEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_dir', default='outputs/phase1')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


class PPOAlgo:
    """简化版 PPO 实现 (基于 BabyAI 原论文参数)"""
    def __init__(self, model, config, device):
        self.model = model
        self.device = device
        p = config['ppo']
        t = config['training']['phase1']

        self.clip_eps = p['clip_eps']
        self.gamma = p['gamma']
        self.gae_lambda = p['gae_lambda']
        self.entropy_coef = p['entropy_coef']
        self.value_loss_coef = p['value_loss_coef']
        self.max_grad_norm = p['max_grad_norm']
        self.ppo_epochs = p['ppo_epochs']
        self.batch_size = p.get('batch_size', 256)

        self.opt = torch.optim.Adam(model.parameters(), lr=t['lr'])

    def compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1-dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.tensor(advantages), torch.tensor(returns)

    def ppo_update(self, batch_obs, batch_instr, batch_actions, batch_old_logp,
                   batch_advantages, batch_returns, batch_values):
        """PPO policy + value update"""
        # Normalize advantages
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

        total_samples = len(batch_obs)
        indices = np.random.permutation(total_samples)

        for _ in range(self.ppo_epochs):
            for start in range(0, total_samples, self.batch_size):
                idx = indices[start:start+self.batch_size]
                obs_b = batch_obs[idx].to(self.device)
                instr_b = batch_instr[idx].to(self.device)
                act_b = batch_actions[idx].to(self.device)
                old_lp_b = batch_old_logp[idx].to(self.device)
                adv_b = batch_advantages[idx].to(self.device)
                ret_b = batch_returns[idx].to(self.device)
                val_b = batch_values[idx].to(self.device)

                action_logits, value = self.model(obs_b, instr_b)
                log_probs = nn.functional.log_softmax(action_logits, dim=-1)
                new_logp = log_probs.gather(1, act_b.unsqueeze(-1)).squeeze(-1)

                ratio = torch.exp(new_logp - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(value, ret_b)
                entropy = -(log_probs.exp() * log_probs).sum(-1).mean()

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'entropy': entropy.item()}


def train_phase1(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建多任务环境
    env = MultiTaskBabyAIEnv(config['env']['levels'])
    model = DSNAModel(config, mode='gw_only').to(device)
    ppo = PPOAlgo(model, config, device)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'phase1_checkpoint.pt')

    t_cfg = config['training']['phase1']
    n_envs = t_cfg['n_envs']
    rollout_steps = t_cfg['rollout_steps']
    total_episodes = t_cfg['total_episodes']

    episode = 0
    best_success = 0

    while episode < total_episodes:
        # Collect rollouts
        all_obs, all_instr, all_actions, all_logps = [], [], [], []
        all_rewards, all_values, all_dones = [], [], []

        obs_list, instr_list = env.reset(n_envs)
        model.reset_episode(n_envs)

        for _ in range(rollout_steps):
            obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)
            instr_t = torch.tensor(np.stack(instr_list), dtype=torch.long).to(device)

            # Encode instruction (simplified: use random embedding for now)
            # In production, use BabyAI's vocabulary and GRU
            with torch.no_grad():
                instr_emb = model.encoder.encode_instruction(instr_t)

            action_logits, value = model(obs_t, instr_emb)
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step environment
            next_obs, rewards, dones, _ = zip(*[env.step(a.item()) for a in action])

            all_obs.append(obs_t.cpu())
            all_instr.append(instr_t.cpu())
            all_actions.append(action.cpu())
            all_logps.append(log_prob.cpu())
            all_rewards.append(torch.tensor(rewards))
            all_values.append(value.detach().cpu())
            all_dones.append(torch.tensor(dones, dtype=torch.float32))

            obs_list = next_obs
            episode += n_envs

            if episode >= total_episodes:
                break

        # Stack and flatten
        cat = lambda x: torch.cat(x, dim=0)
        batch_obs = cat(all_obs)
        batch_instr = cat(all_instr)
        batch_actions = cat(all_actions)
        batch_logps = cat(all_logps)
        batch_rewards = cat(all_rewards)
        batch_values = cat(all_values)
        batch_dones = cat(all_dones)

        advantages, returns = ppo.compute_gae(batch_rewards, batch_values, batch_dones)

        metrics = ppo.ppo_update(batch_obs, batch_instr, batch_actions, batch_logps,
                                 advantages, returns, batch_values)

        # Save checkpoint
        if episode % 5000 < n_envs:
            torch.save({
                'model_state': model.state_dict(),
                'episode': episode,
                'config': config,
            }, save_path)
            print(f"[Phase 1] Episode {episode:6d} | policy_loss={metrics['policy_loss']:.4f} "
                  f"| value_loss={metrics['value_loss']:.4f} | entropy={metrics['entropy']:.4f}")

    # Final save
    torch.save({
        'model_state': model.state_dict(),
        'episode': episode,
        'config': config,
    }, save_path)
    print(f"[Phase 1] Done! Model saved to {save_path}")


if __name__ == '__main__':
    args = parse_args()
    train_phase1(args)
