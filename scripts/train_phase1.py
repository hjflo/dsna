#!/usr/bin/env python3
"""Phase 1: GW + Encoder + AC Head 预训练 (CPU)"""
import sys, os, time, yaml, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.env.multitask_wrapper import MultiTaskBabyAIEnv
from src.models import DSNAModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/default.yaml')
    p.add_argument('--episodes', type=int, default=200000)
    p.add_argument('--save_every', type=int, default=5000)
    p.add_argument('--n_envs', type=int, default=4)
    p.add_argument('--rollout_steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cpu')
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env = MultiTaskBabyAIEnv(cfg['env']['levels'], max_steps=cfg['env']['max_steps_per_episode'])
    model = DSNAModel(cfg, mode='gw_only').to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['training']['phase1']['lr'])
    pc = cfg['ppo']
    os.makedirs('outputs/phase1', exist_ok=True)
    ckpt_path, log_path = 'outputs/phase1/checkpoint.pt', 'outputs/phase1/log.csv'
    with open(log_path, 'w') as f: f.write('episode,loss,reward_mean,reward_max\n')
    total_ep, losses, t0 = 0, [], time.time()
    for ep in range(args.episodes):
        obs_list, instr_list = env.reset(args.n_envs)
        model.reset_episode(args.n_envs)
        total_loss, ep_rewards, n_steps = 0, [[] for _ in range(args.n_envs)], 0
        for _ in range(args.rollout_steps):
            ot = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(args.device)
            it = torch.tensor(np.stack(instr_list), dtype=torch.long).to(args.device)
            al, v = model(ot, it)
            d = torch.distributions.Categorical(logits=al)
            a = d.sample()
            no, rw, dn, _ = env.step(a.tolist())
            rt = torch.tensor(rw, dtype=torch.float32).to(args.device)
            adv = rt - v.detach()
            pl = -(d.log_prob(a) * adv).mean()
            vl = torch.nn.functional.mse_loss(v, rt)
            loss = pl + pc['value_loss_coef'] * vl - pc['entropy_coef'] * d.entropy().mean()
            total_loss += loss; n_steps += 1
            for i,(r,d) in enumerate(zip(rw,dn)): ep_rewards[i].append(r)
            obs_list = no
            if any(dn):
                obs_list, instr_list = env.reset(args.n_envs)
                model.reset_episode(args.n_envs)
                total_ep += sum(dn)
        opt.zero_grad(); (total_loss/n_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), pc['max_grad_norm']); opt.step()
        losses.append(total_loss.item()/n_steps)
        ar = [r for rl in ep_rewards for r in rl]
        if ep % 100 == 0:
            al = np.mean(losses[-100:]) if len(losses)>=100 else np.mean(losses)
            print(f'Ep {total_ep:6d} | loss={al:.4f} | r_mean={np.mean(ar):.3f} | r_max={np.max(ar):.3f} | {total_ep/(time.time()-t0):.1f} eps/s')
            with open(log_path,'a') as f: f.write(f'{total_ep},{al},{np.mean(ar):.4f},{np.max(ar):.4f}\n')
        if ep>0 and ep%args.save_every==0:
            torch.save({'model_state':model.state_dict(),'episode':total_ep,'config':cfg},ckpt_path)
    torch.save({'model_state':model.state_dict(),'episode':total_ep,'config':cfg},ckpt_path)
    print(f'\nDone! {total_ep} eps, {time.time()-t0:.0f}s. Saved to {ckpt_path}')

if __name__=='__main__': main()
