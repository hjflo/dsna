# DSNA: Dual-System Neural Architecture

> Combining Kahneman's Fast/Slow thinking with Modular Skills and a Shared Global Workspace

> **v2 架构已发布**: 联合训练 + 三阶段课程 + 双编码器 → [docs/architecture_v2.md](docs/architecture_v2.md)

## Architecture (v1 — 当前实现)

```
obs(7×7×3) + instr(text)
        │
        ▼
┌──────────────────────────────────┐
│  Encoder (冻结于Phase2)           │
│  CNN+FiLM → 共享指令GRU           │
│  → S=8 路技能GRU (各64维)        │
│  → skill_h (8, B, 64)            │
└──────────────┬───────────────────┘
               │
   ┌───────────┴───────────┐
   ▼                       ▼
┌──────────────┐    ┌──────────────┐
│  TaskMLP      │    │  Alpha 控制器 │
│  Concat→MLP   │    │  α = σ(-b +  │
│  →skill_logits│    │   H + γ·nov) │
│  →novelty_log │    └──────┬───────┘
└──────┬───────┘           │
       │                   │
       ▼           task_label (Gumbel)
┌──────────────┐           │
│  System 1     │◄─────────┘
│  活跃技能MHA  │    ┌──────────────┐
│  → h_s1(B,64) │    │  System 2     │
│  (可训练)      │    │  GW N_iter=2  │
└───────┬───────┘    │  +加权聚合     │
        │            │  → h_s2(B,64)  │
        │            │  (冻结)        │
        └─────┬──────┴───────┘
              │  α 融合
              ▼
        h_t = (1-α)·h_s1 + α·h_s2
              │
              ▼
     ┌────────────────┐
     │  AC Head (冻结) │
     │  Actor→7动作    │
     │  Critic→价值    │
     └────────────────┘
```

| 组件 | 状态 | 说明 |
|------|------|------|
| Encoder | Phase1→冻结 | CNN+FiLM+指令GRU + **8路独立技能GRU** (各64维) |
| System 2 (GW) | Phase1→冻结 | N_iter=2 写入, 4槽位, Top-k竞争, 加权聚合 |
| AC Head | Phase1→冻结 | Actor(h_t)→7 + Critic(h_t)→1 |
| TaskMLP | **Phase2可训练** | Concat 8×64 → slow_base → fast → (skill_logits, novelty_logit) |
| System 1 | **Phase2可训练** | 活跃技能 MHA (4头), key_padding_mask 排除非活跃 |
| Alpha (b,γ) | **Phase2可训练** | `α = σ(-b + entropy + γ·novelty)` |

**单步数据流**: `encoder → task_mlp → α, task_label → s1(活跃MHA) + s2(GW) → α融合 → ac_head`

## Training

### Phase 1: GW Pre-training (current)

```bash
python scripts/train_phase1.py --episodes 50000 --save_every 5000
# 后台: nohup python scripts/train_phase1.py ... > outputs/phase1/train.log 2>&1 &
```

| 项目 | 值 |
|------|-----|
| 训练组件 | Encoder + GW + AC Head |
| 损失 | `L_ppo` (无 Alpha 惩罚) |
| 方法 | 累积 rollout(20步) → 单次 backward |
| 环境 | 4并行 × 8关卡均匀采样 |
| 关键超参 | lr=1e-4, γ=0.99, clip=0.2, ent_coef=0.01 |

### Phase 2: Reptile Meta-Learning

```bash
python scripts/train_phase2.py --phase1_ckpt outputs/phase1/checkpoint.pt
```

| 项目 | 值 |
|------|-----|
| 可训练 | TaskMLP + S1注意力 + b,γ (~40K参数) |
| 冻结 | Encoder, GW, AC Head (~570K参数) |
| 损失 | `L_ppo + 0.1·ReLU(α_raw)` |
| 内循环 | K=5步 Adam(lr=0.01), 每任务重置 |
| 外循环 | Reptile ε=0.1 更新 slow_base |
| Fast Head | 每 episode 重置为零 |
| 目标 | 最小化适配后α + 最大化成功率 |

## Quick Start

```bash
git clone git@github.com:hjflo/dsna.git && cd dsna_project
pip install -r requirements.txt
python -m pytest tests/ -v        # 14 tests
python scripts/train_phase1.py    # start training
```

## Monitoring

```bash
tail -f outputs/phase1/train.log   # 实时日志
tail outputs/phase1/log.csv        # CSV 指标
ps -p $(pgrep -f train_phase1) -o pid,etime,%cpu,%mem  # 进程状态
```

## Docs

| 文档 | 内容 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 完整架构 + 数据流公式 + 训练流程 |
| [docs/environment.md](docs/environment.md) | BabyAI obs/action/reward + 8关卡详情 |
| [docs/dsna_project_plan.md](docs/dsna_project_plan.md) | 完整项目设计计划 |

## Project Structure

```
dsna_project/
├── config/default.yaml       # 超参数
├── src/
│   ├── env/                  # BabyAI 多任务环境
│   ├── models/               # 编码器, GW, S1, TaskMLP, Alpha, AC Head
│   ├── training/             # PPO, Reptile
│   └── utils/
├── scripts/                  # 训练/评估脚本
├── tests/                    # 单元测试 (14个)
├── docs/                     # 文档
│   ├── architecture.md       # 架构设计
│   ├── environment.md        # 环境说明
│   ├── design_plan.md        # 设计计划
│   ├── dsna_project_plan.md  # 详细计划
│   └── *.pdf                 # 3篇参考论文
├── outputs/                  # Checkpoints + 日志
├── dataset/                  # 预留
├── pyproject.toml
└── README.md
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| S (n_skills) | 8 | 技能模块数量 |
| skill_gru_dim | 64 | 每路技能 GRU 维度 |
| gw_n_slots | 4 | GW 工作空间槽位数 |
| gw_n_write_iters | 2 | GW 写入迭代次数 |
| gw_top_k | 2 | 每槽位写入竞争 top-k |
| K_inner | 5 | Reptile 内循环步数 |
| ε_meta | 0.1 | Reptile 元步长 |
| λ_alpha | 0.1 | GW 使用惩罚权重 |
| lr (Phase1) | 1e-4 | Adam 学习率 |
| γ | 0.99 | PPO 折扣因子 |
| clip_eps | 0.2 | PPO clip 范围 |

## Current Status

```
训练: Phase 1 (GW pretraining)
状态: 运行中 (CPU)
命令: python scripts/train_phase1.py --episodes 50000
日志: outputs/phase1/train.log
模型: outputs/phase1/checkpoint.pt
```

## References

- **BabyAI**: Chevalier-Boisvert et al., "BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning", ICLR 2019
- **Shared Global Workspace**: Goyal et al., "Coordination Among Neural Modules Through a Shared Global Workspace", ICLR 2022
- **Polytropon**: Ponti et al., "Combining Modular Skills in Multitask Learning", 2022

## License

MIT
