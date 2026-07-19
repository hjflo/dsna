# DSNA: Dual-System Neural Architecture

> Combining Kahneman's Fast/Slow thinking with Modular Skills and a Shared Global Workspace

## Overview

DSNA implements a dual-system cognitive architecture for multi-task reinforcement learning on the BabyAI platform:

- **System 1 (Fast)**: Modular skill modules with pairwise attention — automatic, intuitive processing
- **System 2 (Slow)**: Shared Global Workspace (GW) — deliberate, conscious reasoning
- **Alpha Gating**: Dynamic α = σ(-b + entropy + γ·novelty) controls the balance

As experience accumulates, knowledge crystallizes from System 2 into System 1 via Reptile meta-learning.

## Documentation

| 文档 | 内容 |
|------|------|
| [architecture.md](docs/architecture.md) | 完整架构设计与训练流程 |
| [environment.md](docs/environment.md) | BabyAI 环境说明 |
| [dsna_project_plan.md](docs/dsna_project_plan.md) | 详细项目计划 |


## Quick Start

### 1. Install

```bash
git clone git@github.com:hjflo/dsna.git
cd dsna_project
pip install -r requirements.txt
```

### 2. Test

```bash
python -m pytest tests/ -v    # 14 unit tests
```

### 3. Phase 1: GW Pre-training (current)

```bash
# 前台运行
python scripts/train_phase1.py --episodes 50000 --save_every 5000

# 后台运行
nohup python scripts/train_phase1.py --episodes 50000 --save_every 5000 > outputs/phase1/train.log 2>&1 &

# 监控
tail -f outputs/phase1/train.log
tail outputs/phase1/log.csv
```

**训练流程**:
1. 4 个并行 BabyAI 环境, 8 关卡随机采样
2. 累积 rollout (20 步) → 单次 backward
3. PPO 更新: `L = L_ppo` (无 Alpha 惩罚)
4. 每 5000 episodes 保存 checkpoint

### 4. Phase 2: Reptile Meta-Learning

```bash
python scripts/train_phase2.py \
    --config config/default.yaml \
    --phase1_ckpt outputs/phase1/checkpoint.pt
```

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
