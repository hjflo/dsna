# DSNA: Dual-System Neural Architecture

> Combining Kahneman's Fast/Slow thinking with Modular Skills and a Shared Global Workspace

## Overview

DSNA implements a dual-system cognitive architecture for multi-task reinforcement learning on the BabyAI platform:

- **System 1 (Fast)**: Modular skill modules with pairwise attention — automatic, intuitive processing
- **System 2 (Slow)**: Shared Global Workspace (GW) — deliberate, conscious reasoning
- **Alpha Gating**: Dynamic α = σ(-b + entropy + γ·novelty) controls the balance

As experience accumulates, knowledge crystallizes from System 2 into System 1 via Reptile meta-learning.

## Project Structure

```
dsna_project/
├── config/           # YAML configuration files
├── src/              # Source code
│   ├── env/          # Multi-task BabyAI environment wrappers
│   ├── models/       # Encoder, TaskMLP, Alpha, S1, S2, AC Head, DSNA model
│   ├── training/     # PPO trainer, Reptile meta-learner
│   └── utils/        # Metrics, logging, visualization
├── scripts/          # Training/evaluation scripts
├── tests/            # Unit tests
├── dataset/          # Dataset storage (demonstrations, preprocessed data)
├── outputs/          # Model checkpoints, logs, experiment results
├── docs/             # Documentation and design documents
├── pyproject.toml    # Project configuration
├── requirements.txt  # Dependencies
└── README.md
```

## Quick Start

### Installation

```bash
pip install -e .
```

### Phase 1: GW Pre-training

```bash
python scripts/train_phase1.py --config config/default.yaml --seed 1
```

### Phase 2: Reptile Meta-Learning

```bash
python scripts/train_phase2.py \
    --config config/default.yaml \
    --phase1_ckpt outputs/phase1/phase1_checkpoint.pt \
    --seed 1
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| S (n_skills) | 8 | Number of skill modules |
| skill_gru_dim | 64 | Per-skill GRU dimension |
| gw_n_slots | 4 | GW workspace slots |
| gw_n_write_iters | 2 | GW write iterations |
| K_inner | 5 | Reptile inner loop steps |
| ε_meta | 0.1 | Reptile meta step size |
| λ_alpha | 0.1 | GW usage penalty weight |

## References

- BabyAI: Chevalier-Boisvert et al., ICLR 2019
- Shared Global Workspace: Goyal et al., ICLR 2022
- Polytropon (Modular Skills): Ponti et al., 2022

## License

MIT
