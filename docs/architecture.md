# DSNA 架构设计与训练流程

> 最后更新: 2026-07-19 | 训练状态: Phase 1 运行中 (PID 323548)

---

## 一、整体架构

```
                        ┌──────────────────────────────────┐
                        │        BabyAI 环境 (8关卡)        │
                        │  GoToObj, GoToLocal, PickupLoc,   │
                        │  PutNextLocal, GoToRedBall*, GoTo  │
                        └──────────────┬───────────────────┘
                                       │ obs(7×7×3) + instr
                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Encoder (冻结于Phase2)                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────────────┐ │
│  │ CNN+FiLM │   │ 共享GRU  │   │  S=8 路技能GRU (各64维)      │ │
│  │ 视觉编码  │   │ 指令编码  │   │  GRU₀ GRU₁ ... GRU₇         │ │
│  └────┬─────┘   └────┬─────┘   │  各维护独立隐藏状态            │ │
│       └──────┬───────┘         └──────────┬───────────────────┘ │
│              └─ concat ──────────────────→ skill_h (S,B,64)     │
└────────────────────────────────────────────┬─────────────────────┘
                                             │
              ┌──────────────────────────────┴──────────────────┐
              │                                                 │
              ▼                                                 │
┌──────────────────────────┐                    ┌──────────────────────────┐
│   TaskMLP (可训练)        │                    │                          │
│                          │                    │   Alpha 控制器 (可训练)   │
│ slow_base → fast → ──────┼── skill_logits ──→│                          │
│                │         │                    │ α = σ(-b + H + γ·novelty) │
│                └─────────┼── novelty_logit ──→│                          │
└──────────────────────────┘                    └────────────┬─────────────┘
                                              task_label     │
                                              (GumbelSigmoid)│
                                                    │        │
              ┌─────────────────────────────────────┘        │
              ▼                                              ▼
┌──────────────────────────┐                    ┌──────────────────────────┐
│  System 1 (可训练)        │                    │  System 2 (冻结)          │
│  活跃技能 MHA 成对交互    │                    │  GW N_iter=2 + 加权聚合   │
│  → h_s1 (B,64)           │                    │  → h_s2 (B,64)            │
└──────────┬───────────────┘                    └──────────┬───────────────┘
           │                                               │
           └───────────┬──────────────┬────────────────────┘
                       │   α 融合     │
                       ▼               ▼
                  h_t = (1-α)·h_s1 + α·h_s2
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   AC Head (冻结)          │
                 │   Actor → 7动作          │
                 │   Critic → 状态价值       │
                 └──────────────────────────┘
```

---

## 二、组件清单

| 组件 | 文件 | 状态 | 参数量(约) |
|------|------|------|-----------|
| **Encoder** (CNN+FiLM+指令GRU+8路技能GRU) | `base_encoder.py` | Phase1训练→冻结 | ~500K |
| **GW** (N_iter=2, 4槽位, Top-k竞争, 加权聚合) | `system2_workspace.py` | Phase1训练→冻结 | ~50K |
| **AC Head** (Actor→7, Critic→1) | `ac_head.py` | Phase1训练→冻结 | ~20K |
| **TaskMLP** (Concat 8×64→slow_base→fast→双输出) | `task_mlp.py` | **Phase2可训练** | ~30K |
| **System 1** (MHA 4头) | `system1_pairwise.py` | **Phase2可训练** | ~8K |
| **Alpha** (b, γ) | `alpha_controller.py` | **Phase2可训练** | 2个标量 |

---

## 三、数据流（单步 forward）

```python
# ===== Step 1: 编码 =====
skill_h = encoder(obs, instr_tokens, skill_states_{t-1})  # (S=8, B, 64)
# CNN+FiLM处理视觉, 共享GRU编码指令, 8路独立技能GRU各自更新

# ===== Step 2: TaskMLP 预测 =====
skill_logits, novelty_logit = task_mlp(skill_h)
# skill_logits:  (B, S)  — 每技能激活倾向
# novelty_logit: (B, 1)  — 情境新颖度倾向

# ===== Step 3: Alpha 计算 =====
probs  = σ(skill_logits)
entropy = -Σ p_i·log(p_i) - (1-p_i)·log(1-p_i)   # 技能不确定度
novelty = σ(novelty_logit)                         # 情境新颖度
α_raw   = -b + entropy + γ·novelty                 # b,γ 可学习
α       = σ(α_raw)                                 # GW 参与度 ∈ (0,1)

# ===== Step 4: 双系统并行 =====
task_label = GumbelSigmoid(skill_logits)           # 训练时
h_s1 = system1(skill_h, task_label)                # 活跃技能MHA (可训练)
h_s2, _ = gw(skill_h)                              # GW全局协调 (冻结)

# ===== Step 5: 融合 + 决策 =====
h_t    = (1-α)·h_s1 + α·h_s2
action, value = ac_head(h_t)                       # 冻结AC头
```

---

## 四、训练流程

### Phase 1: GW 预训练 (当前运行中)

```
┌─────────────────────────────────────────────────────┐
│ 训练组件: Encoder + GW + AC Head                     │
│ 损失:     L = L_ppo (无α惩罚)                        │
│ 方法:     累积rollout → 单次backward                 │
│ 环境:     4并行 × 8关卡均匀采样                      │
│                                                    │
│ 超参数:                                             │
│   lr = 1e-4 (Adam)                                  │
│   γ = 0.99 (折扣)                                   │
│   λ = 0.99 (GAE)                                    │
│   clip_eps = 0.2                                    │
│   entropy_coef = 0.01                               │
│   value_loss_coef = 0.5                             │
│   max_grad_norm = 0.5                               │
│                                                    │
│ 目标: 50,000 episodes                               │
│ 输出: outputs/phase1/checkpoint.pt                  │
└─────────────────────────────────────────────────────┘
```

### Phase 2: Reptile 元学习 (待 Phase 1 完成后)

```
┌─────────────────────────────────────────────────────┐
│ 训练组件: TaskMLP(slow+fast) + S1注意力 + b,γ        │
│ 冻结:     Encoder, GW, AC Head                       │
│ 损失:     L = L_ppo + 0.1·ReLU(α_raw)                │
│                                                    │
│ Reptile 外循环:                                      │
│   for 每个任务 (BabyAI关卡):                         │
│     1. 保存 slow_base 副本                           │
│     2. 重置 fast_head 为零 (高熵 → 初始依赖GW)       │
│     3. 内循环 K=5 步:                                │
│        - Adam (lr=0.01, 每任务重置)                   │
│        - 损失 = L_ppo + λ·ReLU(α_raw)                │
│        - 仅更新 fast_head                             │
│     4. 外循环: slow_base += 0.1×(adapted - original) │
│                                                    │
│ 目标: 最小化适配后的 α + 最大化成功率                  │
│ 输出: outputs/phase2/checkpoint.pt                  │
└─────────────────────────────────────────────────────┘
```

---

## 五、关键设计决策

| # | 决策 | 详情 |
|---|------|------|
| 1 | 编码器 | 无状态设计，调用者管理GRU状态；每步 detach 打破RNN计算图 |
| 2 | 训练方式 | 累积 rollout loss → 单次 backward（避免"backward twice"错误） |
| 3 | GW | N_iter=2 写入 + S路独立输出 + 可学习加权聚合 `Σ w_i·h_i'` |
| 4 | S1 | 活跃技能 MHA，key_padding_mask 排除非活跃；仅活跃技能参与聚合 |
| 5 | α 公式 | `σ(-b + entropy + γ·novelty)`，entropy/novelty 均来自 TaskMLP |
| 6 | Novelty | TaskMLP 自身预测 novelty_logit → σ → novelty（无历史缓存） |
| 7 | 防记忆化 | Fast/Slow 双速 TaskMLP + 每 episode 重置 Fast Head + Reptile 元学习 |
| 8 | 技能模块 | 每路独立 GRU 即技能模块（无需额外 LoRA/adapter） |
| 9 | 环境 | gymnasium + minigrid (Farama Foundation 维护版) |
| 10 | AC 头 | 仅需 h_t 输入 (obs+instr 已编码其中) |
| 11 | GPU | 当前 CPU 训练 (CUDA driver 版本过旧) |

---

## 六、当前训练状态

```
进程:    PID 323548
命令:    python scripts/train_phase1.py --episodes 50000 --save_every 5000
设备:    CPU
进度:    运行中

日志:    outputs/phase1/train.log
CSV:     outputs/phase1/log.csv
模型:    outputs/phase1/checkpoint.pt (每 5000 ep 保存)

初始:    Ep 0: loss=0.226, reward=0.000 (随机策略)
```

**监控命令**:
```bash
tail -f outputs/phase1/train.log    # 实时日志
tail outputs/phase1/log.csv         # CSV进度
ps -p 323548 -o pid,etime,%cpu,%mem # 进程状态
```

---

## 七、项目结构

```
dsna_project/
├── config/default.yaml       # 完整超参数配置
├── src/
│   ├── env/multitask_wrapper.py   # BabyAI 8关卡多任务环境
│   ├── models/
│   │   ├── base_encoder.py        # 无状态 S路技能GRU编码器
│   │   ├── task_mlp.py            # Fast/Slow 双速 TaskMLP
│   │   ├── alpha_controller.py    # α = σ(-b + H + γ·novelty)
│   │   ├── system1_pairwise.py    # 活跃技能 MHA 成对交互
│   │   ├── system2_workspace.py   # GW (N_iter=2, 加权聚合)
│   │   ├── ac_head.py             # Actor+Critic 双MLP
│   │   └── dsna_model.py          # 主模型 (Phase1/2双模式)
│   ├── training/                  # PPO训练器
│   └── utils/                     # 工具函数
├── scripts/
│   ├── train_phase1.py            # GW预训练脚本
│   └── train_phase2.py            # Reptile元学习脚本
├── tests/test_modules.py          # 14个单元测试
├── docs/                          # 论文+设计文档
├── outputs/phase1/                # 训练checkpoint+日志
├── dataset/                       # 数据集存储
├── pyproject.toml
└── README.md
```

---

## 八、参考文献

1. **BabyAI**: Chevalier-Boisvert et al., "BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning", ICLR 2019
2. **Shared Global Workspace**: Goyal et al., "Coordination Among Neural Modules Through a Shared Global Workspace", ICLR 2022
3. **Polytropon**: Ponti et al., "Combining Modular Skills in Multitask Learning", 2022
