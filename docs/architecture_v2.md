# DSNA v2: 联合训练 + 课程学习 + 共享权重

> 核心: S1/S2 **联合训练**，**共享 Encoder+Skill 权重** (各自独立GRU状态)，**各自独立 AC Head**，**无 Alpha**，三阶段课程。

---

## 一、v1 → v2 核心变更

| 维度 | v1 | v2 |
|------|-----|-----|
| 训练方式 | Phase1 GW→冻结→Phase2 S1 | **S1+S2 联合训练** |
| Encoder | 共享一套 | **共享权重, 各自独立GRU状态** |
| Skill GRU | 共享 | **共享权重, S2梯度不回传** |
| Alpha | α=σ(-b+H+γ·novelty) | **无 Alpha — 各自独立决策** |
| S2→S1 监督 | 无 | **无** |
| S1 正则 | ReLU(α_raw) | **ReLU(-b+entropy)** 技能稀疏 |
| 训练顺序 | 先GW后S1 | **三阶段课程**: 简单→中等→困难 |
| 采样 | S2驱动环境 | **S1/S2 各自分别采样** |
| Task ID | 无 | **Embedding→Encoder+TaskMLP+ACHead** |

---

## 二、v2 架构总览

```
              obs(7×7×3) + instr + task_id
                      │
              ┌───────┴───────┐
              │  task_id →    │
              │  Embedding(32)│
              │  → e_task     │
              └───────┬───────┘
                      │
      ┌───────────────┴───────────────────────────┐
      │   共享 Encoder 权重, 各自独立 GRU 状态     │
      │   CNN+FiLM(obs,instr) + e_task             │
      │   → S=8 路 Skill GRU (各64维)              │
      │                                            │
      │   S1: h_s1 (S,B,64) — 有梯度               │
      │   S2: h_s2 (S,B,64) — detach, 梯度不回传    │
      └───────┬───────────────────────────────────┘
              │
   ┌──────────┴──────────┐
   │  S1                  │         │  S2
   │                      │         │
   │  TaskMLP:            │         │  GW (1次迭代, 全8路竞争):
   │  Concat(h_flat,e_task)         │  write → broadcast
   │  → skill_logits(B,S) │         │  → h_s2_gw (B,64)
   │  → task_label        │         │
   │  (GumbelSigmoid)     │         │  AC Head S2:
   │                      │         │  Concat(h_s2_gw, e_task)
   │  S1 MHA (活跃技能):  │         │  → a2(7), v2(1)
   │  h_s1[active]→attn   │         │
   │  → h_s1_out(B,64)    │         │
   │                      │         │
   │  AC Head S1:         │         │
   │  Concat(h_s1_out,e_task)       │
   │  → a1(7), v1(1)      │         │
   └──────────────────────┘         └──────────────────────┘
```

---

## 三、训练流程

```
每个 epoch:
  ┌─────────────────────────────────────────────┐
  │ 1. S1 采样:                                  │
  │    env.reset() → S1 GRU状态重置              │
  │    for t in rollout:                         │
  │      h_s1 = encoder(obs)  [有梯度]            │
  │      a1 = S1_forward(h_s1)                   │
  │      env.step(a1) → 收集 (obs,a1,r,done)     │
  │    S1 PPO更新 (on-policy)                    │
  ├─────────────────────────────────────────────┤
  │ 2. S2 采样:                                  │
  │    env.reset() → S2 GRU状态重置              │
  │    sync S2 GRU权重 ← S1 GRU权重 (detach)     │
  │    for t in rollout:                         │
  │      h_s2 = encoder(obs)  [detach,无梯度]     │
  │      a2 = S2_forward(h_s2)                   │
  │      env.step(a2) → 收集 (obs,a2,r,done)     │
  │    S2 PPO更新 (GW+AC Head only)              │
  └─────────────────────────────────────────────┘
```

---

## 四、损失函数

```python
# === S1 采样 (on-policy) ===
for t in rollout_s1:
    h_s1 = encoder(obs, instr, e_task)               # (S,B,64) 有梯度
    h_flat = h_s1.permute(1,0,2).reshape(B, -1)
    skill_logits = s1_task_mlp(torch.cat([h_flat, e_task], -1))
    task_label = gumbel_sigmoid(skill_logits)
    h_s1_out = s1_mha(h_s1, task_label)
    a1, v1 = ac_head_s1(torch.cat([h_s1_out, e_task], -1))
    # env.step(a1) → reward

L_s1 = PPO(a1, v1, reward_s1)
entropy  = -Σ p_i·log(p_i) - (1-p_i)·log(1-p_i)
L_sparse = ReLU(-b_sparse + entropy.mean())

# === S2 采样 (on-policy, 权重同步) ===
sync_weights(s2_encoder, s1_encoder)   # detach copy
for t in rollout_s2:
    h_s2 = s2_encoder(obs, instr, e_task)            # (S,B,64) 无梯度到S1
    h_s2_gw, _ = s2_gw(h_s2)                          # 全8路竞争写入
    a2, v2 = ac_head_s2(torch.cat([h_s2_gw, e_task], -1))
    # env.step(a2) → reward

L_s2 = PPO(a2, v2, reward_s2)

# === 总损失 ===
L_total = L_s1 + L_s2 + λ_sparse * L_sparse
```

---

## 五、三阶段课程

| 阶段 | Episodes | 关卡 | S2 预期 | S1 预期 |
|------|----------|------|---------|---------|
| **Simple** | 0 ~ 10K | GoToObj, GoToRedBall, GoToLocal | 快速收敛 | 技能缓慢特化 |
| **Medium** | 10K ~ 30K | +PickupLoc, PutNextLocal, GoToObjMaze | 快速迁移 | 独立追赶, 不遗忘旧技能 |
| **Hard** | 30K ~ end | +GoTo (完整Baby Language) | 处理复杂推理 | 稳定, L_sparse防过拟合 |

```
评估: 每 1K episodes 在所有关卡上分别测试 S1 和 S2 的 success rate
```

---

## 六、训练算法

```python
encoder = SkillEncoder(n_skills=8)             # 共享权重
encoder_s2 = copy_encoder(encoder)              # 独立GRU状态
s1_task_mlp = TaskMLP(input_dim=544)            # 512 + 32
s1_mha = System1MHA()
s2_gw = GlobalWorkspace(n_iters=1)              # 单次迭代,全8路竞争
ac_s1 = ActorCriticHead(input_dim=96)
ac_s2 = ActorCriticHead(input_dim=96)
task_emb = nn.Embedding(n_levels, 32)

b_sparse = 0.5   # 固定阈值 (max entropy ≈ 5.5 for S=8)
λ_sparse = 0.01

opt = Adam([encoder, s1_task_mlp, s1_mha, s2_gw, ac_s1, ac_s2, task_emb], lr=1e-4)

for ep in range(total):
    task_id = sample_curriculum(ep)
    
    # === S1 采样 ===
    rollout_s1 = []
    obs, instr = env.reset(task_id)
    encoder.reset_states()
    for _ in range(rollout_steps):
        e_task = task_emb(tensor([task_id]))
        h_s1 = encoder(obs, instr, e_task)
        a1, v1, info = s1_forward(h_s1, e_task)
        next_obs, r, done = env.step(a1)
        rollout_s1.append((obs, a1, r, v1, done, info))
        obs = next_obs; if done: break
    
    # === S2 采样 ===
    sync_weights(encoder_s2, encoder)  # detach
    rollout_s2 = []
    obs, instr = env.reset(task_id)
    encoder_s2.reset_states()
    for _ in range(rollout_steps):
        e_task = task_emb(tensor([task_id]))
        h_s2 = encoder_s2(obs, instr, e_task)
        a2, v2 = s2_forward(h_s2, e_task)
        next_obs, r, done = env.step(a2)
        rollout_s2.append((obs, a2, r, v2, done))
        obs = next_obs; if done: break
    
    # === PPO 更新 ===
    L = ppo_update(rollout_s1, ac_s1) + ppo_update(rollout_s2, ac_s2) \
        + λ_sparse * sparse_loss(rollout_s1)
    opt.zero_grad(); L.backward(); opt.step()
    
    # === 评估 ===
    if ep % 1000 == 0:
        for level in all_levels:
            s1_sr = evaluate(encoder, s1_forward, level)
            s2_sr = evaluate(encoder_s2, s2_forward, level)
            log(ep, level, s1_sr, s2_sr)

---

## 七、关键设计决策

| # | 决策 | 详情 |
|---|------|------|
| 1 | 共享权重, 独立状态 | Encoder+Skill GRU权重共享; S1/S2各维护自己的GRU hidden state |
| 2 | S2梯度截断 | `encoder_s2.weight = encoder.weight.detach()` — S2不更新encoder |
| 3 | 分别采样 | S1和S2各自独立与环境交互, 各自on-policy PPO |
| 4 | GW全竞争 | S2的GW对所有8路Skill GRU状态做竞争写入, 不依赖task_label |
| 5 | L_sparse | `ReLU(-0.5 + entropy)` — entropy < 0.5时无惩罚, 鼓励技能集中 |
| 6 | 无Alpha | 两系统完全独立决策, 无融合 |
| 7 | 三阶段课程 | 0~10K简单, 10K~30K中等, 30K+困难 |

---

## 八、预期现象

```
         S2 (GW)              S1 (Skill)
Stage1:  ████████████ 快收敛   ████░░░░░░ 慢特化
Stage2:  ████████████ 快迁移   ████████░░ 追赶中
Stage3:  ██████████░ 复杂ok    ██████████ 稳定
回顾S1:  ██████████░ 轻微忘    ██████████ 不遗忘 ← L_sparse防过拟合
```
