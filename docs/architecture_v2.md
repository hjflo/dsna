# DSNA v2: 联合训练 + 课程学习 + 共享编码器

> 核心: S1/S2 **联合训练**，**共享 Encoder+Skill** (S2梯度不回传)，**各自独立 AC Head**，Alpha=σ(-b+γ·KL(h_s1,h_s2)) 仅监控不融合，三阶段课程。

---

## 一、v1 → v2 核心变更

| 维度 | v1 | v2 |
|------|-----|-----|
| 训练方式 | Phase1 GW→冻结→Phase2 S1 | **S1+S2 联合训练** |
| Encoder+Skill | S1/S2 各自独立 | **共享一套** Encoder + Skill GRU |
| AC Head | S1/S2 各自独立 | **S1/S2 各自独立** AC Head |
| S2 梯度 | 无限制 | **S2 梯度不回传** Encoder/Skill (detach) |
| GW 迭代 | N_iter=2 | **每时间步 1 次** |
| Alpha | α=σ(-b+H+γ·novelty) | **α=σ(-b+γ·KL(h_s1,h_s2))** 仅监控 |
| Novelty | TaskMLP 预测 | **KL(h_s1_out, h_s2_gw)** |
| S2→S1 监督 | 无 | **无** (各自独立训练) |
| 训练顺序 | 先GW后S1 | **三阶段课程**: 简单→中等→困难 |
| S1 正则 | ReLU(α_raw) | **ReLU(-b+entropy)** 技能稀疏 |
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
      │   共享 Encoder (S1 可训练, S2 只读)        │
      │   CNN+FiLM(obs,instr) + e_task             │
      │   → S=8 路 Skill GRU (各64维)              │
      │   → h_skills (S, B, 64)                    │
      └───────┬───────────────────────────────────┘
              │
              ├── S1: h_skills (有梯度, 训练 Encoder+Skill)
              │
              └── S2: h_skills.detach() ← 梯度截断, 不回传
                      │
   ┌──────────────────┴──────────────────┐
   │  S1 (可训练)                         │  S2 (可训练: GW only)
   │                                     │
   │  TaskMLP:                           │  GW (1次迭代):
   │  Concat(h_flat, e_task) → MLP       │  write(竞争) → broadcast
   │  → skill_logits (B,S)               │  → h_s2_gw (B,64)
   │  → task_label (GumbelSigmoid)       │
   │                                     │  AC Head S2:
   │  S1 MHA (活跃技能):                 │  Concat(h_s2_gw, e_task)
   │  h_skills[active] → attention       │  → action2(7), value2(1)
   │  → h_s1_out (B,64)                  │
   │                                     │
   │  AC Head S1:                        │
   │  Concat(h_s1_out, e_task)           │
   │  → action1(7), value1(1)            │
   └──────────────────┬──────────────────┘
                      │
              ┌───────┴───────┐
              │  Alpha (监控)  │
              │  KL(h_s1||h_s2)│
              │  α=σ(-b+γ·KL) │
              │  (不参与决策)  │
              └───────────────┘
```

---

## 三、损失函数

```python
e_task = task_emb(task_id)                        # (B, 32)

# === Encoder (共享) ===
h_skills = encoder(obs, instr, e_task)            # (S, B, 64)

# === S1 forward (有梯度) ===
h_flat = h_skills.permute(1,0,2).reshape(B, -1)   # (B, 512)
skill_logits = s1_task_mlp(torch.cat([h_flat, e_task], -1))
task_label = gumbel_sigmoid(skill_logits)
h_s1_out = s1_mha(h_skills, task_label)             # (B, 64)
a1, v1 = ac_head_s1(torch.cat([h_s1_out, e_task], -1))

# === S2 forward (detach, 梯度不回传 Encoder) ===
h_s2_gw, _ = s2_gw(h_skills.detach())              # (B, 64)
a2, v2 = ac_head_s2(torch.cat([h_s2_gw, e_task], -1))

# === Alpha: KL 散度 (仅监控, 不参与损失) ===
kl = KL_divergence(h_s1_out, h_s2_gw.detach())
α  = σ(-b_alpha + γ * kl)                            # 仅记录

# === 环境交互 (用 S2 action — S2 学得快) ===
next_obs, reward, done = env.step(a2.argmax())

# === 损失 (各自独立) ===
L_s1 = PPO(a1, v1, reward)
L_s2 = PPO(a2, v2, reward)
entropy  = -Σ p_i·log(p_i) - (1-p_i)·log(1-p_i)
L_sparse = ReLU(-b_sparse + entropy)

L_total = L_s1 + L_s2 + λ_sparse * L_sparse
# 无 L_aux, 无 α 惩罚 — 两个系统独立学习
```

---

## 五、三阶段课程

```
Stage 1: Simple  (GoToObj, GoToRedBall, GoToLocal)
  → S2 GW 快速学会, S1 技能缓慢特化
  → KL 高 (S1≠S2) — S2 领先

Stage 2: Medium  (新增 PickupLoc, PutNextLocal, GoToObjMaze)
  → S2 快速迁移, S1 独立追赶
  → KL 逐渐降低 (S1 逼近 S2)

Stage 3: Hard    (新增 GoTo)
  → S2 处理复杂推理, S1 保持稳定
  → 回顾简单关卡: S1 不遗忘 (稀疏技能分配)
```

---

## 六、训练算法

```python
encoder = SkillEncoder(n_skills=8)
s1_task_mlp = TaskMLP(input_dim=544)       # 512 + 32
s1_mha = System1MHA()
s2_gw = GlobalWorkspace(n_iters=1)
ac_head_s1 = ActorCriticHead(input_dim=96)  # 64 + 32
ac_head_s2 = ActorCriticHead(input_dim=96)  # 独立
task_emb = nn.Embedding(n_levels, 32)

b_alpha = nn.Parameter(torch.tensor(1.5))   # Alpha 仅监控
gamma   = nn.Parameter(torch.tensor(1.0))
b_sparse = 0.5

opt = Adam([encoder, s1_task_mlp, s1_mha, s2_gw,
            ac_head_s1, ac_head_s2, task_emb, b_alpha, gamma], lr=1e-4)

for ep in range(total):
    task_id = sample_curriculum(ep)
    obs, instr = env.reset(task_id)
    
    for t in range(ep_len):
        e_task = task_emb(torch.tensor([task_id]*B))
        h_skills = encoder(obs, instr, e_task, states)
        
        # S1
        h_flat = h_skills.permute(1,0,2).reshape(B,-1)
        skill_logits = s1_task_mlp(torch.cat([h_flat, e_task], -1))
        task_label = gumbel_sigmoid(skill_logits)
        h_s1_out = s1_mha(h_skills, task_label)
        a1, v1 = ac_head_s1(torch.cat([h_s1_out, e_task], -1))
        
        # S2 (detach, 梯度隔离)
        h_s2_gw, _ = s2_gw(h_skills.detach())
        a2, v2 = ac_head_s2(torch.cat([h_s2_gw, e_task], -1))
        
        # Alpha (仅监控)
        kl = kl_div(h_s1_out, h_s2_gw.detach())
        α  = torch.sigmoid(-b_alpha + gamma * kl)
        
        # 环境交互 (S2 action)
        next_obs, reward, done = env.step(a2.argmax())
        
        # 损失 (各自独立)
        L = ppo_loss(a1, v1, reward) + ppo_loss(a2, v2, reward) \
            + λ_sparse * F.relu(-b_sparse + entropy(skill_logits))
        
        opt.zero_grad(); L.backward(); opt.step()
        states = h_skills

---

## 七、预期现象

```
         S2 (GW)              S1 (Skill)           α (KL-based)
Stage1:  ████████████ 快       ████░░░░░░ 慢         ████████ 高 (S1≠S2)
Stage2:  ████████████ 迁移快   ████████░░ 追赶       ████░░░░ 中
Stage3:  ██████████░ 复杂ok    ██████████ 稳定       ██░░░░░░ 低 (S1≈S2)
回顾S1:  ██████████░ 轻微忘    ██████████ 不遗忘     █░░░░░░░ 很低
```
