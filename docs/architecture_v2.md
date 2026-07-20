# DSNA v2: 联合训练 + 课程学习 + 双编码器架构

> 基于 v1 的重大架构修订。核心变更：System 1 和 System 2 **联合训练**，S2 复用 S1 的技能参数（梯度不回传），各自拥有独立隐藏状态，三阶段课程学习。

---

## 一、v1 → v2 核心变更

| 维度 | v1 | v2 |
|------|-----|-----|
| 训练方式 | Phase1 GW预训练→冻结→Phase2 S1元学习 | **S1 和 S2 联合训练** (端到端) |
| 编码器 | 共享一套 Skill GRU | **S1/S2 各自独立编码器** + 独立隐藏状态 |
| 技能参数 | S1/S2 共享同一套 GRU 权重 | **S2 复用 S1 的 Skill GRU 权重**，梯度不回传 |
| Alpha | 存在，S1/S2 竞争 | 保留（可选），但焦点转移 |
| 训练顺序 | 先 GW 后 S1 | **三阶段课程**: 简单→中等→困难 |
| S1 正则 | ReLU(α_raw) 惩罚 GW 使用 | **ReLU(-b + entropy(label))** 保证技能稀疏 |
| S2→S1 监督 | 无 | **MSE(h_s1, h_s2)** 辅助损失 |
| Task ID | 无 | **task_id → embedding 输入 Encoder 和 TaskMLP** |

---

## 二、v2 架构总览

```
                    ┌─────────────────────────────┐
                    │       BabyAI 环境             │
                    │   8 关卡,三阶段课程            │
                    └──────────────┬──────────────┘
                                   │ obs(7×7×3) + instr + task_id
                    ┌──────────────┴──────────────────────────┐
                    │   task_id → Embedding(task_id) → e_task │
                    └──────────────┬──────────────────────────┘
                                   │ e_task (B, 32)
          ┌────────────────────────┼────────────────────────┐
          │                        │                         │
          ▼                        ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────┐
│  S1 Encoder (可训练)  │  │  S2 Encoder (可训练)  │  │  S1 TaskMLP  │
│  CNN+FiLM(obs,instr) │  │  CNN+FiLM(obs,instr) │  │              │
│  + e_task ──────────→│  │  + e_task ──────────→│  │ Concat(      │
│  S1 Skill GRUs ×8    │  │  S2 Skill GRUs ×8    │  │  h_s1_flat,  │
│  → h_s1 (S,B,64)     │  │  → h_s2_raw (S,B,64)│  │  e_task      │
└──────────┬───────────┘  └──────────┬───────────┘  │ )→MLP        │
           │                         │               │ →skill_logits│
           │  θ_skills ─────────────→│ (detach copy) │ →novelty_log │
           │  S1 更新 S2 只读         │               └──────┬───────┘
           │                         │                      │
           ▼                         ▼                      ▼
   ┌───────────────┐        ┌───────────────┐      task_label
   │ S1 MHA(活跃)  │        │ S2 GW N_iter=2 │      (GumbelSigmoid)
   │ → h_s1_out    │        │ → h_s2_gw      │          │
   └───────┬───────┘        └───────┬───────┘          │
           │                        │                   │
           │   L_aux = MSE          │                   │
           │◄───────────────────────┘                   │
           ▼                                            │
   ┌───────────────┐                                    │
   │ S1 AC Head    │                                    │
   │ + e_task ────→│                                    │
   │ → a1, v1      │                                    │
   └───────────────┘                                    │
                                                        ▼
                                                ┌───────────────┐
                                                │ S2 AC Head    │
                                                │ + e_task ────→│
                                                │ → a2, v2      │
                                                └───────────────┘
```

---

## 三、损失函数

```python
# === 每个 timestep ===
e_task = task_embedding(task_id)                     # (B, 32)

# S1 forward — e_task 注入 Encoder 和 TaskMLP
h_s1 = s1_encoder(obs, instr, e_task)               # (S,B,64)
h_s1_flat = h_s1.permute(1,0,2).reshape(B, S*64)    # (B, 512)
skill_logits, novelty_logit = s1_task_mlp(
    torch.cat([h_s1_flat, e_task], dim=-1))          # (B, 544) → (B,S)
task_label = GumbelSigmoid(skill_logits)
h_s1_out = s1_mha(h_s1, task_label)                  # (B,64)
a1, v1 = s1_ac_head(torch.cat([h_s1_out, e_task], dim=-1))

# S2 forward — e_task 注入 Encoder
sync_weights(s2_encoder.skill_grus, s1_encoder.skill_grus)  # detach
h_s2_raw = s2_encoder(obs, instr, e_task)            # (S,B,64)
h_s2_gw, _ = s2_gw(h_s2_raw)                         # (B,64)
a2, v2 = s2_ac_head(torch.cat([h_s2_gw, e_task], dim=-1))

# === 损失 ===
L_task_s1 = PPO(a1, v1, reward)
L_task_s2 = PPO(a2, v2, reward)
entropy   = -Σ p_i·log(p_i) - (1-p_i)·log(1-p_i)
L_sparse  = ReLU(-b_sparse + entropy)
L_aux     = MSE(h_s1_out, h_s2_gw.detach())

L_total = L_task_s1 + L_task_s2 + λ_sparse * L_sparse + λ_aux * L_aux

---

## 四、三阶段课程学习

```
Stage 1: Simple (episodes 0 ~ N1)
  ┌────────────────────────────────────────────┐
  │ 关卡: GoToObj, GoToRedBall, GoToLocal       │
  │ 特征: 单房间, 无门, 简单导航+基础语言        │
  │                                            │
  │ S2 行为: 快速学会 (~few thousand eps)        │
  │ S1 行为: 缓慢稳定学习                        │
  │ 现象: S2 的 h_s2_gw 质量高 → L_aux 有效      │
  │       S1 技能开始稀疏化 (L_sparse)            │
  └────────────────────────────────────────────┘
                    ↓ 关卡扩充
Stage 2: Medium (episodes N1 ~ N2)
  ┌────────────────────────────────────────────┐
  │ 新增: PickupLoc, PutNextLocal, GoToObjMaze  │
  │ 特征: 多房间, 物体操作, 迷宫导航              │
  │                                            │
  │ S2 行为: 快速迁移 — 利用已有技能适应新关卡    │
  │ S1 行为: 学习新技能 + 保持旧技能 (防遗忘)     │
  │ 现象: S2 继续领先, S1 稳步追赶               │
  │       L_sparse 防止技能分配膨胀              │
  └────────────────────────────────────────────┘
                    ↓ 关卡扩充
Stage 3: Hard (episodes N2 ~ end)
  ┌────────────────────────────────────────────┐
  │ 新增: GoTo (完整 Baby Language)             │
  │ 特征: 复合指令, 隐式子任务, 复杂推理          │
  │                                            │
  │ S2 行为: 继续快速适应复杂推理                  │
  │ S1 行为: 已建立扎实技能基础, 稳定学习          │
  │ 现象: S1 不会遗忘简单任务 (稀疏技能分配)       │
  │       S2 的 GW 在复杂推理中最有用             │
  └────────────────────────────────────────────┘
```

---

## 五、v2 训练算法

```python
def train_v2():
    # 初始化
    task_emb = nn.Embedding(n_levels, 32)          # task_id → (B,32)
    
    s1_encoder = SkillEncoder(n_skills=8)
    s2_encoder = SkillEncoder(n_skills=8)
    s2_encoder.load_state_dict(s1_encoder.state_dict())
    
    s1_task_mlp = TaskMLP(input_dim=8*64+32)       # 512 + 32 = 544
    s1_mha = System1MHA()
    s1_ac = ActorCriticHead(input_dim=64+32)        # 64 + 32 = 96
    
    s2_gw = GlobalWorkspace()
    s2_ac = ActorCriticHead(input_dim=64+32)
    
    # S1 优化器 (包含 Skill GRU 权重)
    opt_s1 = Adam([s1_encoder, s1_task_mlp, s1_mha, s1_ac])
    # S2 优化器 (不包含 Skill GRU 权重 — 梯度在此截断)
    opt_s2 = Adam([s2_encoder.non_skill_params, s2_gw, s2_ac])
    
    stage = 1
    for episode in range(total_episodes):
        # 课程调度
        if episode == N1: stage = 2  # 加入中等关卡
        if episode == N2: stage = 3  # 加入困难关卡
        level = sample_level(stage)
        
        # 前向
        obs, instr = env.reset(level)
        s1_h, s2_h = zeros_states()
        
        for t in range(ep_len):
            e_task = task_emb(torch.tensor([task_id]*B))  # (B,32)
            
            # S1 forward
            h_s1 = s1_encoder(obs, instr, e_task, s1_h)
            h_s1_flat = h_s1.permute(1,0,2).reshape(B, -1)
            skill_logits, _ = s1_task_mlp(torch.cat([h_s1_flat, e_task], -1))
            task_label = gumbel_sigmoid(skill_logits)
            h_s1_out = s1_mha(h_s1, task_label)
            a1, v1 = s1_ac(torch.cat([h_s1_out, e_task], -1))
            
            # S2 forward
            sync_weights(s2_encoder.skill_grus, s1_encoder.skill_grus)
            h_s2_raw = s2_encoder(obs, instr, e_task, s2_h)
            h_s2_gw, _ = s2_gw(h_s2_raw)
            a2, v2 = s2_ac(torch.cat([h_s2_gw, e_task], -1))
            
            # 环境交互 (用 S2 action, S2 学得更快)
            next_obs, reward, done = env.step(a2.argmax())
            
            # 损失
            L1 = ppo_loss(a1, v1, reward)
            L2 = ppo_loss(a2, v2, reward)
            entropy = compute_entropy(skill_logits)
            L_sparse = F.relu(-b_sparse + entropy)
            L_aux = F.mse_loss(h_s1_out, h_s2_gw.detach())
            
            L = L1 + L2 + λ_sparse * L_sparse + λ_aux * L_aux
            
            # 分别更新
            opt_s1.zero_grad(); L.backward(retain_graph=True); opt_s1.step()
            opt_s2.zero_grad(); L2.backward(); opt_s2.step()
            
            s1_h, s2_h = h_s1, h_s2_raw  # 各自独立演化
```

---

## 六、预期现象

```
Stage 1 (Simple):
  S2 success:  ████████████████████ 95%  ← 快速收敛
  S1 success:  ████████████░░░░░░░░ 65%  ← 缓慢但稳定
  S1 entropy:  ████░░░░░░░░░░░░░░░░ 0.4  ← 技能开始稀疏

Stage 2 (Medium):
  S2 success:  ████████████████████ 90%  ← 快速迁移
  S1 success:  ████████████████░░░░ 80%  ← 借助 S2 监督追赶
  S1 entropy:  ██████░░░░░░░░░░░░░░ 0.6  ← 更多技能激活,但受 L_sparse 约束

Stage 3 (Hard):
  S2 success:  ██████████████████░░ 85%  ← 复杂任务仍快
  S1 success:  ████████████████░░░░ 80%  ← 稳定,不遗忘简单任务
  S1 entropy:  ██████████░░░░░░░░░░ 1.0  ← 更多技能,但不过度膨胀

跨阶段评估 (回到 Stage 1 关卡):
  S2:  ████████████████████ 94%  ← 轻微遗忘
  S1:  █████████████████████ 97%  ← 反而更好! (防遗忘优势)
```

**关键验证点**:
1. S2 学习速度 > S1 (简单阶段 S2 先收敛)
2. S2 迁移速度 > S1 (新关卡加入时 S2 快速适应)
3. S1 遗忘率 < S2 (回顾旧关卡时 S1 更稳定)
4. `L_aux` 随训练下降 (S1 逐渐逼近 S2 的表征质量)
5. `L_sparse` 保持合理范围 (技能稀疏但有效)

---

## 七、v1 → v2 代码变更清单

| 文件 | 变更 |
|------|------|
| `src/models/base_encoder.py` | S1/S2 各自实例化, 添加 `sync_weights()` 方法 |
| `src/models/dsna_model.py` | 重写为 `DSNAModelV2` — 双编码器 + 联合训练 |
| `src/models/task_mlp.py` | 保持, 仅 S1 使用 |
| `src/models/system1_pairwise.py` | 保持, 仅 S1 使用 |
| `src/models/system2_workspace.py` | 保持, 仅 S2 使用 |
| `src/models/ac_head.py` | S1/S2 各自独立实例化 |
| `scripts/train_v2.py` | 新文件 — 联合训练 + 课程调度 |
| `config/default_v2.yaml` | 新配置 — 三阶段参数 |
| `src/env/multitask_wrapper.py` | 添加 `sample_level(stage)` 按阶段采样 |
