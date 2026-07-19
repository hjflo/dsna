# BabyAI 环境说明

> DSNA 项目使用 [Farama Foundation](https://farama.org) 维护的 `minigrid` 包中的 BabyAI 关卡。

---

## 一、环境来源

```
pip install minigrid==3.1.0
  → 安装位置: {python}/site-packages/minigrid/
  → BabyAI 关卡: minigrid/envs/babyai/
  → 核心引擎:    minigrid/core/
```

BabyAI 关卡是**程序化生成**的——无需下载任何外部数据集。每次 `env.reset()` 实时随机生成全新的房间布局、物体放置和指令文本。

---

## 二、观测空间

每次 `env.step()` 返回的观测字典：

| 字段 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `image` | numpy uint8 | (7, 7, 3) | 局部 7×7 视野，每格 3 个整数 |
| `mission` | str | — | 自然语言指令，如 `"go to the red ball"` |
| `direction` | int | — | Agent 朝向 (0=右, 1=下, 2=左, 3=上) |

**image 的 3 通道编码**:

| 通道 | 含义 | 取值 |
|------|------|------|
| 通道 0 | 物体类型 | 0=空, 1=墙, 2=门, 3=钥匙, 4=球, 5=盒子, 6=目标 |
| 通道 1 | 物体颜色 | 0=红, 1=绿, 2=蓝, 3=紫, 4=黄, 5=灰 |
| 通道 2 | 门状态 | 0=开, 1=关, 2=锁 |

---

## 三、动作空间（7 个离散动作）

| 编号 | 动作 | 说明 |
|------|------|------|
| 0 | left | 左转 90° |
| 1 | right | 右转 90° |
| 2 | forward | 向前移动一格 |
| 3 | pickup | 拾取面前物体 |
| 4 | drop | 放下手中物体 |
| 5 | toggle | 开关面前的门 |
| 6 | done | 完成任务（仅用于模仿学习） |

---

## 四、奖励函数

```
成功完成指令:  reward = 1 - 0.9 × (step_count / max_steps)
失败/超时:     reward = 0

说明: 越快完成任务，奖励越接近 1.0；
      未完成或超时奖励为 0（稀疏奖励）。
```

---

## 五、8 个训练关卡

| 关卡 | 环境ID | 能力要求 | 源码文件 |
|------|--------|----------|----------|
| **GoToObj** | `BabyAI-GoToObj-v0` | 房间导航，找到唯一物体 | `goto.py` |
| **GoToRedBallGrey** | `BabyAI-GoToRedBallGrey-v0` | 在灰色干扰物中找到红球 | `goto.py` |
| **GoToRedBall** | `BabyAI-GoToRedBall-v0` | 在多彩干扰物中找到红球 | `goto.py` |
| **GoToLocal** | `BabyAI-GoToLocal-v0` | 理解位置语言（"你左边的红球"） | `goto.py` |
| **PutNextLocal** | `BabyAI-PutNextLocal-v0` | 把一个物体放到另一个旁边 | `putnext.py` |
| **PickupLoc** | `BabyAI-PickupLoc-v0` | 捡起特定位置的物体 | `pickup.py` |
| **GoToObjMaze** | `BabyAI-GoTo-v0` ⚠️ | 迷宫+多房间导航+找物体 | `goto.py` |
| **GoTo** | `BabyAI-GoTo-v0` | 完整 Baby Language 理解 | `goto.py` |

> ⚠️ `GoToObjMaze` 在 minigrid 3.1.0 中不存在独立的 Gym ID，使用 `BabyAI-GoTo-v0` 替代。

---

## 六、程序化关卡生成流程

```python
# 每次 env.reset() 时调用:
def gen_mission(self):
    self.place_agent()                          # 随机放置 Agent
    objs = self.add_distractors(num_distractors) # 随机生成干扰物
    obj = objs[0]                               # 随机选择一个作为目标
    self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))  # 生成指令文本
```

**随机化范围**:
- 房间大小: 6×6 ~ 8×8（取决于关卡）
- 物体类型: 球/钥匙/盒子
- 物体颜色: 红/绿/蓝/紫/黄/灰
- 指令类型: go to / pick up / put next to / open / unlock
- 迷宫复杂度: 1~9 个房间，门随机锁定

---

## 七、环境的 Python 环境信息

```
Python:     3.13
minigrid:   3.1.0
gymnasium:  1.3.0  (OpenAI Gym 的继任者)
pygame-ce:  2.5.7  (渲染引擎)
numpy:      2.x
```

---

## 八、在我们的项目中使用

```python
from src.env.multitask_wrapper import MultiTaskBabyAIEnv

env = MultiTaskBabyAIEnv(
    levels=['GoToObj', 'GoToLocal', 'PickupLoc', 'PutNextLocal',
            'GoToRedBallGrey', 'GoToRedBall', 'GoToObjMaze', 'GoTo'],
    max_steps=128
)

obs_list, instr_list = env.reset(n_envs=4)   # 4 并行环境, 随机采样关卡
next_obs, rewards, dones, infos = env.step(actions)
```
