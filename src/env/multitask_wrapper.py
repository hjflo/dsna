"""
BabyAI 多任务环境包装器 (使用 gymnasium + minigrid)
支持多关卡均匀采样和多进程并行。
"""
import numpy as np
import gymnasium as gym
from minigrid.envs.babyai import (
    GoToObj, GoToRedBallGrey, GoToRedBall, GoToLocal,
    PutNextLocal, PickupLoc, GoTo, Pickup
)

# 环境ID映射 (8关: 7关直接映射 + 1关用GoTo替代GoToObjMaze)
LEVEL_MAP = {
    'GoToObj': 'BabyAI-GoToObj-v0',
    'GoToRedBallGrey': 'BabyAI-GoToRedBallGrey-v0',
    'GoToRedBall': 'BabyAI-GoToRedBall-v0',
    'GoToLocal': 'BabyAI-GoToLocal-v0',
    'PutNextLocal': 'BabyAI-PutNextLocal-v0',
    'PickupLoc': 'BabyAI-PickupLoc-v0',
    'GoToObjMaze': 'BabyAI-GoTo-v0',         # GoTo 替代 GoToObjMaze (迷宫+目标导航)
    'GoTo': 'BabyAI-GoTo-v0',
}


class MultiTaskBabyAIEnv:
    """
    包装多个 BabyAI 关卡为统一的多任务环境。
    每个 episode 随机采样一个关卡。
    """
    def __init__(self, levels, max_steps=128):
        self.levels = levels
        self.max_steps = max_steps
        self.envs = {}
        self.current_levels = []
        self._step_counts = []

    def _get_env(self, level_name):
        if level_name not in self.envs:
            gym_id = LEVEL_MAP.get(level_name, f'BabyAI-{level_name}-v0')
            self.envs[level_name] = gym.make(gym_id)
        return self.envs[level_name]

    def reset(self, n_envs=1):
        """重置 n_envs 个并行环境，随机采样关卡 (gymnasium API)"""
        self.current_levels = [np.random.choice(self.levels) for _ in range(n_envs)]
        self._step_counts = [0] * n_envs

        obs_list, instr_list = [], []
        for lvl in self.current_levels:
            env = self._get_env(lvl)
            obs, _ = env.reset()           # gymnasium returns (obs, info)
            obs_list.append(obs['image'])
            instr_tokens = self._tokenize(obs['mission'])
            instr_list.append(instr_tokens)

        return obs_list, instr_list

    def step(self, actions):
        """执行一步，支持并行 (gymnasium API)"""
        if isinstance(actions, (int, np.integer)):
            actions = [actions]

        next_obs, rewards, dones, infos = [], [], [], []
        for i, (action, level) in enumerate(zip(actions, self.current_levels)):
            env = self._get_env(level)
            self._step_counts[i] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if self._step_counts[i] >= self.max_steps:
                done = True

            next_obs.append(obs['image'])
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return next_obs, rewards, dones, infos

    def _tokenize(self, mission_str, max_len=32):
        """简化版 tokenization: 用字符 hash 编码"""
        tokens = [hash(c) % 1000 + 1 for c in mission_str[:max_len]]
        tokens = tokens + [0] * (max_len - len(tokens))  # pad
        return np.array(tokens, dtype=np.int64)
