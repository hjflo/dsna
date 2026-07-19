"""
BabyAI 多任务环境包装器
支持多关卡均匀采样和多进程并行。
"""
import numpy as np
import gym
import babyai


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
            gym_id = f'BabyAI-{level_name}-v0'
            self.envs[level_name] = gym.make(gym_id)
        return self.envs[level_name]

    def reset(self, n_envs=1):
        """重置 n_envs 个并行环境，随机采样关卡"""
        self.current_levels = [np.random.choice(self.levels) for _ in range(n_envs)]
        self._step_counts = [0] * n_envs

        obs_list, instr_list = [], []
        for lvl in self.current_levels:
            env = self._get_env(lvl)
            obs = env.reset()
            obs_list.append(obs['image'])
            # 将指令转为 token IDs (简化: 用 hash)
            instr_tokens = self._tokenize(obs['mission'])
            instr_list.append(instr_tokens)

        return obs_list, instr_list

    def step(self, actions):
        """执行一步 (支持并行)"""
        if isinstance(actions, (int, np.integer)):
            actions = [actions]

        next_obs, rewards, dones, infos = [], [], [], []
        for i, (action, level) in enumerate(zip(actions, self.current_levels)):
            env = self._get_env(level)
            self._step_counts[i] += 1

            obs, reward, done, info = env.step(action)

            # 超时也算 done
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
