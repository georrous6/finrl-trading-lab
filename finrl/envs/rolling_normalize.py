from collections import deque
import numpy as np
import gymnasium as gym


class RollingNormalize(gym.Wrapper):

    def __init__(
        self,
        env,
        obs_window=1000,
        reward_window=1000,
        norm_obs=True,
        norm_reward=True,
        eps=1e-8,
    ):
        super().__init__(env)

        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.eps = eps

        self.obs_buffer = deque(maxlen=obs_window)
        self.reward_buffer = deque(maxlen=reward_window)

        obs_shape = env.observation_space.shape

        self.obs_sum = np.zeros(obs_shape, dtype=np.float64)
        self.obs_sq_sum = np.zeros(obs_shape, dtype=np.float64)

        self.reward_sum = 0.0
        self.reward_sq_sum = 0.0

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        if self.norm_obs:
            obs = self._normalize_obs(obs)

        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.norm_obs:
            obs = self._normalize_obs(obs)

        if self.norm_reward:
            reward = self._normalize_reward(reward)

        return obs, reward, terminated, truncated, info

    def _normalize_obs(self, obs):

        obs = obs.astype(np.float64)

        if len(self.obs_buffer) == self.obs_buffer.maxlen:

            old = self.obs_buffer[0]

            self.obs_sum -= old
            self.obs_sq_sum -= old * old

        self.obs_buffer.append(obs)

        self.obs_sum += obs
        self.obs_sq_sum += obs * obs

        n = len(self.obs_buffer)

        if n < 2:
            return obs.astype(np.float32)

        mean = self.obs_sum / n

        var = (self.obs_sq_sum / n) - mean**2

        std = np.sqrt(
            np.maximum(var, 0.0)
        )

        return (
            (obs - mean)
            / (std + self.eps)
        ).astype(np.float32)

    def _normalize_reward(self, reward):

        reward = float(reward)

        if len(self.reward_buffer) == self.reward_buffer.maxlen:

            old = self.reward_buffer[0]

            self.reward_sum -= old
            self.reward_sq_sum -= old**2

        self.reward_buffer.append(reward)

        self.reward_sum += reward
        self.reward_sq_sum += reward**2

        n = len(self.reward_buffer)

        if n < 2:
            return reward

        mean = self.reward_sum / n

        var = (
            self.reward_sq_sum / n
            - mean**2
        )

        std = np.sqrt(
            max(var, 0.0)
        )

        return float(
            (reward - mean)
            / (std + self.eps)
        )