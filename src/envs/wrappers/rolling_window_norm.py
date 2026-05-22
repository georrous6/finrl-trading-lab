from collections import deque
import numpy as np
import gymnasium as gym


class RollingWindowNorm(gym.Wrapper):

    def __init__(
        self,
        env,
        obs_window=100,
        reward_window=100,
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

        obs_space = env.observation_space
        self._dict_obs = isinstance(obs_space, gym.spaces.Dict)

        if self._dict_obs:
            self.obs_buffer = {
                key: deque(maxlen=obs_window) for key in obs_space.spaces
            }
            self.obs_sum = {
                key: np.zeros(obs_space.spaces[key].shape, dtype=np.float64)
                for key in obs_space.spaces
            }
            self.obs_sq_sum = {
                key: np.zeros(obs_space.spaces[key].shape, dtype=np.float64)
                for key in obs_space.spaces
            }
        else:
            obs_shape = obs_space.shape
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

        if self._dict_obs:
            return {
                key: self._normalize_obs_array(obs[key], key)
                for key in obs
            }

        return self._normalize_obs_array(obs)

    def _normalize_obs_array(self, obs, key=None):
        obs = obs.astype(np.float64)

        if key is None:
            obs_buffer = self.obs_buffer
            obs_sum = self.obs_sum
            obs_sq_sum = self.obs_sq_sum
        else:
            obs_buffer = self.obs_buffer[key]
            obs_sum = self.obs_sum[key]
            obs_sq_sum = self.obs_sq_sum[key]

        if len(obs_buffer) == obs_buffer.maxlen:
            old = obs_buffer[0]
            obs_sum -= old
            obs_sq_sum -= old * old

        obs_buffer.append(obs)
        obs_sum += obs
        obs_sq_sum += obs * obs

        n = len(obs_buffer)

        if n < 2:
            return obs.astype(np.float32)

        mean = obs_sum / n
        var = (obs_sq_sum / n) - mean**2
        std = np.sqrt(np.maximum(var, 0.0))

        return ((obs - mean) / (std + self.eps)).astype(np.float32)

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