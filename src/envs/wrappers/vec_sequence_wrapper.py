import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv
import gymnasium as gym


class VecSequenceWrapper(VecEnvWrapper):
    """
    Wraps a VecEnv so each observation becomes (seq_len, obs_dim)
    by maintaining a rolling buffer of the last seq_len observations.

    - On reset: buffer is zero-padded
    - On done:  that env's buffer is zeroed and refilled from new obs
    """

    def __init__(self, venv: VecEnv, seq_len: int):
        obs_space = venv.observation_space
        assert len(obs_space.shape) == 1, \
            "VecSequenceWrapper expects flat obs (obs_dim,)"
        assert seq_len > 0, "seq_len must be positive"

        self.seq_len = seq_len
        self.obs_dim = obs_space.shape[0]
        self.n_envs  = venv.num_envs

        # Rolling buffer: (n_envs, seq_len, obs_dim)
        self._buffer = np.zeros(
            (self.n_envs, seq_len, self.obs_dim), dtype=np.float32
        )

        # New observation space: (seq_len, obs_dim)
        new_obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_len, self.obs_dim),
            dtype=np.float32,
        )

        print(f'Wrapping VecEnv with VecSequenceWrapper: seq_len={seq_len}, obs_dim={self.obs_dim}')
        print(f'Initialized VecSequenceWrapper with {self.n_envs} envs and observation shape: {new_obs_space.shape}')

        super().__init__(venv, observation_space=new_obs_space)

    def _push(self, obs: np.ndarray, env_indices=None):
        """Append new obs to buffer, drop oldest."""
        if env_indices is None:
            env_indices = np.arange(self.n_envs)
        # Shift left, insert at end
        self._buffer[env_indices, :-1, :] = self._buffer[env_indices, 1:, :]
        self._buffer[env_indices, -1, :]  = obs[env_indices]

    def reset(self):
        obs = self.venv.reset()
        self._buffer[:] = 0.0              # full zero-pad on global reset
        self._buffer[:, -1, :] = obs       # place first obs at last position
        return self._buffer.copy()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # Push new obs for ALL envs
        self._push(obs)

        # Reset buffer for envs that finished
        done_envs = np.where(dones)[0]
        if len(done_envs):
            self._buffer[done_envs] = 0.0
            # The obs returned for done envs is already the NEXT episode's
            # first obs (SB3 auto-resets) — place it at last position
            self._buffer[done_envs, -1, :] = obs[done_envs]

        return self._buffer.copy(), rewards, dones, infos