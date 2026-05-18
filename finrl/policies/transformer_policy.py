import torch as th
import numpy as np
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv


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

        print(f'Initialized VecSequenceWrapper with observation shape: {new_obs_space.shape}')

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


class _TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Input:  obs (batch, lookback, obs_dim) <- directly from the env
    Output: (batch, d_model) -> fed into SB3 actor/critic heads
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim=d_model)

        print(f'Observation shape: {observation_space.shape}')
        lookback, obs_dim = observation_space.shape  # unpack (seq, feat)

        self.input_proj = nn.Linear(obs_dim, d_model)

        # Learnable positional encoding — better than sinusoidal for RL
        self.pos_embedding = nn.Parameter(th.zeros(1, lookback, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,        # (batch, seq, d_model) convention
            norm_first=True,         # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # obs: (batch, lookback, obs_dim)
        x = self.input_proj(obs)             # (batch, lookback, d_model)
        x = x + self.pos_embedding           # add positional info
        x = self.transformer(x)              # (batch, lookback, d_model)
        x = self.norm(x[:, -1, :])          # last token → (batch, d_model)
        return x


class TransformerPolicy(ActorCriticPolicy):
    """
    Drop-in SB3 policy that uses _TransformerFeatureExtractor.
    Pass directly as policy=TransformerPolicy — no policy_kwargs needed.
    """

    def __init__(self, *args, transformer_kwargs: dict = {}, **kwargs):
        print('Using custom transformer policy')
        kwargs["features_extractor_class"]  = _TransformerFeatureExtractor
        kwargs["features_extractor_kwargs"] = transformer_kwargs
        kwargs.setdefault("normalize_images", False)
        super().__init__(*args, **kwargs)
