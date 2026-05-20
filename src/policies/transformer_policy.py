import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.sac.policies import SACPolicy


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


_POLICY_BASE_MAP = {
    "ppo":  ActorCriticPolicy,
    "a2c":  ActorCriticPolicy,
    "td3":  TD3Policy,
    "ddpg": TD3Policy,
    "sac":  SACPolicy,
}

def make_transformer_policy(algo: str):
    """
    Returns a TransformerPolicy class with the correct base
    for the given algorithm.

    Usage:
        policy_cls = make_transformer_policy("ppo")
        model = PPO(policy=policy_cls, env=env)
    """
    if algo not in _POLICY_BASE_MAP:
        raise ValueError(
            f"No transformer policy for '{algo}'. "
            f"Choose from: {list(_POLICY_BASE_MAP)}"
        )

    base = _POLICY_BASE_MAP[algo]

    class TransformerPolicy(base):
        def __init__(self, *args, **kwargs):
            print(f"[TransformerPolicy] base={base.__name__}, algo={algo}")
            kwargs["features_extractor_class"]  = _TransformerFeatureExtractor
            super().__init__(*args, **kwargs)

    TransformerPolicy.__name__     = f"Transformer{base.__name__}"
    TransformerPolicy.__qualname__ = TransformerPolicy.__name__

    return TransformerPolicy
