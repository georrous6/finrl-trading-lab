import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.sac.policies import SACPolicy

try:
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy
except ImportError:
    MultiInputActorCriticPolicy = ActorCriticPolicy

try:
    from stable_baselines3.td3.policies import MultiInputPolicy as TD3MultiInputPolicy
except ImportError:
    TD3MultiInputPolicy = TD3Policy

try:
    from stable_baselines3.sac.policies import MultiInputPolicy as SACMultiInputPolicy
except ImportError:
    SACMultiInputPolicy = SACPolicy


class _TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Input:
        Dict obs: {"portfolio": (batch, lookback, dim_p), "market": (batch, lookback, dim_m)}
        Flat obs: (batch, lookback, obs_dim)
    Output: (batch, d_model) -> fed into SB3 actor/critic heads
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim=d_model)

        if isinstance(observation_space, gym.spaces.Dict):
            portfolio_space = observation_space.spaces["portfolio"]
            market_space = observation_space.spaces["market"]
            lookback = market_space.shape[0]
            obs_dim = portfolio_space.shape[1] + market_space.shape[1]
            print(f"Observation dict shapes: portfolio={portfolio_space.shape}, market={market_space.shape}")
        else:
            print(f"Observation shape: {observation_space.shape}")
            lookback, obs_dim = observation_space.shape

        self.input_proj = nn.Linear(obs_dim, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(th.zeros(1, lookback, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, d_model) convention
            norm_first=True,  # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs) -> th.Tensor:
        if isinstance(obs, dict):
            x = th.cat([obs["portfolio"], obs["market"]], dim=-1)
        else:
            x = obs
        x = self.input_proj(x)             # (batch, lookback, d_model)
        x = x + self.pos_embedding           # add positional info
        x = self.transformer(x)              # (batch, lookback, d_model)
        x = self.norm(x[:, -1, :])          # last token → (batch, d_model)
        return x


_POLICY_BASE_MAP = {
    "ppo": MultiInputActorCriticPolicy,
    "a2c": MultiInputActorCriticPolicy,
    "td3": TD3MultiInputPolicy,
    "ddpg": TD3MultiInputPolicy,
    "sac": SACMultiInputPolicy,
}

def make_transformer_policy(model_name: str):
    """
    Returns a TransformerPolicy class with the correct base
    for the given model.

    Usage:
        policy_cls = make_transformer_policy("ppo")
        model = PPO(policy=policy_cls, env=env)
    """
    if model_name not in _POLICY_BASE_MAP:
        raise ValueError(
            f"No transformer policy for '{model_name}'. "
            f"Choose from: {list(_POLICY_BASE_MAP)}"
        )

    base = _POLICY_BASE_MAP[model_name]

    class TransformerPolicy(base):
        def __init__(self, *args, **kwargs):
            print(f"[TransformerPolicy] base={base.__name__}, model_name={model_name}")
            kwargs["features_extractor_class"]  = _TransformerFeatureExtractor
            super().__init__(*args, **kwargs)

    TransformerPolicy.__name__ = f"Transformer{base.__name__}"
    TransformerPolicy.__qualname__ = TransformerPolicy.__name__
    TransformerPolicy.display_name = "TransformerPolicy"

    return TransformerPolicy
