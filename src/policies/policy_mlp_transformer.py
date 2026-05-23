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


class _MlpTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Split architecture:

    Market: (batch, lookback, market_dim) -> Transformer
    Portfolio: (batch, portfolio_dim) -> MLP
    Fusion: concat([market_emb, portfolio_emb])
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

        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("MlpTransformerFeatureExtractor expects Dict observation_space")

        market_space = observation_space.spaces["market"]
        portfolio_space = observation_space.spaces["portfolio"]
        lookback = market_space.shape[0]
        self.market_dim = market_space.shape[1]
        self.portfolio_dim = portfolio_space.shape[1]

        print(f"Observation dict shapes: portfolio={portfolio_space.shape}, market={market_space.shape}")

        # Market transformer
        self.market_proj = nn.Linear(self.market_dim, d_model)

        self.pos_embedding = nn.Parameter(
            th.zeros(1, lookback, d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.market_norm = nn.LayerNorm(d_model)

        # Portfolio MLP
        self.portfolio_net = nn.Sequential(
            nn.Linear(self.portfolio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Market + Portfolio fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 64, d_model),
            nn.ReLU(),
        )

    def forward(self, obs) -> th.Tensor:
        market = obs["market"]
        portfolio = obs["portfolio"][:, -1, :]

        # Market -> Transformer
        x = self.market_proj(market)
        x = x + self.pos_embedding
        x = self.transformer(x)
        market_emb = self.market_norm(x[:, -1, :])

        # Portfolio -> MLP
        portfolio_emb = self.portfolio_net(portfolio)

        # Fusion
        fused = th.cat([market_emb, portfolio_emb], dim=-1)
        return self.fusion(fused)


_POLICY_BASE_MAP = {
    "ppo": MultiInputActorCriticPolicy,
    "a2c": MultiInputActorCriticPolicy,
    "td3": TD3MultiInputPolicy,
    "ddpg": TD3MultiInputPolicy,
    "sac": SACMultiInputPolicy,
}

def make_mlp_transformer_policy(model_name: str):
    """
    Returns a MlpTransformerPolicy class with the correct base
    for the given model.

    Usage:
        policy_cls = make_mlp_transformer_policy("ppo")
        model = PPO(policy=policy_cls, env=env)
    """
    if model_name not in _POLICY_BASE_MAP:
        raise ValueError(
            f"No transformer policy for '{model_name}'. "
            f"Choose from: {list(_POLICY_BASE_MAP)}"
        )

    base = _POLICY_BASE_MAP[model_name]

    class MlpTransformerPolicy(base):
        def __init__(self, *args, **kwargs):
            print(f"[MlpTransformerPolicy] base={base.__name__}, model_name={model_name}")
            kwargs["features_extractor_class"]  = _MlpTransformerFeatureExtractor
            super().__init__(*args, **kwargs)

    MlpTransformerPolicy.__name__ = f"MlpTransformer{base.__name__}"
    MlpTransformerPolicy.__qualname__ = MlpTransformerPolicy.__name__
    MlpTransformerPolicy.display_name = "MlpTransformerPolicy"

    return MlpTransformerPolicy
