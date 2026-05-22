import gymnasium as gym

from .mlp_transformer_policy import make_mlp_transformer_policy
from .transformer_policy import make_transformer_policy
from configs import config


_POLICY_MAP = {
    "MlpPolicy": {
        "cls": "MlpPolicy",
        "kwargs": {},
        "supported_models": {"ppo", "a2c", "td3", "ddpg", "sac"},
        "requires_sequence": False,
    },
    "MlpLstmPolicy": {
        "cls": "MlpLstmPolicy",
        "kwargs": {},
        "supported_models": {"recurrent_ppo"},
        "requires_sequence": False,
    },
    "TransformerPolicy": {
        "cls": make_transformer_policy,
        "kwargs": {
            "ppo": config.ON_POLICY_TRANSFORMER_PARAMS,
            "a2c": config.ON_POLICY_TRANSFORMER_PARAMS,
            "td3": config.OFF_POLICY_TRANSFORMER_PARAMS,
            "ddpg": config.OFF_POLICY_TRANSFORMER_PARAMS,
            "sac": config.OFF_POLICY_TRANSFORMER_PARAMS,
        },
        "supported_models": {"ppo", "a2c", "td3", "ddpg", "sac"},
        "requires_sequence": True,
    },
    "MlpTransformerPolicy": {
        "cls": make_mlp_transformer_policy,
        "kwargs": {
            "ppo": config.ON_POLICY_TRANSFORMER_PARAMS,
            "a2c": config.ON_POLICY_TRANSFORMER_PARAMS,
            "td3": config.OFF_POLICY_TRANSFORMER_PARAMS,
            "ddpg": config.OFF_POLICY_TRANSFORMER_PARAMS,
            "sac": config.OFF_POLICY_TRANSFORMER_PARAMS,
        },
        "supported_models": {"ppo", "a2c", "td3", "ddpg", "sac"},
        "requires_sequence": True,
    },
}


def make_policy(model_name: str, 
                policy_name: str, 
                observation_space: gym.spaces.Space | None = None):
    if policy_name not in _POLICY_MAP:
        raise ValueError(
            f"Unknown policy '{policy_name}'. "
            f"Choose from: {list(_POLICY_MAP)}"
        )
    if model_name not in _POLICY_MAP[policy_name]["supported_models"]:
        raise ValueError(
            f"Policy '{policy_name}' does not support model '{model_name}'. "
            f"Supported models: {_POLICY_MAP[policy_name]['supported_models']}"
        )

    policy_cls = _POLICY_MAP[policy_name]["cls"]
    if isinstance(observation_space, gym.spaces.Dict):
        if policy_name == "MlpPolicy":
            policy_cls = "MultiInputPolicy"
        elif policy_name == "MlpLstmPolicy":
            policy_cls = "MultiInputLstmPolicy"

    policy = policy_cls(model_name) if callable(policy_cls) else policy_cls
    policy_kwargs = _POLICY_MAP[policy_name]["kwargs"].get(model_name, {})
    requires_sequence = _POLICY_MAP[policy_name]["requires_sequence"]
    return policy, policy_kwargs, requires_sequence
