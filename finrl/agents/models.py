from __future__ import annotations

from typing import Any, Dict, Optional, Type
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from policies.transformer_policy import TransformerPolicy, VecSequenceWrapper


MODEL_MAP: Dict[str, Type] = {
    "a2c": A2C,
    "ppo": PPO,
    "ddpg": DDPG,
    "sac": SAC,
    "td3": TD3,
    "recurrent_ppo": RecurrentPPO,
}


class DRLAgent:
    """
    Minimal Stable-Baselines3 agent wrapper for RL trading or general RL.
    """

    def __init__(
        self,
        model_name: str,
        env,
        model_kwargs: Optional[Dict[str, Any]] = None,
        policy: str = "MlpPolicy",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        if model_name not in MODEL_MAP:
            raise ValueError(f"Unknown model_name '{model_name}'")

        self.model_name = model_name
        self.env = env
        self.policy = policy
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log

        self.model_kwargs = model_kwargs or {}
        self.model = None

    def build(self):
        """
        Create SB3 model instance.
        """
        model_class = MODEL_MAP[self.model_name]

        kwargs = dict(self.model_kwargs)

        env = DummyVecEnv([lambda e=self.env: e])
        self.env = VecSequenceWrapper(env, seq_len=32)

        self.model = model_class(
            policy=TransformerPolicy,
            env=self.env,
            verbose=self.verbose,
            tensorboard_log=self.tensorboard_log,
            **kwargs,
        )

        print(f'Policy:')
        print(self.model.policy)
        return self.model

    def train(
        self,
        total_timesteps: int = 100_000,
        tb_log_name: Optional[str] = None,
    ):
        """
        Train the agent.
        """
        if self.model is None:
            self.build()

        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name or self.model_name,
        )

        return self

    def predict(self, obs, deterministic: bool = True):
        """
        Get action from trained policy.
        """
        if self.model is None:
            raise ValueError("Model is not built or loaded.")

        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str):
        """
        Save trained model.
        """
        if self.model is None:
            raise ValueError("No model to save.")

        self.model.save(path)

    @classmethod
    def load(
        cls,
        model_name: str,
        env,
        path: str,
        seed: Optional[int] = None,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load a trained SB3 model into wrapper.
        """
        if model_name not in MODEL_MAP:
            raise ValueError(f"Unknown model_name '{model_name}'")

        model_class = MODEL_MAP[model_name]
        model = model_class.load(path, env=env)

        wrapper = cls(
            model_name=model_name,
            env=env,
            model_kwargs=model_kwargs,
            seed=seed,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
        )

        wrapper.model = model
        return wrapper

    def get_model(self):
        """
        Return raw SB3 model (optional escape hatch).
        """
        return self.model