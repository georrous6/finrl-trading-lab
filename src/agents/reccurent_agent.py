from typing import Optional, Literal
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
import gymnasium as gym
import numpy as np

from policies.transformer_policy import make_transformer_policy
from utils.normalize import RollingWindowNorm
from utils.sequential import VecSequenceWrapper
from stable_baselines3.common.vec_env import VecMonitor
from utils.loggers import TrainingLogger, BacktestLogger
from pathlib import Path
from configs import config


_NORM_MAP = {
    "rolling_window": {
        "cls": RollingWindowNorm,
        "kwargs": config.ROLLING_WINDOW_NORM_PARAMS,
    },
    None: {
        "cls": lambda env, **kwargs: env,  # identity function
        "kwargs": {},
    },
}

_POLICY_MAP = {
    "TransformerPolicy": {
        "cls": make_transformer_policy,
        "kwargs": {
            "ppo": config.ON_POLICY_TRANSFORMER_PARAMS,
            "a2c": config.ON_POLICY_TRANSFORMER_PARAMS,
            "td3": config.OFF_POLICY_TRANSFORMER_PARAMS,
            "ddpg": config.OFF_POLICY_TRANSFORMER_PARAMS,
            "sac": config.OFF_POLICY_TRANSFORMER_PARAMS,
        }
    }
}

_MODEL_MAP = {
    "ppo":  {
        "cls": PPO,
        "kwargs": config.PPO_PARAMS
    },
    "a2c":  {
        "cls": A2C,
        "kwargs": config.A2C_PARAMS
    },
    "td3":  {
        "cls": TD3,
        "kwargs": config.TD3_PARAMS
    },
    "ddpg": {
        "cls": DDPG,
        "kwargs": config.DDPG_PARAMS
    },
    "sac":  {
        "cls": SAC,
        "kwargs": config.SAC_PARAMS
    },
}


class ReccurentDRLAgent:
    """
    Sequential DRL agent.

    Args:
        model_name:      Model name -- 'ppo', 'a2c', 'sac', 'td3', 'ddpg'
        policy:          Policy name -- 'TransformerPolicy', 'MlpPolicy', 'LstmMlpPolicy'
        env:             Raw gymnasium env (unwrapped)
        seq_len:         Lookback window for VecSequenceWrapper
        norm:            Normalizer name -- 'rolling_window' or None
        verbose:         SB3 verbosity
        mode:            One of 'train', 'backtest', 'trade'. Affects log directory.
    """

    def __init__(
        self,
        model_name: str,
        policy: str,
        env: gym.Env,
        mode: Literal["train", "backtest", "trade"],
        seq_len: int = 32,
        norm: Optional[str] = "rolling_window",
        verbose: int = 1,
    ):
        self.model_name = model_name
        self.policy_name = policy

        # validation
        if policy not in _POLICY_MAP:
            raise ValueError(f"Unknown policy '{policy}'. "
                             f"Choose from: {list(_POLICY_MAP.keys())}")
        if model_name not in _MODEL_MAP:
            raise ValueError(f"Unknown model '{model_name}'. "
                             f"Choose from: {list(_MODEL_MAP.keys())}")
        if norm not in _NORM_MAP:
            raise ValueError(f"Unknown norm '{norm}'. "
                             f"Choose from: {list(_NORM_MAP.keys())}")

        self.log_root = (
            Path(config.LOG_DIR) / 
            mode / 
            self.model_name / 
            self.policy_name
        )

        norm_kwargs = _NORM_MAP[norm]["kwargs"]
        policy_kwargs = _POLICY_MAP[policy]["kwargs"][model_name]
        model_kwargs = dict(_MODEL_MAP[model_name]["kwargs"])  # Copy, don't mutate global config
        model_kwargs["policy_kwargs"] = policy_kwargs

        # environment pipeline
        norm_cls = _NORM_MAP[norm]["cls"]
        env = norm_cls(env, **norm_kwargs)
        env = DummyVecEnv([lambda e=env: e])
        env = VecSequenceWrapper(env, seq_len=seq_len)
        self.env = VecMonitor(env)
        self.verbose = verbose

        # policy
        policy_cls = _POLICY_MAP[policy]["cls"](model_name)

        # trainable model
        model_cls = _MODEL_MAP[model_name]["cls"]
        self.model = model_cls(
            policy=policy_cls,
            env=self.env,
            verbose=self.verbose,
            tensorboard_log=str(self.log_root),
            **model_kwargs,
        )


    def train(self, 
              total_timesteps: int, 
              interval: str = "1d",):

        callback = TrainingLogger(interval=interval)
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name="run",
            callback=callback
        )
        return self


    def _make_run_dir(self) -> str:

        i = 1
        path = self.log_root / f"run_{i}"

        # Keep incrementing counter as long as the file already exists
        while path.exists():
            i += 1
            path = self.log_root / f"run_{i}"
        
        path.mkdir(parents=True, exist_ok=False)
        return str(path)


    def _make_output_path(self, root_dir: str, suffix: str = "") -> str:
        i = 1
        path = Path(root_dir) / f"{self.model_name}_{self.policy_name}_{i}{suffix}"

        # Keep incrementing counter as long as the file already exists
        while path.exists():
            i += 1
            path = Path(root_dir) / f"{self.model_name}_{self.policy_name}_{i}{suffix}"
        
        return str(path)


    def backtest(
        self,
        path: str,
        deterministic: bool = True,
        interval: str = "1d",
    ):

        self._load_pretrained_model(path)
        log_dir = self._make_run_dir()
        logger = BacktestLogger(log_dir=log_dir, 
                                interval=interval, 
                                verbose=self.verbose)

        obs = self.env.reset()
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, _, done, info = self.env.step(action)

            if "portfolio_value" in info[0]:
                logger.log_step(info[0]["portfolio_value"])

            done = done[0]   # DummyVecEnv returns array

        logger.close()


    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)


    def save(self, path: Optional[str] = None):
        if path is None:
            path = self._make_output_path(config.TRAINED_MODEL_DIR, suffix=".zip")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"Model saved to {str(path)}")


    def _load_pretrained_model(self, path: str):
        model_cls = _MODEL_MAP[self.model_name]["cls"]
        self.model = model_cls.load(path, env=self.env)