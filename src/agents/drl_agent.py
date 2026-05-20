from typing import Optional, Literal
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
import gymnasium as gym
import numpy as np
from pathlib import Path
from sb3_contrib import RecurrentPPO

from policies.transformer_policy import make_transformer_policy
from utils.loggers import TrainingLogger, BacktestLogger
from envs.assembly import make_env
from configs import config


_POLICY_MAP = {
    "MlpPolicy": {
        "type": "builtin",
        "requires_sequence": False,
        "cls": None,
        "kwargs": {},  # use kwargs from _MODEL_MAP
        "supported_models": {"ppo", "a2c", "td3", "ddpg", "sac"},
    },
    "MlpLstmPolicy": {
        "type": "builtin",
        "requires_sequence": False,
        "cls": None,
        "kwargs": {},  # use kwargs from _MODEL_MAP
        "supported_models": {"recurrent_ppo"},
    },
    "TransformerPolicy": {
        "type": "custom",
        "requires_sequence": True,
        "cls": make_transformer_policy,
        "kwargs": {
            "ppo": config.ON_POLICY_TRANSFORMER_PARAMS,
            "a2c": config.ON_POLICY_TRANSFORMER_PARAMS,
            "td3": config.OFF_POLICY_TRANSFORMER_PARAMS,
            "ddpg": config.OFF_POLICY_TRANSFORMER_PARAMS,
            "sac": config.OFF_POLICY_TRANSFORMER_PARAMS,
        },
        "supported_models": {"ppo", "a2c", "td3", "ddpg", "sac"},
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
    "recurrent_ppo": {
        "cls": RecurrentPPO,
        "kwargs": config.RECURRENT_PPO_PARAMS
    }
}


class DRLAgent:
    """
    Unified DRL agent for training, backtesting, and trading.

    Args:
        model_name:      Model name -- 'ppo', 'a2c', 'sac', 'td3', 'ddpg'
        policy_name:     Policy name -- 'TransformerPolicy', 'MlpPolicy', 'LstmMlpPolicy'
        env:             Raw gymnasium env (unwrapped)
        seq_len:         Lookback window for VecSequenceWrapper
        norm:            Normalizer name -- 'rolling_window' or None
        verbose:         SB3 verbosity
        mode:            One of 'train', 'backtest', 'trade'. Affects log directory.
    """

    def __init__(
        self,
        model_name: str,
        policy_name: str,
        env: gym.Env,
        mode: Literal["train", "backtest", "trade"],
        seq_len: int = 32,
        norm: Optional[str] = "rolling_window",
        verbose: int = 1,
    ):
        self.model_name = model_name
        self.policy_name = policy_name

        # validation
        if policy_name not in _POLICY_MAP:
            raise ValueError(f"Unknown policy '{policy_name}'. "
                             f"Choose from: {list(_POLICY_MAP.keys())}")
        if model_name not in _MODEL_MAP:
            raise ValueError(f"Unknown model '{model_name}'. "
                             f"Choose from: {list(_MODEL_MAP.keys())}")
        if model_name not in _POLICY_MAP[policy_name]["supported_models"]:
            raise ValueError(
                f"Policy '{policy_name}' does not support model '{model_name}'. "
                f"Supported models: {list(_POLICY_MAP[policy_name]['supported_models'])}"
            )

        self.log_root = (
            Path(config.LOG_DIR) / 
            mode / 
            self.model_name / 
            self.policy_name
        )

        model_kwargs = dict(_MODEL_MAP[model_name]["kwargs"])  # Copy, don't mutate config

        if _POLICY_MAP[policy_name]["type"] == "custom":
            policy_kwargs = _POLICY_MAP[policy_name]["kwargs"][model_name]
            model_kwargs["policy_kwargs"] = policy_kwargs
            policy = _POLICY_MAP[policy_name]["cls"](model_name)
        else:
            policy = policy_name

        # build environment
        self.env = make_env(
            env=env,
            norm=norm,
            seq_len=seq_len,
            requires_sequence=_POLICY_MAP[policy_name]["requires_sequence"],
        )
        self.verbose = verbose

        # trainable model
        model_cls = _MODEL_MAP[model_name]["cls"]
        self.model = model_cls(
            policy=policy,
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


    @staticmethod
    def _make_unique_path(path: str | Path) -> Path:
        path = Path(path)

        # If file does not exist, return original path
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        i = 1
        while True:
            new_path = parent / f"{stem}_{i}{suffix}"

            if not new_path.exists():
                return new_path

            i += 1


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

        lstm_states = None
        episode_start = np.ones((self.env.num_envs,), dtype=bool)

        while not done:
            action, lstm_states = self.model.predict(
                obs, 
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic)
            
            obs, _, dones, info = self.env.step(action)

            if "portfolio_value" in info[0]:
                logger.log_step(info[0]["portfolio_value"])

            episode_start = dones
            done = dones[0]   # DummyVecEnv returns array

        logger.close()


    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)


    def save(self, path: Optional[str] = None):
        if path is None:
            path = (Path(config.TRAINED_MODEL_DIR) / 
                    f"{self.model_name}_{self.policy_name}.zip")
            path = self._make_unique_path(path)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"Model saved to {str(path)}")


    def _load_pretrained_model(self, path: str):
        model_cls = _MODEL_MAP[self.model_name]["cls"]
        self.model = model_cls.load(path, env=self.env)
