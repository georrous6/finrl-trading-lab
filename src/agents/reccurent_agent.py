from typing import Optional
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
from policies.transformer_policy import make_transformer_policy
from utils.normalize import RollingWindowNorm
from utils.sequential import VecSequenceWrapper
from stable_baselines3.common.vec_env import VecMonitor
from utils.callbacks import FinancialMetricsCallback
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
        "kwargs": config.TRANSFORMER_POLICY_PARAMS,
    }
}

_ALGO_MAP = {
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
        algo:            Algorithm name -- 'ppo', 'a2c', 'sac', 'td3', 'ddpg'
        policy:          Policy name -- 'TransformerPolicy', 'MlpPolicy', 'LstmMlpPolicy'
        env:             Raw gymnasium env (unwrapped)
        seq_len:         Lookback window for VecSequenceWrapper
        norm:            Normalizer name -- 'rolling_window' or None
        verbose:         SB3 verbosity
        tensorboard_log: TensorBoard log dir
    """

    def __init__(
        self,
        algo: str,
        policy: str,
        env,
        seq_len: int = 32,
        norm: Optional[str] = "rolling_window",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        self.algo_name = algo
        self.policy_name = policy

        # validation
        if policy not in _POLICY_MAP:
            raise ValueError(f"Unknown policy '{policy}'. "
                             f"Choose from: {list(_POLICY_MAP.keys())}")
        if algo not in _ALGO_MAP:
            raise ValueError(f"Unknown algo '{algo}'. "
                             f"Choose from: {list(_ALGO_MAP.keys())}")
        if norm not in _NORM_MAP:
            raise ValueError(f"Unknown norm '{norm}'. "
                             f"Choose from: {list(_NORM_MAP.keys())}")

        norm_kwargs = _NORM_MAP[norm]["kwargs"]
        policy_kwargs = _POLICY_MAP[policy]["kwargs"]
        algo_kwargs = dict(_ALGO_MAP[algo]["kwargs"])  # Copy, don't mutate global config
        algo_kwargs["policy_kwargs"] = policy_kwargs

        # environment pipeline
        norm_cls = _NORM_MAP[norm]["cls"]
        env = norm_cls(env, **norm_kwargs)
        env = DummyVecEnv([lambda e=env: e])
        env = VecSequenceWrapper(env, seq_len=seq_len)
        env = VecMonitor(env)

        # policy
        policy_cls = _POLICY_MAP[policy]["cls"](algo)

        # trainable model
        algo_cls = _ALGO_MAP[algo]["cls"]
        self.model = algo_cls(
            policy=policy_cls,
            env=env,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            **algo_kwargs,
        )


    def train(self, 
              total_timesteps: int, 
              tb_log_name: Optional[str] = None,
              interval: str = "1d",):

        tb_log_name = tb_log_name or f"{self.algo_name}_{self.policy_name}"
        callback = FinancialMetricsCallback(interval=interval)
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=callback
        )
        return self


    def predict(self, obs, deterministic: bool = True):
        return self.model.predict(obs, deterministic=deterministic)


    def save(self, path: str):
        if not path:
            i = 1
            path = (Path(config.TRAINED_MODEL_DIR) / 
                    f"{self.algo_name}_{self.policy_name}_{i}.zip")

            # Keep incrementing counter as long as the file already exists
            while path.exists():
                i += 1
                path = (Path(config.TRAINED_MODEL_DIR) / 
                        f"{self.algo_name}_{self.policy_name}_{i}.zip")
        self.model.save(str(path))
        print(f"Model saved to {path}")


    @classmethod
    def load(cls, algo: str, path: str, env):
        model_cls = _ALGO_MAP[algo]["cls"]
        instance = object.__new__(cls)
        instance.model = model_cls.load(path, env=env)
        return instance
