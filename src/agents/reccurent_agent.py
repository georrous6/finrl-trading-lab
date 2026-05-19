from typing import Dict, Optional, Any
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
from policies.transformer_policy import make_transformer_policy
from utils.normalize import RollingWindowNorm
from utils.sequential import VecSequenceWrapper
from stable_baselines3.common.vec_env import VecMonitor
from utils.callbacks import FinancialMetricsCallback


_NORM_MAP = {
    "rolling_window": RollingWindowNorm,
}

_POLICY_MAP = {
    "transformer": make_transformer_policy,
}

_ALGO_MAP = {
    "ppo":  PPO,
    "a2c":  A2C,
    "td3":  TD3,
    "ddpg": DDPG,
    "sac":  SAC,
}


class ReccurentDRLAgent:
    """
    Sequential DRL agent.

    Args:
        algo:            Algorithm name -- 'ppo', 'a2c', 'sac', 'td3', 'ddpg'
        policy:          Policy name -- 'transformer'
        env:             Raw gymnasium env (unwrapped)
        seq_len:         Lookback window for VecSequenceWrapper
        norm:            Normalizer name -- 'rolling_window'
        norm_kwargs:     kwargs forwarded to the normalizer  e.g. obs_window=500
        policy_kwargs:   kwargs forwarded to the policy      e.g. d_model=256
        algo_kwargs:     kwargs forwarded to the algorithm   e.g. n_steps=2048
        verbose:         SB3 verbosity
        tensorboard_log: TensorBoard log dir
    """

    def __init__(
        self,
        algo: str,
        policy: str,
        env,
        seq_len: int = 32,
        norm: str = "rolling_window",
        norm_kwargs:   Optional[Dict[str, Any]] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        algo_kwargs:   Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        self.algo_name = algo
        self.policy_name = policy

        # validation
        if policy not in _POLICY_MAP:
            raise ValueError(f"Unknown policy '{policy}'. "
                             f"Choose from: {list(_POLICY_MAP)}")
        if algo not in _ALGO_MAP:
            raise ValueError(f"Unknown algo '{algo}'. "
                             f"Choose from: {list(_ALGO_MAP)}")
        if norm not in _NORM_MAP:
            raise ValueError(f"Unknown norm '{norm}'. "
                             f"Choose from: {list(_NORM_MAP)}")

        norm_kwargs   = norm_kwargs   or {}
        policy_kwargs = policy_kwargs or {}
        algo_kwargs   = algo_kwargs   or {}

        # environment pipeline
        env = _NORM_MAP[norm](env, **norm_kwargs)
        env = DummyVecEnv([lambda e=env: e])
        env = VecMonitor(env)
        env = VecSequenceWrapper(env, seq_len=seq_len)

        # policy
        policy_cls = _POLICY_MAP[policy](algo)

        if policy_kwargs:
            if algo_kwargs.get("policy_kwargs"):
                raise ValueError(
                    "Pass policy_kwargs either as a top-level argument or inside "
                    "algo_kwargs, not both."
                )
            algo_kwargs["policy_kwargs"] = policy_kwargs

        # trainable model
        self.model = _ALGO_MAP[algo](
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
        self.model.save(path)


    @classmethod
    def load(cls, algo: str, path: str, env):
        model_cls = _ALGO_MAP[algo]
        instance = object.__new__(cls)
        instance.model = model_cls.load(path, env=env)
        return instance
