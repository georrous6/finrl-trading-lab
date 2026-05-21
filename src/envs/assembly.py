import gymnasium as gym
from stable_baselines3.common.vec_env import (DummyVecEnv, 
                                              VecMonitor,
                                              VecEnv)

from envs.wrappers.vec_sequence_wrapper import VecSequenceWrapper
from envs.wrappers.rolling_window_norm import RollingWindowNorm
from configs import config


_NORM_MAP = {
    "rolling_window": {
        "cls": RollingWindowNorm,
        "kwargs": config.ROLLING_WINDOW_NORM_PARAMS,
    }
}

def make_env(env: gym.Env, 
             norm: str | None, 
             seq_len: int, 
             requires_sequence: bool) -> VecEnv:
    """
    Build a vectorized SB3 environment pipeline.

    Pipeline:
        gym.Env
        -> optional normalization
        -> DummyVecEnv
        -> optional sequence wrapper
        -> VecMonitor

    Args:
        env:
            Raw Gymnasium environment.

        norm:
            Optional normalization strategy name.

        seq_len:
            Sequence length for sequential wrappers.

        requires_sequence:
            Whether to apply sequential observation wrapping.

    Returns:
        Wrapped vectorized environment compatible with SB3.
    """

    # normalization
    if norm is not None:

        if norm not in _NORM_MAP:
            raise ValueError(f"Unknown norm '{norm}'. "
                             f"Choose from: {list(_NORM_MAP.keys())}")

        norm_cls = _NORM_MAP[norm]["cls"]
        norm_kwargs = _NORM_MAP[norm]["kwargs"]
        env = norm_cls(env, **norm_kwargs)

    # vectorization
    env = DummyVecEnv([lambda e=env: e])

    # sequential observations
    if requires_sequence:
        env = VecSequenceWrapper(env, seq_len=seq_len)

    # monitoring
    return VecMonitor(env)
