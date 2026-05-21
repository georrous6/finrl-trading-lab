from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from configs.config import TOTAL_TIMESTEPS


def build_parser(mode: str) -> ArgumentParser:
    parser = ArgumentParser()

    # Add common arguments
    parser.add_argument(
        "--model-name",
        dest="model_name",
        help="model name, a2c, ppo, ddpg, td3, sac, recurrent_ppo",
        metavar="MODEL",
        choices=["a2c", "ppo", "ddpg", "td3", "sac", "recurrent_ppo"],
        default="ppo",
    )
    parser.add_argument(
        "--policy",
        dest="policy",
        help="policy architecture, MlpPolicy, MlpLstmPolicy, TransformerPolicy",
        metavar="POLICY",
        choices=["MlpPolicy", "MlpLstmPolicy", "TransformerPolicy"],
        default="TransformerPolicy",
    )
    parser.add_argument(
        "--norm",
        dest="norm",
        default="rolling_window",       # default when not provided
        choices=["rolling_window"],     # don't include None here
        required=False,
        nargs="?",                      # makes it optional with no value
        const=None,                     # --norm with no value -> None
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        help="verbosity level: 0 (no output), 1 (info), 2 (debug)",
        metavar="VERBOSE",
        type=int,
        choices=[0, 1, 2],
        default=1,
    )
    parser.add_argument(
        "--no-vix",
        dest="no_vix",
        help="exclude VIX data",
        action="store_true",
    )

    if mode == "train":
        parser = _add_train_args(parser)
    elif mode == "backtest":
        parser = _add_test_args(parser)
    elif mode == "trade":
        parser = _add_test_args(parser)
        parser = _add_trade_args(parser)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'train' or 'backtest'.")

    return parser


def _add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--total-timesteps",
        dest="total_timesteps",
        help="total timesteps for training",
        metavar="TOTAL_TIMESTEPS",
        type=int,
        default=TOTAL_TIMESTEPS,
    )
    parser.add_argument(
        "--model-save-path",
        dest="model_save_path",
        help="path to save the trained model",
        metavar="MODEL_SAVE_PATH",
        type=str,
        default=None,  # if None, handled by the agent internally
    )
    return parser


def _add_test_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--model-path",
        dest="model_path",
        help="path to load the trained model from",
        metavar="MODEL_PATH",
        type=str,
        required=True,
    )
    return parser


def _add_trade_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--trading-interval",
        dest="trading_interval",
        help="trading interval for live trading (e.g. '1m', '5m', '1h')",
        metavar="TRADING_INTERVAL",
        type=str,
        default="1m",
        choices=["1m", "5m", "15m", "1h", "1d"],
    )
    return parser


def make_directories(directories: list[str]):
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
