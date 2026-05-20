from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from configs import config


def build_parser(mode: str) -> ArgumentParser:
    parser = ArgumentParser()

    # Add common arguments
    parser.add_argument(
        "--model-name",
        dest="model_name",
        help="model name, a2c, ppo, ddpg, td3, sac, erl",
        metavar="MODEL",
        choices=["a2c", "ppo", "ddpg", "td3", "sac", "erl"],
        default="ppo",
    )
    parser.add_argument(
        "--policy",
        dest="policy",
        help="policy architecture, mlp, lstm, gru",
        metavar="POLICY",
        choices=["TransformerPolicy", "MlpPolicy", "LstmMlpPolicy"],
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
        return _add_train_args(parser)
    elif mode == "test":
        return _add_test_args(parser)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'train' or 'test'.")


def _add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--total-timesteps",
        dest="total_timesteps",
        help="total timesteps for training",
        metavar="TOTAL_TIMESTEPS",
        type=int,
        default=config.TOTAL_TIMESTEPS,
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


def make_directories() -> dict[str, str]:
    CURRENT_DIR = Path(__file__).parent
    SRC_DIR = CURRENT_DIR.parent
    ROOT_DIR = SRC_DIR.parent
    
    directories = {}
    directories["DATA_SAVE_DIR"] = ROOT_DIR / config.DATA_SAVE_DIR
    directories["TRAINED_MODEL_DIR"] = ROOT_DIR / config.TRAINED_MODEL_DIR
    directories["LOG_DIR"] = ROOT_DIR / config.LOG_DIR
    directories["RESULTS_DIR"] = ROOT_DIR / config.RESULTS_DIR

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return {key: str(path) for key, path in directories.items()}
