from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from src.configs.config import DATA_SAVE_DIR
from src.configs.config import INDICATORS
from src.configs.config import RESULTS_DIR
from src.configs.config import LOG_DIR
from src.configs.config import TRAIN_END_DATE
from src.configs.config import TRAIN_START_DATE
from src.configs.config import TRAINED_MODEL_DIR
from src.configs.config import TRAIN_DATA_FILE
from src.configs.config import PPO_PARAMS
from src.configs.config_env import MULTI_ASSET_ENV_PARAMS
from src.configs.config_policy import TRANSFORMER_POLICY_PARAMS
from src.configs.config_tickers import DOW_30_TICKER
from src.envs.multi_asset import MultiAssetTradingEnv


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="training, testing or trading",
        choices=["train", "test", "trade"],
        metavar="MODE",
        required=True,
    )
    parser.add_argument(
        "--data_source",
        dest="data_source",
        help="data source, yahoofinance, binance, alpaca",
        metavar="DATA_SOURCE",
        default="yahoofinance",
    )
    parser.add_argument(
        "--time_interval",
        dest="time_interval",
        help="time interval, 1D, 1H, 1M",
        metavar="TIME_INTERVAL",
        default="1D",
    )
    parser.add_argument(
        "--algo_name",
        dest="algo_name",
        help="algorithm name, a2c, ppo, ddpg, td3, sac, erl",
        metavar="ALGO_NAME",
        choices=["a2c", "ppo", "ddpg", "td3", "sac", "erl"],
        default="ppo",
    )

    return parser


def make_directories(directories: list[str]):
    SRC_DIR = Path(__file__).parent
    ROOT_DIR = SRC_DIR.parent
    directories = [ROOT_DIR / directory for directory in directories]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def main() -> int:

    parser = build_parser()
    options = parser.parse_args()
    make_directories(
        [DATA_SAVE_DIR, 
         TRAINED_MODEL_DIR, 
         LOG_DIR, 
         RESULTS_DIR]
    )

    if options.mode == "train":
        from src.modes.train import train

        env = MultiAssetTradingEnv

        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            indicator_list=INDICATORS,
            ticker_list=DOW_30_TICKER,
            data_path=Path(DATA_SAVE_DIR) / TRAIN_DATA_FILE,
            env=env,
            norm="rolling_window",
            algo=options.algo_name,
            policy="transformer",
            env_kwargs=MULTI_ASSET_ENV_PARAMS,
            norm_kwargs={"obs_window": 500},
            policy_kwargs=TRANSFORMER_POLICY_PARAMS,
            algo_kwargs=PPO_PARAMS,
            tensorboard_log=LOG_DIR,
            seq_len=32,
            verbose=1,
            total_timesteps=10_000,
            save_path=Path(TRAINED_MODEL_DIR) / "ppo_transformer.zip",
        )
    else:
        raise ValueError("Wrong mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
