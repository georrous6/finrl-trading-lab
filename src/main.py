from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from src.configs import config
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
        [config.DATA_SAVE_DIR, 
         config.TRAINED_MODEL_DIR, 
         config.LOG_DIR, 
         config.RESULTS_DIR]
    )

    if options.mode == "train":
        from src.modes.train import train

        env = MultiAssetTradingEnv

        train(
            start_date=config.TRAIN_START_DATE,
            end_date=config.TRAIN_END_DATE,
            indicator_list=config.INDICATORS,
            ticker_list=DOW_30_TICKER,
            interval=config.TIME_INTERVAL,
            data_path=Path(config.DATA_SAVE_DIR) / config.TRAIN_DATA_FILE,
            env=env,
            norm="rolling_window",
            algo=options.algo_name,
            policy="transformer",
            env_kwargs=config.MULTI_ASSET_ENV_PARAMS,
            norm_kwargs=config.ROLLING_WINDOW_NORM_PARAMS,
            policy_kwargs=config.TRANSFORMER_POLICY_PARAMS,
            algo_kwargs=config.PPO_PARAMS,
            tensorboard_log=config.LOG_DIR,
            seq_len=config.SEQUENCE_LENGTH,
            verbose=1,
            total_timesteps=config.TOTAL_TIMESTEPS,
            save_path=Path(config.TRAINED_MODEL_DIR) / "ppo_transformer.zip",
        )
    else:
        raise ValueError("Wrong mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
