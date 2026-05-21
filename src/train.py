from __future__ import annotations

from pathlib import Path
from argparse import ArgumentParser

from agents.drl_agent import DRLAgent
from configs import config
from envs.crypto_trading_env import CryptoTradingEnv
from configs.config_tickers import DOW_30_TICKER
from utils.common import (make_directories, 
                          build_parser, 
                          get_data)


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


def main() -> int:

    parser = build_parser()
    parser = _add_train_args(parser)

    options = parser.parse_args()
    make_directories([
        config.DATA_SAVE_DIR, 
        config.TRAINED_MODEL_DIR, 
        config.LOG_DIR,
    ])

    data_path = str(Path(config.DATA_SAVE_DIR) / config.TRAIN_DATA_FILE)

    data = get_data(
        ticker_list=DOW_30_TICKER,
        tech_indicator_list=config.INDICATORS,
        interval=config.TIME_INTERVAL,
        start_date=config.TRAIN_START_DATE,
        end_date=config.TRAIN_END_DATE,
        data_path=data_path,
        use_vix=not options.no_vix,
        use_turbulence=not options.no_turbulence,
    )

    env = CryptoTradingEnv(
        price_array=data['price_array'],
        tech_array=data['tech_array'],
        **config.CRYPTO_TRADING_ENV_PARAMS,
    )

    agent = DRLAgent(
        model_name=options.model_name,
        policy_name=options.policy,
        env=env,
        mode="train",
        seq_len=config.SEQUENCE_LENGTH,
        norm=options.norm,
        verbose=options.verbose,
    )

    agent.train(total_timesteps=options.total_timesteps, 
                interval=config.TIME_INTERVAL)
    
    print("\nTraining finished")
    agent.save(options.model_save_path)


if __name__ == "__main__":
    raise SystemExit(main())
