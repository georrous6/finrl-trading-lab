from __future__ import annotations

from pathlib import Path
from argparse import ArgumentParser

from agents.drl_agent import DRLAgent
from configs import config
from envs.env_crypto_trading import CryptoTradingEnv
from envs.env_stock_trading import StockTradingEnv
from envs.assembly import make_env
from policies.factory import make_policy
from configs.config_tickers import (CRYPTO_TICKERS, 
                                    STOCK_TICKERS)
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

    mode = "train"
    parser = build_parser()
    parser = _add_train_args(parser)

    options = parser.parse_args()
    make_directories([
        config.DATA_SAVE_DIR, 
        config.TRAINED_MODEL_DIR, 
        config.LOG_DIR,
    ])

    data_path = str(Path(config.DATA_SAVE_DIR) / 
                    f"{options.asset_type}_{mode}_data.csv")

    if options.asset_type == "crypto":
        ticker_list = CRYPTO_TICKERS
        env_cls = CryptoTradingEnv
        data_params = config.CRYPTO_DATA_PARAMS
        env_params = config.CRYPTO_TRADING_ENV_PARAMS
    else:  # stock
        ticker_list = STOCK_TICKERS
        env_cls = StockTradingEnv
        data_params = config.STOCK_DATA_PARAMS
        env_params = config.STOCK_TRADING_ENV_PARAMS


    data = get_data(
        ticker_list=ticker_list,
        tech_indicator_list=config.INDICATORS,
        interval=config.TIME_INTERVAL,
        start_date=config.TRAIN_START_DATE,
        end_date=config.TRAIN_END_DATE,
        data_path=data_path,
        **data_params,
    )

    env = env_cls(**data, **env_params)

    policy, policy_kwargs, requires_sequence = make_policy(
        model_name=options.model_name,
        policy_name=options.policy_name,
    )

    env = make_env(env=env, 
                   norm=options.norm, 
                   seq_len=config.SEQUENCE_LENGTH,
                   requires_sequence=requires_sequence)

    agent = DRLAgent(
        model_name=options.model_name,
        policy=policy,
        env=env,
        mode=mode,
        verbose=options.verbose,
        policy_kwargs=policy_kwargs,
    )

    agent.train(total_timesteps=options.total_timesteps, 
                interval=config.TIME_INTERVAL)
    
    print("\nTraining finished")
    agent.save(options.model_save_path)


if __name__ == "__main__":
    raise SystemExit(main())
