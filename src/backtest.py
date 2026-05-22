from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from agents.drl_agent import DRLAgent
from configs import config
from envs.env_stock_trading import StockTradingEnv
from envs.env_crypto_trading import CryptoTradingEnv
from envs.assembly import make_env
from policies.factory import make_policy
from configs.config_tickers import (CRYPTO_TICKERS,
                                    STOCK_TICKERS)
from utils.common import (make_directories,
                          build_parser, 
                          get_data)


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


def main() -> int:

    mode = "backtest"
    parser = build_parser()
    parser = _add_test_args(parser)
    options = parser.parse_args()
    make_directories([
        config.DATA_SAVE_DIR, 
        config.LOG_DIR,
    ])

    data_path = str(Path(config.DATA_SAVE_DIR) / 
                    f"{options.asset_type}_{mode}_data.csv")

    if options.asset_type == "crypto":
        ticker_list = CRYPTO_TICKERS
        env_cls = CryptoTradingEnv
        data_params = config.CRYPTO_DATA_PARAMS
        env_params = config.CRYPTO_TRADING_ENV_PARAMS
    else: # stock
        ticker_list = STOCK_TICKERS
        env_cls = StockTradingEnv
        data_params = config.STOCK_DATA_PARAMS
        env_params = config.STOCK_TRADING_ENV_PARAMS

    data = get_data(
        ticker_list=ticker_list,
        tech_indicator_list=config.INDICATORS,
        interval=config.TIME_INTERVAL,
        start_date=config.TEST_START_DATE,
        end_date=config.TEST_END_DATE,
        data_path=data_path,
        **data_params,
    )

    env = env_cls(**data, **env_params)

    policy, policy_kwargs, requires_sequence = make_policy(
        model_name=options.model_name,
        policy_name=options.policy_name,
        observation_space=env.observation_space,
    )

    env = make_env(
        env=env,
        norm=options.norm,
        seq_len=config.SEQUENCE_LENGTH,
        requires_sequence=requires_sequence,
    )

    agent = DRLAgent(
        model_name=options.model_name,
        policy=policy,
        env=env,
        mode=mode,
        verbose=options.verbose,
        policy_kwargs=policy_kwargs,
    )

    agent.backtest(
        path=options.model_path,
        deterministic=True,
        interval=config.TIME_INTERVAL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
