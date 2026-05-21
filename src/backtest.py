from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from agents.drl_agent import DRLAgent
from configs import config
from envs.stock_trading_env import StockTradingEnv
from configs.config_tickers import DOW_30_TICKER
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

    parser = build_parser()
    parser = _add_test_args(parser)
    options = parser.parse_args()
    make_directories([
        config.DATA_SAVE_DIR, 
        config.LOG_DIR,
    ])

    data_path = str(Path(config.DATA_SAVE_DIR) / config.TEST_DATA_FILE)

    data = get_data(
        ticker_list=DOW_30_TICKER,
        tech_indicator_list=config.INDICATORS,
        interval=config.TIME_INTERVAL,
        start_date=config.TEST_START_DATE,
        end_date=config.TEST_END_DATE,
        data_path=data_path,
        use_vix=not options.no_vix,
        use_turbulence=not options.no_turbulence,
    )

    env = StockTradingEnv(
        price_array=data['price_array'],
        tech_array=data['tech_array'],
        vix_array=data.get('vix_array'),
        turbulence_array=data.get('turbulence_array'),
        **config.STOCK_TRADING_ENV_PARAMS,
    )

    agent = DRLAgent(
        model_name=options.model_name,
        policy_name=options.policy,
        env=env,
        mode="backtest",
        seq_len=config.SEQUENCE_LENGTH,
        norm=options.norm,
        verbose=options.verbose,
    )

    agent.backtest(
        path=options.model_path,
        deterministic=True,
        interval=config.TIME_INTERVAL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
