from __future__ import annotations

import os
from argparse import ArgumentParser

from finrl.config import ALPACA_API_BASE_URL
from finrl.config import DATA_SAVE_DIR
from finrl.config import ERL_PARAMS
from finrl.config import INDICATORS
from finrl.config import RESULTS_DIR
from finrl.config import TENSORBOARD_LOG_DIR
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config import TRADE_END_DATE
from finrl.config import TRADE_START_DATE
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config import TRAINED_MODEL_DIR
from finrl.config_tickers import SINGLE_TICKER
from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoTradingEnv


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        required=True,
    )
    parser.add_argument(
        "--data_source",
        dest="data_source",
        help="data source, yahoofinance, joinquant, alpaca",
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
        "--drl_lib",
        dest="drl_lib",
        help="drl library, elegantrl, rllib",
        metavar="DRL_LIB",
        default="elegantrl",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="model name, a2c, ppo, ddpg, td3, sac, erl",
        metavar="MODEL_NAME",
        default="ppo",
    )

    return parser


def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def main() -> int:
    parser = build_parser()
    options = parser.parse_args()
    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
    )

    env = CryptoTradingEnv

    if options.mode == "train":
        from finrl import train

        # demo for elegantrl
        kwargs = (
            {}
        )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=SINGLE_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name="ppo",
            cwd="./test_ppo",
            erl_params=ERL_PARAMS,
            break_step=1e5,
            if_vix=True,
            kwargs=kwargs,
        )
    elif options.mode == "test":
        from finrl import test

        # demo for elegantrl
        # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        kwargs = {}

        account_value_erl = test(  # noqa
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=SINGLE_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name="ppo",
            cwd="./test_ppo",
            net_dimension=512,
            kwargs=kwargs,
        )
    elif options.mode == "trade":
        from finrl import trade

        try:
            from finrl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
        except ImportError:
            raise FileNotFoundError(
                "Please set your own ALPACA_API_KEY and ALPACA_API_SECRET in config_private.py"
            )

        kwargs = {}
        trade(
            start_date=TRADE_START_DATE,
            end_date=TRADE_END_DATE,
            ticker_list=SINGLE_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name="ppo",
            API_KEY=ALPACA_API_KEY,
            API_SECRET=ALPACA_API_SECRET,
            API_BASE_URL=ALPACA_API_BASE_URL,
            trade_mode="paper_trading",
            if_vix=True,
            kwargs=kwargs,
            state_dim=len(SINGLE_TICKER) * (len(INDICATORS) + 3) + 3,
            action_dim=len(SINGLE_TICKER),
        )
    else:
        raise ValueError("Wrong mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
