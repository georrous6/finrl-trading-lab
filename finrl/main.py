from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

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
from finrl.config import TRAIN_DATA_FILE
from finrl.config import TEST_DATA_FILE
from finrl.config import TRADE_DATA_FILE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoTradingEnv


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="download, training, testing or trading",
        choices=["download", "train", "test", "trade"],
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
        "--model_name",
        dest="model_name",
        help="model name, a2c, ppo, ddpg, td3, sac, erl",
        metavar="MODEL_NAME",
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
         TENSORBOARD_LOG_DIR, 
         RESULTS_DIR]
    )

    train_data_path = Path(DATA_SAVE_DIR) / TRAIN_DATA_FILE
    test_data_path = Path(DATA_SAVE_DIR) / TEST_DATA_FILE
    trade_data_path = Path(DATA_SAVE_DIR) / TRADE_DATA_FILE

    if options.mode == "download":
        from finrl.download import download_data

        download_data(
            train_start_date=TRAIN_START_DATE,
            train_end_date=TRAIN_END_DATE,
            test_start_date=TEST_START_DATE,
            test_end_date=TEST_END_DATE,
            trade_start_date=TRADE_START_DATE,
            trade_end_date=TRADE_END_DATE,
            ticker_list=DOW_30_TICKER,
            tech_indicator_list=INDICATORS,
            train_output_path=train_data_path,
            test_output_path=test_data_path,
            trade_output_path=trade_data_path,
            use_vix=True,
        )

    elif options.mode == "train":
        from finrl.train import train

        train_data = pd.read_csv(train_data_path)
        train_data = train_data.set_index(train_data.columns[0])
        print(f"Training data length: {len(train_data)}")
        print(train_data.head())
        return 0

        kwargs = {}
        train(
            env=env,
            model_name=options.model_name,
            cwd="./" + str(options.model_name),
            agent_params=ERL_PARAMS,
            total_timesteps=1e6,
            kwargs=kwargs,
        )
    elif options.mode == "test":
        from finrl.test import test

        kwargs = {}

        account_value_erl = test(
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
        from finrl.trade import trade

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
