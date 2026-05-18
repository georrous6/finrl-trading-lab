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
from finrl.envs.multi_asset import MultiAssetTradingEnv
from finrl.policies.transformer_policy import TransformerPolicy
from finrl.download import download_data


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

    if options.mode == "train":
        from finrl.train import train

        price_array, tech_array = download_data(
            ticker_list=DOW_30_TICKER,
            tech_indicator_list=INDICATORS,
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            output_path=train_data_path,
            use_vix=True,
        )

        env = MultiAssetTradingEnv(
            price_array=price_array,
            tech_array=tech_array,
            initial_capital=1e6,
            transaction_cost=1e-3,
            position_limit=0.2,
            volatility_penalty=0.0,
        )

        kwargs = {}
        train(
            env=env,
            model_name="ppo",
            policy="MlpLstmPolicy",
            cwd="./" + str(options.model_name),
            agent_params={},
            total_timesteps=1_000
        )
    elif options.mode == "test":
        from finrl.test import test

        kwargs = {}

        test(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
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
