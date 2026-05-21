from __future__ import annotations

from dotenv import dotenv_values
from argparse import ArgumentParser

from meta.paper_trading import StockPaperTrading
from utils.common import build_parser, make_directories
from configs import config
from configs.config_tickers import DOW_30_TICKER


def _add_trade_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--model-path",
        dest="model_path",
        help="path to load the trained model from",
        metavar="MODEL_PATH",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--trading-interval",
        dest="trading_interval",
        help="trading interval for paper trading, e.g., 1m, 5m, 1h",
        metavar="TRADING_INTERVAL",
        type=str,
        default="1m",
    )
    return parser


def main() -> int:

    parser = build_parser()
    parser = _add_trade_args(parser)
    options = parser.parse_args()
    make_directories([
        config.RESULTS_DIR
    ])

    # Load environment variables
    env_vars = dotenv_values(".env")
    API_KEY = env_vars.get("ALPACA_API_KEY")
    API_SECRET = env_vars.get("ALPACA_API_SECRET")

    # initialize paper trading env
    paper_trading = StockPaperTrading(
        model_path=options.model_path,
        ticker_list=DOW_30_TICKER,
        tech_indicator_list=config.INDICATORS,
        api_key=API_KEY,
        api_secret=API_SECRET,
        trading_interval=options.trading_interval,
        max_stocks=config.STOCK_TRADING_ENV_PARAMS["max_stocks"],
        transaction_cost=config.STOCK_TRADING_ENV_PARAMS["transaction_cost"],
        min_trade_fraction=0.05,
        timeframe=config.TIME_INTERVAL,
        limit=100,
        use_vix=not options.no_vix,
        use_turbulence=not options.no_turbulence,
    )

    paper_trading.run()

if __name__ == "__main__":
    SystemExit(main())
