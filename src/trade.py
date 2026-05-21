from __future__ import annotations

from dotenv import dotenv_values

from meta.paper_trading import AlpacaPaperTrading
from utils.setup import build_parser, make_directories
from configs import config
from configs.config_tickers import DOW_30_TICKER


def main() -> int:

    mode = "trade"
    parser = build_parser(mode)
    options = parser.parse_args()
    make_directories([
        config.RESULTS_DIR
    ])

    # Load environment variables
    env_vars = dotenv_values(".env")
    API_KEY = env_vars.get("ALPACA_API_KEY")
    API_SECRET = env_vars.get("ALPACA_API_SECRET")

    # initialize paper trading env
    paper_trading = AlpacaPaperTrading(
        model_path=options.model_path,
        ticker_list=DOW_30_TICKER,
        tech_indicator_list=config.INDICATORS,
        api_key=API_KEY,
        api_secret=API_SECRET,
        trading_interval=options.trading_interval,
        position_limit=0.2,
        transaction_cost=1e-3,
        min_trade_fraction=0.05,
        timeframe=config.TIME_INTERVAL,
        limit=100,
    )

    paper_trading.run()

if __name__ == "__main__":
    SystemExit(main())
