from __future__ import annotations

from agents.reccurent_agent import ReccurentDRLAgent
from utils.download import download_data
from configs import config
from envs.multi_asset import MultiAssetTradingEnv
from configs.config_tickers import DOW_30_TICKER
from utils.setup import make_directories, build_parser

from pathlib import Path

def main() -> int:

    parser = build_parser("test")
    options = parser.parse_args()
    dirs = make_directories()

    data_path = str(Path(dirs["DATA_SAVE_DIR"]) / config.TEST_DATA_FILE)
    log_root = str(Path(dirs["LOG_DIR"]) / "test")

    price_array, tech_array = download_data(
        ticker_list=DOW_30_TICKER,
        tech_indicator_list=config.INDICATORS,
        interval=config.TIME_INTERVAL,
        start_date=config.TEST_START_DATE,
        end_date=config.TEST_END_DATE,
        data_path=data_path,
        use_vix=not options.no_vix,
    )

    env = MultiAssetTradingEnv(
        price_array=price_array,
        tech_array=tech_array,
        **config.MULTI_ASSET_ENV_PARAMS,
    )

    agent = ReccurentDRLAgent(
        model_name=options.model_name,
        policy=options.policy,
        env=env,
        log_root=log_root,
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
