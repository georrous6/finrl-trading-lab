from __future__ import annotations

from agents.drl_agent import DRLAgent
from utils.download import download_data
from configs import config
from envs.multi_asset import MultiAssetTradingEnv
from configs.config_tickers import DOW_30_TICKER
from utils.setup import make_directories, build_parser

from pathlib import Path


def main() -> int:

    mode = "train"
    parser = build_parser(mode)
    options = parser.parse_args()
    make_directories([
        config.DATA_SAVE_DIR, 
        config.TRAINED_MODEL_DIR, 
        config.LOG_DIR,
    ])

    data_path = str(Path(config.DATA_SAVE_DIR) / config.TRAIN_DATA_FILE)

    price_array, tech_array = download_data(
        ticker_list=DOW_30_TICKER,
        tech_indicator_list=config.INDICATORS,
        interval=config.TIME_INTERVAL,
        start_date=config.TRAIN_START_DATE,
        end_date=config.TRAIN_END_DATE,
        data_path=data_path,
        use_vix=not options.no_vix,
    )

    env = MultiAssetTradingEnv(
        price_array=price_array,
        tech_array=tech_array,
        **config.MULTI_ASSET_ENV_PARAMS,
    )

    agent = DRLAgent(
        model_name=options.model_name,
        policy_name=options.policy,
        env=env,
        mode=mode,
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
