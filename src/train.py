from __future__ import annotations

from agents.reccurent_agent import ReccurentDRLAgent
from utils.download import download_data
from configs import config
from envs.multi_asset import MultiAssetTradingEnv
from configs.config_tickers import DOW_30_TICKER
from utils.setup import make_directories, build_parser

def main() -> int:

    parser = build_parser("train")
    options = parser.parse_args()
    make_directories(
        [config.DATA_SAVE_DIR, 
         config.TRAINED_MODEL_DIR, 
         config.LOG_DIR]
    )

    price_array, tech_array = download_data(
        ticker_list=DOW_30_TICKER,
        tech_indicator_list=config.INDICATORS,
        interval=config.TIME_INTERVAL,
        start_date=config.TRAIN_START_DATE,
        end_date=config.TRAIN_END_DATE,
        data_path=config.DATA_SAVE_DIR / config.TRAIN_DATA_FILE,
        use_vix=True,
    )

    env = MultiAssetTradingEnv(
        price_array=price_array,
        tech_array=tech_array,
        **config.MULTI_ASSET_ENV_PARAMS,
    )

    agent = ReccurentDRLAgent(
        algo=options.algo,
        policy=options.policy,
        env=env,
        seq_len=config.SEQUENCE_LENGTH,
        norm=options.norm,
        verbose=options.verbose,
        tensorboard_log=config.LOG_DIR,
    )

    agent.train(total_timesteps=options.total_timesteps, 
                interval=config.TIME_INTERVAL)
    
    print("Training is finished!")
    agent.save(options.model_save_path)
