from __future__ import annotations

from pathlib import Path
from argparse import ArgumentParser

from agents.drl_agent import DRLAgent
from configs import config
from envs.env_crypto_trading import CryptoTradingEnv
from envs.env_stock_trading import StockTradingEnv
from envs.env_stock_trading_fuzzy import StockTradingFuzzyEnv
from envs.assembly import make_env
from policies.factory import make_policy
from configs.config_tickers import (CRYPTO_TICKERS, 
                                    STOCK_TICKERS)
from utils.common import (make_directories, 
                          build_parser, 
                          get_data)


def _add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--total-timesteps",
        dest="total_timesteps",
        help="total timesteps for training",
        metavar="TOTAL_TIMESTEPS",
        type=int,
        default=config.TOTAL_TIMESTEPS,
    )
    parser.add_argument(
        "--model-save-path",
        dest="model_save_path",
        help="path to save the trained model",
        metavar="MODEL_SAVE_PATH",
        type=str,
        default=None,  # if None, handled by the agent internally
    )
    parser.add_argument(
        "--trainer",
        dest="trainer",
        choices=["sb3", "ga"],
        default="sb3",
        help="training strategy: sb3 (default) or ga",
    )
    parser.add_argument("--ga-population", type=int, default=8)
    parser.add_argument("--ga-generations", type=int, default=5)
    parser.add_argument("--ga-elite", type=float, default=0.25)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.3)
    parser.add_argument("--ga-train-timesteps", type=int, default=10_000)
    parser.add_argument("--ga-eval-episodes", type=int, default=1)
    parser.add_argument("--ga-workers", type=int, default=1)
    return parser


def _resolve_model_save_path(model_name: str, policy_name: str, path: str | None) -> str:
    if path is None:
        path = Path(config.TRAINED_MODEL_DIR) / f"{model_name}_{policy_name}_ga.zip"
        path = DRLAgent._make_unique_path(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def main() -> int:

    mode = "train"
    parser = build_parser()
    parser = _add_train_args(parser)

    options = parser.parse_args()
    make_directories([
        config.DATA_SAVE_DIR, 
        config.TRAINED_MODEL_DIR, 
        config.LOG_DIR,
    ])

    data_path = str(Path(config.DATA_SAVE_DIR) / 
                    f"{options.asset_type}_{mode}_data.csv")

    if options.asset_type == "crypto":
        ticker_list = CRYPTO_TICKERS
        env_cls = CryptoTradingEnv
        data_params = config.CRYPTO_DATA_PARAMS
        env_params = config.CRYPTO_TRADING_ENV_PARAMS
    else:  # stock
        ticker_list = STOCK_TICKERS
        if options.use_fuzzy:
            env_cls = StockTradingFuzzyEnv
        else:
            env_cls = StockTradingEnv
        data_params = config.STOCK_DATA_PARAMS
        env_params = config.STOCK_TRADING_ENV_PARAMS


    data = get_data(
        ticker_list=ticker_list,
        tech_indicator_list=config.INDICATORS,
        interval=config.TIME_INTERVAL,
        start_date=config.TRAIN_START_DATE,
        end_date=config.TRAIN_END_DATE,
        data_path=data_path,
        **data_params,
    )

    if options.trainer == "ga":
        from agents.ga_trainer import GAConfig, GABuildSpec, run_ga, build_model_for_params

        ga_config = GAConfig(
            population_size=options.ga_population,
            generations=options.ga_generations,
            elite_fraction=options.ga_elite,
            mutation_rate=options.ga_mutation_rate,
            train_timesteps=options.ga_train_timesteps,
            eval_episodes=options.ga_eval_episodes,
            workers=options.ga_workers,
        )

        build_spec = GABuildSpec(
            model_name=options.model_name,
            policy_name=options.policy_name,
            env_cls=env_cls,
            env_params=env_params,
            data=data,
            norm=options.norm,
            seq_len=config.SEQUENCE_LENGTH,
            verbose=options.verbose,
        )

        best_score, best_params = run_ga(None, None, ga_config, build_spec=build_spec)
        print(f"\nGA finished. Best score: {best_score:.4f}")
        print(f"Best hyperparameters: {best_params}")

        final_model, final_env = build_model_for_params(build_spec, best_params)
        final_model.learn(total_timesteps=options.total_timesteps)

        save_path = _resolve_model_save_path(
            model_name=options.model_name,
            policy_name=options.policy_name,
            path=options.model_save_path,
        )
        final_model.save(save_path)
        final_env.close()
        print(f"Model saved to {save_path}")
        return 0

    env = env_cls(**data, **env_params)

    policy, policy_kwargs, requires_sequence = make_policy(
        model_name=options.model_name,
        policy_name=options.policy_name,
        observation_space=env.observation_space,
    )

    env = make_env(env=env,
                   norm=options.norm,
                   seq_len=config.SEQUENCE_LENGTH,
                   requires_sequence=requires_sequence)

    agent = DRLAgent(
        model_name=options.model_name,
        policy=policy,
        env=env,
        mode=mode,
        verbose=options.verbose,
        policy_kwargs=policy_kwargs,
    )

    agent.train(total_timesteps=options.total_timesteps,
                interval=config.TIME_INTERVAL)

    print("\nTraining finished")
    agent.save(options.model_save_path)


if __name__ == "__main__":
    raise SystemExit(main())
