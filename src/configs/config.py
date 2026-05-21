from __future__ import annotations

# ==================================
# Files and directories
# ==================================
TRAIN_DATA_FILE = "train_data.csv"
TEST_DATA_FILE = "test_data.csv"
TRADE_DATA_FILE = "trade_data.csv"

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
LOG_DIR = "logs"
RESULTS_DIR = "results"

# ==================================
# Data parameters
# ==================================
# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2014-01-06"
TRAIN_END_DATE = "2025-12-31"

TEST_START_DATE = "2026-01-01"
TEST_END_DATE = "2026-03-20"

TIME_INTERVAL = "1D"  # "1D", "1H", "5M", "1M"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]


# ==================================
# Training config
# ==================================
TOTAL_TIMESTEPS = 300_000
SEQUENCE_LENGTH = 10


# ==================================
# Algorithm Parameters
# ==================================
A2C_PARAMS = {
    "n_steps": 128,
    "learning_rate": 1e-5,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
    "gamma": 0.99,
}

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 3e-4,
    "batch_size": 64,
    "target_kl": 0.1,
}

DDPG_PARAMS = {
    "batch_size": 256,
    "buffer_size": 20_000,
    "learning_rate": 1e-5,
    "max_grad_norm": 10,
    "learning_starts": 1000,
}

TD3_PARAMS = {
    "batch_size": 256,
    "buffer_size": 20_000,
    "learning_rate": 1e-5,
    "max_grad_norm": 10,
    "learning_starts": 1000,
}

SAC_PARAMS = {
    "batch_size": 256,
    "buffer_size": 20_000,
    "learning_rate": 1e-5,
    "learning_starts": 1000,
    "ent_coef": "auto",
    "max_grad_norm": 10,
}

ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,
}

RECURRENT_PPO_PARAMS = {
    "learning_rate": 3e-5,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "target_kl": 0.02,
    "ent_coef": 0.001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 312,
}


# ==================================
# Normalization parameters
# ==================================
ROLLING_WINDOW_NORM_PARAMS = {
    "obs_window": 500,
    "reward_window": 500,
    "norm_obs": True,
    "norm_reward": False,
}


# ==================================
# Policy configurations
# ==================================

# Separate policy kwargs per algo family
ON_POLICY_TRANSFORMER_PARAMS = {
    "features_extractor_kwargs": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "net_arch": dict(pi=[64, 64], vf=[64, 64]),  # PPO, A2C
}

OFF_POLICY_TRANSFORMER_PARAMS = {
    "features_extractor_kwargs": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "net_arch": dict(pi=[64, 64], qf=[64, 64]),   # DDPG, TD3, SAC
}


# ==================================
# Environment parameters
# ==================================
MULTI_ASSET_ENV_PARAMS = {
    "initial_capital": 1e6,
    "transaction_cost": 1e-3,
    "position_limit": 0.2,
    "volatility_penalty": 0.0,
}


# ==================================
# Possible time zones
# ==================================
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
