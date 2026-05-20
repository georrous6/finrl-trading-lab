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

TRADE_START_DATE = "2026-01-01"
TRADE_END_DATE = "2026-03-20"

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
TOTAL_TIMESTEPS = 100_000
SEQUENCE_LENGTH = 10

# ==================================
# Algorithm Parameters
# ==================================
A2C_PARAMS = {
    "n_steps": 5, 
    "ent_coef": 0.01, 
    "learning_rate": 0.0007
}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 1e-5,
    "batch_size": 64,
    "target_kl": 0.02,
}
DDPG_PARAMS = {
    "batch_size": 128, 
    "buffer_size": 50000, 
    "learning_rate": 0.001
}
TD3_PARAMS = {
    "batch_size": 100, 
    "buffer_size": 1000000, 
    "learning_rate": 0.001
}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
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
TRANSFORMER_POLICY_PARAMS = {
    "features_extractor_kwargs": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "net_arch": dict(pi=[64, 64], vf=[64, 64]),
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
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)


# ==================================
# Parameters for data sources
# ==================================
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url
