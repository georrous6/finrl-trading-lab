## FinRL Trading Lab

[![test](https://github.com/georrous6/finrl-trading-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/georrous6/finrl-trading-lab/actions/workflows/ci.yml)

User-friendly research lab for training, backtesting, 
and paper trading DRL agents on multi-asset stock and 
crypto data using Gymnasium + Stable-Baselines3. 
This repo is based on 
[FinRL](https://github.com/AI4Finance-Foundation/FinRL).

## Table of Contents

- [Introduction](#finrl-trading-lab)
- [Project Structure](#project-structure)
- [Valid Model-Policy Combinations](#valid-model-policy-combinations)
- [Setup Instructions](#setup-instructions)
- [Training](#training-instructions)
- [Backtesting](#testing-instructions)
- [Paper Trading](#paper-trading)
- [Genetic Algorithm Training](#genetic-algorithm-training)

## Project Structure

```
finrl-trading-lab/
├─ datasets/
├─ logs/
├─ scripts/
├─ src/
│  ├─ agents/
│  ├─ configs/
│  ├─ data/
│  ├─ envs/
│  ├─ meta/
│  ├─ policies/
│  ├─ utils/
│  ├─ backtest.py
│  ├─ train.py
│  └─ trade.py
├─ trained_models/
├─ environment.yml
├─ requirements.txt
└─ README.md
```

## Valid Model-Policy Combinations

| Policy | ppo | a2c | sac | td3 | ddpg | recurrent_ppo |
|---|---|---|---|---|---|---|
| MlpPolicy | Yes | Yes | Yes | Yes | Yes | No |
| TransformerPolicy | Yes | Yes | Yes | Yes | Yes | No |
| MlpTransformerPolicy | Yes | Yes | Yes | Yes | Yes | No |
| MlpLstmPolicy | No | No | No | No | No | Yes |

## Setup Instructions

Clone the repository first:

```bash
git clone https://github.com/georrous6/finrl-trading-lab.git
cd finrl-trading-lab
```

Conda setup (recommended):

```bash
conda env create -f environment.yml
conda activate finrl_env
```

Alternative (pip-only) setup if you prefer:

```bash
python -m venv finrl_env
source finrl_env/bin/activate
pip install -r requirements.txt
```

## Training

To train a model with a specific policy run:

```bash
PYTHONPATH=src python3 src/train.py \
	--total-timesteps=1000 \
	--model-name=ppo \
	--policy-name=MlpPolicy \
	--model-save-path=trained_models/ppo_MlpPolicy.zip
```

>Note:`--policy-name` and `--model-name` should be a 
valid combination from above.

## Genetic Algorithm Training

Evolve SB3 hyperparameters with a lightweight GA loop that reuses the
existing environment and policy pipeline.

```bash
PYTHONPATH=src python3 src/train.py \
	--trainer=ga \
	--model-name=ppo \
	--policy-name=TransformerPolicy \
	--asset-type=stock \
	--ga-population=8 \
	--ga-generations=5 \
	--ga-elite=0.25 \
	--ga-mutation-rate=0.3 \
	--ga-train-timesteps=10000 \
	--ga-eval-episodes=1 \
	--ga-workers=4
```

Or run the helper script:

```bash
bash scripts/ga_train.sh
```

Smoke test with a tiny GA run:

```bash
PYTHONPATH=src python3 scripts/ga_hparam_smoke.py --model-name=ppo --policy-name=TransformerPolicy
```

## Backtesting

To backtest a trained model run:

```bash
PYTHONPATH=src python3 src/backtest.py \
	--model-path=trained_models/ppo_MlpPolicy.zip \
	--model-name=ppo \
	--policy-name=MlpPolicy
```

## Paper Trading

Use Alpaca paper trading to run a trained model in real time.

Create a `.env` file with your Alpaca keys:

```bash
cp .env.example .env
```

```bash
ALPACA_API_KEY="your_alpaca_api_key"
ALPACA_API_SECRET="your_alpaca_api_secret"
```

Then run the trading script:

```bash
PYTHONPATH=src python3 src/trade.py \
	--model-path=trained_models/ppo_MlpPolicy.zip \
	--model-name=ppo \
	--policy-name=MlpPolicy
```

