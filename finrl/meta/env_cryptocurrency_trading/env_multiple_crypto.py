from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiAssetTradingEnv(gym.Env):
    """
    Clean multi-asset trading environment.

    Price: (T, N)
    Tech:  (T, N, F)
    """

    def __init__(
        self,
        price_array,
        tech_array,
        lookback=10,
        initial_capital=1e6,
        transaction_cost=1e-3,
        position_limit=0.1,
        volatility_penalty=0.0,
    ):
        super().__init__()

        assert len(price_array.shape) == 2, "price_array must be (T, N)"
        assert len(tech_array.shape) == 3, "tech_array must be (T, N, F)"
        assert price_array.shape[0] == tech_array.shape[0], "Time dimension mismatch"

        self.price = price_array
        self.tech = tech_array

        self.T, self.N = price_array.shape
        _, _, self.F = tech_array.shape

        self.lookback = lookback
        self.initial_capital = initial_capital
        self.tc = transaction_cost
        self.position_limit = position_limit
        self.vol_penalty = volatility_penalty

        self.max_step = self.T - 1

        # portfolio state
        self.cash = initial_capital
        self.shares = np.zeros(self.N, dtype=np.float32)
        self.time = lookback

        # ======================
        # Action space
        # ======================
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.N,),
            dtype=np.float32,
        )

        # ======================
        # Observation space
        # ======================
        obs_dim = self.N * (1 + self.F + 1) + 1  # price + tech + weights + cash
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback, obs_dim),
            dtype=np.float32,
        )

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.cash = self.initial_capital
        self.shares = np.zeros(self.N, dtype=np.float32)
        self.time = self.lookback

        return self._get_obs(), {}

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):

        self.time += 1

        price = self.price[self.time]

        # portfolio value BEFORE trade
        prev_asset = self._portfolio_value(price)

        # normalize action → target allocation [0,1]
        action = np.clip(action, -1, 1)
        target_weights = (action + 1) / 2
        target_weights = target_weights / (np.sum(target_weights) + 1e-8)

        target_value = target_weights * prev_asset
        current_value = self.shares * price

        diff = target_value - current_value

        # ======================
        # execute trades
        # ======================
        for i in range(self.N):
            if price[i] <= 0:
                continue

            # BUY
            if diff[i] > 0:
                buy_shares = diff[i] / price[i]

                cost = buy_shares * price[i] * (1 + self.tc)

                if cost <= self.cash:
                    self.shares[i] += buy_shares
                    self.cash -= cost
                else:
                    buy_shares = self.cash / (price[i] * (1 + self.tc))
                    self.shares[i] += buy_shares
                    self.cash -= buy_shares * price[i] * (1 + self.tc)

            # SELL
            else:
                sell_shares = min(self.shares[i], -diff[i] / price[i])

                self.shares[i] -= sell_shares
                self.cash += sell_shares * price[i] * (1 - self.tc)

        # next portfolio value
        next_asset = self._portfolio_value(self.price[self.time])

        reward = self._reward(prev_asset, next_asset)

        done = self.time >= self.max_step

        info = {
            "portfolio_value": next_asset,
            "cash": self.cash,
        }

        return self._get_obs(), reward, done, False, info

    # =========================================================
    # OBSERVATION
    # =========================================================
    def _get_obs(self):

        t0 = self.time - self.lookback
        price_window = self.price[t0:self.time]        # (L, N)
        tech_window = self.tech[t0:self.time]          # (L, N, F)

        # log returns
        price_window = np.maximum(price_window, 1e-8)
        log_ret = np.log(price_window[1:] / price_window[:-1])
        log_ret = np.vstack([log_ret, np.zeros((1, self.N))])

        # normalize tech per asset
        mean = tech_window.mean(axis=0)
        std = np.maximum(tech_window.std(axis=0), 1e-8)
        tech_norm = (tech_window - mean) / std

        # portfolio state
        price_now = self.price[self.time]
        total_asset = self._portfolio_value(price_now)

        weights = (self.shares * price_now) / (total_asset + 1e-8)
        cash_ratio = np.array([self.cash / (total_asset + 1e-8)])

        # repeat portfolio info per timestep
        cash_feat = np.repeat(cash_ratio, self.N)[None, :]
        weight_feat = np.repeat(weights[None, :], self.lookback, axis=0)

        tech_flat = tech_norm.reshape(self.lookback, self.N * self.F)

        obs = np.concatenate(
            [
                cash_feat,
                weight_feat,
                log_ret,
                tech_flat,
            ],
            axis=1,
        )

        return obs.astype(np.float32)

    # =========================================================
    # REWARD (clean + stable)
    # =========================================================
    def _reward(self, prev_asset, next_asset):

        log_return = np.log((next_asset + 1e-8) / (prev_asset + 1e-8))

        return log_return

    # =========================================================
    # PORTFOLIO VALUE
    # =========================================================
    def _portfolio_value(self, price):
        return self.cash + np.sum(self.shares * price)

    # =========================================================
    # OPTIONAL
    # =========================================================
    def close(self):
        pass