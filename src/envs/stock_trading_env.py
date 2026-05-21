from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


class StockTradingEnv(gym.Env):
    """
    Stock Trading Environment for OpenAI gymnasium

    Price: (T, N)
    Tech:  (T, N, F)
    Vix:   (T,) or None
    Turb:  (T,) or None
    """

    def __init__(
        self,
        price_array: np.ndarray,
        tech_array: np.ndarray,
        vix_array: Optional[np.ndarray] = None,
        turbulence_array: Optional[np.ndarray] = None,
        max_stocks: np.ndarray | int = 1000,
        initial_capital: float = 1e6,
        transaction_cost: float = 1e-3,
        volatility_penalty: float = 0.0,
        turbulence_threshold: Optional[float] = None,
        eps: float = 1e-8,
    ):
        super().__init__()

        assert len(price_array.shape) == 2, "price_array must be (T, N)"
        assert len(tech_array.shape) == 3, "tech_array must be (T, N, F)"
        assert price_array.shape[0] == tech_array.shape[0], "Time dimension mismatch"
        assert price_array.shape[1] == tech_array.shape[1], "Asset dimension mismatch"

        self.price = price_array
        self.tech = tech_array
        self.eps = eps

        self.T, self.N = price_array.shape
        _, _, self.F = tech_array.shape

        if vix_array is not None:
            assert len(vix_array.shape) == 1, "vix_array must be (T,)"
            assert vix_array.shape[0] == self.T, "VIX time dimension mismatch"
            self.vix = vix_array
        else:
            self.vix = np.zeros(self.T, dtype=np.float32)

        if turbulence_array is not None:
            assert len(turbulence_array.shape) == 1, "turbulence_array must be (T,)"
            assert turbulence_array.shape[0] == self.T, "Turbulence time dimension mismatch"
            self.turbulence = turbulence_array
        else:
            self.turbulence = np.zeros(self.T, dtype=np.float32)

        if isinstance(max_stocks, int):
            self.max_stocks = np.full(self.N, max_stocks, dtype=np.int32)
        else:
            assert len(max_stocks) == self.N, "max_stocks must have length N"
            self.max_stocks = np.array(max_stocks, dtype=np.int32)

        self.initial_capital = initial_capital
        self.tc = transaction_cost
        self.vol_penalty = volatility_penalty
        self.turbulence_threshold = turbulence_threshold

        self.max_step = self.T - 1

        # portfolio state
        self.cash = initial_capital
        self.shares = np.zeros(self.N, dtype=np.int32)
        self.time = 0

        # Action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.N,),
            dtype=np.float32,
        )

        # Observation space
        obs_dim = 3 + self.N * (2 + self.F)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.cash = self.initial_capital
        self.shares = np.zeros(self.N, dtype=np.int32)
        self.time = 0

        return self._get_obs(), {}


    def step(self, action):

        action = np.clip(action, -1.0, 1.0)

        if (self.turbulence_threshold is not None and
            self.turbulence[self.time] > self.turbulence_threshold):
            action = np.minimum(action, 0)  # Force sell in high turbulence

        price = self.price[self.time]
        current_asset = self._portfolio_value(price)

        valid_mask = price > 0
        sell_mask = (action < 0) & valid_mask
        buy_mask = (action > 0) & valid_mask
        sell_idx = np.where(sell_mask)[0]
        buy_idx = np.where(buy_mask)[0]

        # First sell to free up cash
        for i in sell_idx:
            self._sell_asset(i, action)

        # Then buy
        for i in buy_idx:
            self._buy_asset(i, action)

        # Market moves
        self.time += 1
        next_price = self.price[self.time]
        next_asset = self._portfolio_value(next_price)

        reward = self._reward(current_asset, next_asset)
        terminated  = self.time == self.max_step

        info = {
            "portfolio_value": next_asset,
            "cash": self.cash,
            "shares": self.shares.copy(),
        }

        return self._get_obs(), reward, terminated, False, info


    def _sell_asset(self, index, action):

        price = self.price[self.time]
        sell_shares = min(
            self.shares[index], 
            int(-action[index] * self.shares[index])
        )
        self.shares[index] -= sell_shares
        self.cash += sell_shares * price[index] * (1 - self.tc)


    def _buy_asset(self, index, action):

        price = self.price[self.time]
        total_asset = self._portfolio_value(price)

        desired_shares = int(action[index] * total_asset / price[index])
        max_shares = self.max_stocks[index] - self.shares[index]
        buy_shares = min(
            desired_shares,
            self.cash // (price[index] * (1 + self.tc)),
            max_shares
        )

        self.shares[index] += buy_shares
        self.cash -= buy_shares * price[index] * (1 + self.tc)


    def _get_obs(self):

        price = self.price[self.time]
        tech = self.tech[self.time]

        obs = np.concatenate([
            [self.cash],
            self.shares.copy(),
            price.copy(),
            tech.flatten().copy(),
            [self.vix[self.time]],
            [self.turbulence[self.time]],
        ]).astype(np.float32)

        return obs


    def _reward(self, prev_asset, next_asset):

        exposure = (
            np.sum(self.shares * self.price[self.time]) / 
            (prev_asset + self.eps)
        )
        vol_penalty = (
            self.vol_penalty * 
            exposure * 
            self.vix[self.time]
        )
        return np.log((next_asset + self.eps) / (prev_asset + self.eps)) - vol_penalty


    def _portfolio_value(self, price):
        return self.cash + np.sum(self.shares * price)


    def close(self):
        pass