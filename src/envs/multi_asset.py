from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiAssetTradingEnv(gym.Env):
    """
    Multi-asset trading environment.
    Suitable for stock or cryptocurrency trading.

    Price: (T, N)
    Tech:  (T, N, F)
    """

    def __init__(
        self,
        price_array: np.ndarray,
        tech_array: np.ndarray,
        initial_capital: float = 1e6,
        transaction_cost: float = 1e-3,
        position_limit: list[float] | float | np.ndarray = 0.2,
        volatility_penalty: float = 0.0,
    ):
        super().__init__()

        assert len(price_array.shape) == 2, "price_array must be (T, N)"
        assert len(tech_array.shape) == 3, "tech_array must be (T, N, F)"
        assert price_array.shape[0] == tech_array.shape[0], "Time dimension mismatch"
        assert price_array.shape[1] == tech_array.shape[1], "Asset dimension mismatch"

        self.price = price_array
        self.tech = tech_array

        self.T, self.N = price_array.shape
        _, _, self.F = tech_array.shape

        if isinstance(position_limit, float):
            # same limit for all assets
            self.position_limit = np.full(self.N, position_limit, dtype=np.float32)
        else:
            assert len(position_limit) == self.N, "position_limit list must match number of assets"
            self.position_limit = np.array(position_limit, dtype=np.float32)

        # Normalize to add up to 1.0
        self.position_limit /= self.position_limit.sum()

        self.initial_capital = initial_capital
        self.tc = transaction_cost
        self.vol_penalty = volatility_penalty

        self.max_step = self.T - 1

        # portfolio state
        self.cash = initial_capital
        self.shares = np.zeros(self.N, dtype=np.float32)
        self.time = 0

        # Action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.N,),
            dtype=np.float32,
        )

        # Observation space
        obs_dim = 1 + self.N * (1 + 1 + self.F)  # cash + shares + prices + indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.cash = self.initial_capital
        self.shares = np.zeros(self.N, dtype=np.float32)
        self.time = 0

        return self._get_obs(), {}


    def step(self, action):

        # action: (N,) in [-1, 1], negative means sell, positive means buy

        price = self.price[self.time]
        current_asset = self._portfolio_value(price)

        valid_mask = price > 0
        sell_mask = (action < 0) & valid_mask
        buy_mask = (action > 0) & valid_mask
        sell_idx = np.where(sell_mask)[0]
        buy_idx = np.where(buy_mask)[0]

        for i in sell_idx:
            self._sell_asset(i, action)

        # Buy action
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
        sell_shares = min(self.shares[index], -action[index] * self.shares[index])
        self.shares[index] -= sell_shares
        self.cash += sell_shares * price[index] * (1 - self.tc)


    def _buy_asset(self, index, action):

        price = self.price[self.time]
        total_asset = self._portfolio_value(price)
        max_exposure = self.position_limit[index] * total_asset

        current_value = self.shares[index] * price[index]
        remaining = max(0, max_exposure - current_value)

        max_shares = remaining / price[index]

        desired_shares = action[index] * total_asset / price[index]

        buy_shares = min(
            desired_shares,
            self.cash / (price[index] * (1 + self.tc)),
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
        ]).astype(np.float32)

        return obs


    def _reward(self, prev_asset, next_asset):

        return next_asset - prev_asset


    def _portfolio_value(self, price):
        return self.cash + np.sum(self.shares * price)


    def close(self):
        pass