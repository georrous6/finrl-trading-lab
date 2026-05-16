from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CryptoTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        lookback=1,
        initial_capital=1e6,
        position_limit_pct=1e-2,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        volatility_penalty=0.0,
        gamma=0.99,
    ):
        self.lookback = lookback
        self.initial_cash = initial_capital
        self.position_limit_pct = position_limit_pct
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.price_array = config["price_array"]
        self.tech_array = config["tech_array"]
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - lookback - 1
        self.volatility_penalty = volatility_penalty

        # reset
        self.time = lookback - 1
        self.cash = self.initial_cash
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.asset_history = [self.initial_cash]  # for recording sharpe ratio
        self.gamma_return = 0.0
        self.cumu_return = 1.0

        # Evironment information
        self.env_name = "MulticryptoEnv"
        self.state_dim = (
            1 + self.crypto_num +
            (self.crypto_num + self.tech_array.shape[1]) * lookback
        )
        self.action_dim = self.crypto_num

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.crypto_num,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )


    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> tuple[np.ndarray, dict]:
        self.time = self.lookback - 1
        self.cash = self.initial_cash
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.asset_history = [self.total_asset]
        self.gamma_return = 0.0
        self.cumu_return = 1.0

        state = self._get_state()
        return state, {}


    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict]:

        # Move to the next time step
        self.time += 1

        # Get the current price of the assets
        price = self.price_array[self.time]

        # Execute sell actions
        for sell_index in np.where(actions < 0)[0]:

            # Sell only if the price is > 0 
            # (no missing data in this particular date)
            if price[sell_index] > 0:
                sell_num_shares = min(self.stocks[sell_index], -actions[sell_index])
                self.stocks[sell_index] -= sell_num_shares
                self.cash += price[sell_index] * sell_num_shares * (1 - self.sell_cost_pct)

        # Execute buy actions
        for buy_index in np.where(actions > 0)[0]:

            # Buy only if the price is > 0 
            # (no missing data in this particular date)
            if price[buy_index] > 0:
                
                max_position_value = self.position_limit_pct * self.total_asset

                max_shares_allowed = max(0, max_position_value / price[buy_index])

                buy_num_shares = min(
                    self.cash / (price[buy_index] * (1 + self.buy_cost_pct)),
                    actions[buy_index],
                    max_shares_allowed
                )
                self.stocks[buy_index] += buy_num_shares
                self.cash -= price[buy_index] * buy_num_shares * (1 + self.buy_cost_pct)

        # Update current price and technical indicators
        terminated = self.time == self.max_step
        state = self._get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        reward = self._compute_reward(next_total_asset)
        self.asset_history.append(next_total_asset)
        self.total_asset = next_total_asset
        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash

        # Store meta information and performance metrics
        step_info = {
            'total_asset': self.total_asset,
            'cash': self.cash,
            'stocks': self.stocks,
            'cumulative_return': self.cumu_return,
        }

        episode_info = {
            'gamma_return': None,
            'sharpe_ratio': None,
        }

        if terminated:
            episode_info.update({
                'gamma_return': self.gamma_return,
                'sharpe_ratio': self._compute_sharpe_ratio(),
            })

        truncated = False
        info = {
            'step_info': step_info, 
            'episode_info': episode_info
        }
        return state, reward, terminated, truncated, info


    def _get_state(self):

        # ===== Technical indicators state ======
        tech_start = max(0, self.time - self.lookback)
        tech_window = self.tech_array[tech_start:self.time]

        # Compute rolling window statistics for normalization
        if len(tech_window) < 2:
            tech_mean = np.zeros_like(self.tech_array[self.time])
            tech_std = np.ones_like(self.tech_array[self.time])
        else:
            tech_mean = tech_window.mean(axis=0)
            tech_std = np.maximum(tech_window.std(axis=0), 1e-8)

        # Normalize technical indicators
        tech_state = (tech_window - tech_mean) / tech_std

        # Pad technical indicators to match lookback shape
        tech_state_padded = np.zeros((self.lookback, self.tech_array.shape[1]), dtype=np.float32)
        if len(tech_window) > 0:
            tech_state_padded[-len(tech_window):] = tech_state
        tech_state = tech_state_padded

        # ====== Price state ======
        price_start = max(0, self.time - self.lookback - 1)  # +1 point for log return calculation
        price_window = self.price_array[price_start:self.time]
        price_window = np.maximum(price_window, 1e-8) 

        log_returns = np.log(price_window[1:] / price_window[:-1])

        # Pad log returns to match lookback shape
        log_returns_padded = np.zeros((self.lookback, self.crypto_num), dtype=np.float32)
        if len(log_returns) > 0:
            log_returns_padded[-len(log_returns):] = log_returns
        log_returns = log_returns_padded

        # ====== Portfolio state ======
        price = self.price_array[self.time]
        cash_state = np.array([self.cash / (self.total_asset + 1e-8)], dtype=np.float32)
        stocks_state = (self.stocks * price) / (self.total_asset + 1e-8)

        return np.concatenate([cash_state, 
                          stocks_state,
                          log_returns.flatten(), 
                          tech_state.flatten()]).astype(np.float32)


    def close(self):
        pass


    def _compute_sharpe_ratio(self):
        asset = np.array(self.asset_history)

        if len(asset) < 2:
            return 0.0
        
        returns = asset[1:] / asset[:-1] - 1

        if np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / (np.std(returns) + 1e-8)


    def _compute_reward(self, next_total_asset):

        asset = np.array(self.asset_history)

        # step log return
        step_return = np.log((next_total_asset + 1e-8) /
                            (self.total_asset + 1e-8))

        # rolling window
        start = max(0, self.time - self.lookback)
        window = asset[start:self.time + 1]

        log_returns = np.log((window[1:] + 1e-8) / (window[:-1] + 1e-8))

        volatility = np.std(log_returns) if len(log_returns) > 1 else 0.0

        return step_return - self.volatility_penalty * volatility
