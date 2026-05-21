from __future__ import annotations

import datetime
import time
from typing import Sequence

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import numpy as np
from stable_baselines3 import PPO
from meta.processor_alpaca import AlpacaProcessor


class AlpacaPaperTrading:
    """
    Live paper trading wrapper for a PPO agent trained on
    MultiAssetTradingEnv.

    Observation structure MUST match training env:

    [
        cash,
        shares,
        prices,
        indicators_flat
    ]
    """

    def __init__(
        self,
        model_path: str,
        ticker_list: Sequence[str],
        tech_indicator_list: Sequence[str],
        api_key: str,
        api_secret: str,
        trading_interval: str = "1m",
        transaction_cost: float = 1e-3,
        position_limit: float | np.ndarray = 0.2,
        min_trade_fraction: float = 0.05,
        timeframe: str = "1d",
        limit: int = 100,
    ):

        self.tickers = list(ticker_list)
        self.N = len(self.tickers)
        self.tech_indicators = list(tech_indicator_list)

        # Load trained model
        self.model = PPO.load(model_path)

        # Initialize trading client
        self.trade_client = TradingClient(
            api_key,
            api_secret,
            paper=True,
        )

        # Initialize data client
        self.data_client = AlpacaProcessor(
            api_key,
            api_secret,
        )

        # Trading settings
        self.transaction_cost = transaction_cost
        self.min_trade_fraction = min_trade_fraction
        self.timeframe = timeframe
        self.limit = limit

        if isinstance(position_limit, float):
            self.position_limit = np.full(self.N, position_limit)
        else:
            if len(position_limit) != self.N:
                raise ValueError("position_limit length must match number of tickers")
            self.position_limit = np.asarray(position_limit)

        # Trading interval handling
        str_to_sec = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "1d": 86400,
        }

        trading_interval = trading_interval.lower()
        if trading_interval not in str_to_sec:
            raise ValueError(f"Unsupported trading interval: {trading_interval}")

        self.sleep_sec = str_to_sec[trading_interval]
    
    # =========================================================
    # Main loop
    # =========================================================

    def run(self):

        self._cancel_open_orders()

        print("Waiting for market open...")
        self._await_market_open()
        print("Market open.")

        while True:

            clock = self.trade_client.get_clock()

            closing_time = clock.next_close.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()

            current_time = clock.timestamp.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()

            time_to_close = closing_time - current_time

            # stop trading near close
            if time_to_close < 60:
                print("Market closing soon. Stopping trading.")
                break

            try:
                self.trade()

            except Exception as e:
                print(f"[ERROR] trade() failed: {e}")

            time.sleep(self.sleep_sec)

    # =========================================================
    # Single trade step
    # =========================================================

    def trade(self):

        obs = self.get_observation()

        action, _ = self.model.predict(obs, deterministic=True)

        action = np.asarray(action)
        action = np.clip(action, -1.0, 1.0)

        prices = self.get_prices()
        shares = self.get_positions()

        # Sell first
        for i in range(self.N):

            if prices[i] <= 0:
                continue

            if action[i] >= -self.min_trade_fraction:
                continue

            sell_fraction = min(abs(action[i]), 1.0)

            sell_shares = sell_fraction * shares[i]

            qty = sell_shares

            if qty <= 0:
                continue

            self.submit_order(
                symbol=self.tickers[i],
                qty=qty,
                side="sell",
            )

        # refresh after sells
        shares = self.get_positions()
        cash = self.get_cash()
        portfolio_value = cash + np.sum(shares * prices)

        # Buy second
        for i in range(self.N):

            if prices[i] <= 0:
                continue

            if action[i] <= self.min_trade_fraction:
                continue

            # target exposure
            max_exposure = self.position_limit[i] * portfolio_value

            current_exposure = shares[i] * prices[i]

            remaining_exposure = max(
                0.0,
                max_exposure - current_exposure,
            )

            desired_dollars = action[i] * portfolio_value

            buy_dollars = min(
                desired_dollars,
                remaining_exposure,
                cash,
            )

            qty = (
                buy_dollars
                / (prices[i] * (1 + self.transaction_cost))
            )

            if qty <= 0:
                continue

            self.submit_order(
                symbol=self.tickers[i],
                qty=qty,
                side="buy",
            )

            cash -= qty * prices[i] * (1 + self.transaction_cost)


    def get_observation(self):

        cash = self.get_cash()
        shares = self.get_positions()
        prices = self.get_prices()

        tech = self.data_client.get_tech_indicators(
            tickers=self.tickers,
            indicators=self.tech_indicators,
            timeframe=self.timeframe,
            limit=self.limit,
        )

        obs = np.concatenate([
            [cash],
            shares,
            prices,
            tech.flatten(),
        ]).astype(np.float32)

        return obs


    def get_prices(self) -> np.ndarray:

        return self.data_client.get_prices(self.tickers)


    def get_cash(self) -> float:

        account = self.trade_client.get_account()

        return float(account.cash)

    def get_positions(self) -> np.ndarray:

        positions = self.trade_client.get_all_positions()

        shares = np.zeros(self.N, dtype=np.float32)

        for position in positions:

            if position.symbol not in self.tickers:
                continue

            idx = self.tickers.index(position.symbol)

            shares[idx] = float(position.qty)

        return shares

    # =========================================================
    # Order execution
    # =========================================================

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
    ):
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, 
        )

        try:

            self.trade_client.submit_order(order)

            print(f"{side.upper()} {qty} {symbol}")

        except Exception as e:

            print(
                f"[ORDER FAILED] "
                f"{side} {qty} {symbol}: {e}"
            )

    # =========================================================
    # Market utilities
    # =========================================================

    def _cancel_open_orders(self):
        self.trade_client.cancel_orders()


    def _await_market_open(self):

        while True:

            clock = self.trade_client.get_clock()

            if clock.is_open:
                return

            opening_time = clock.next_open.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()

            current_time = clock.timestamp.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()

            minutes = int((opening_time - current_time) / 60)

            print(f"{minutes} minutes until market open")

            time.sleep(60)
