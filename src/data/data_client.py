from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from stockstats import StockDataFrame as Sdf


_TIMEFRAME_MAP = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}

class DataClient:
    """
    Handles ALL market data acquisition for live trading.

    Responsibilities:
    - latest prices
    - technical indicators
    - historical OHLCV retrieval

    Returns NumPy arrays compatible with RL environments.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
    ):

        self.client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )


    def get_latest_prices(
        self,
        tickers: Sequence[str],
    ) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray shape (N,)
        """
        request = StockLatestTradeRequest(symbol_or_symbols=list(tickers))
        trades  = self.client.get_stock_latest_trade(request)

        prices = np.zeros(len(tickers), dtype=np.float32)
        for i, ticker in enumerate(tickers):
            trade = trades.get(ticker)
            if trade is None:
                raise ValueError(f"No latest trade for {ticker}")
            prices[i] = float(trade.price)

        return prices  # (N,)


    def get_latest_indicators(
        self,
        tickers: Sequence[str],
        indicators: Sequence[str],
        timeframe: str = "1d",
        limit: int = 100,
    ) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray shape (N_tickers, N_indicators)
        """
        assert limit >= 60, "limit must be >= 60 for slow indicators like close_60_sma"

        timeframe = timeframe.lower()
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'."
                             f"Supported timeframes: {list(_TIMEFRAME_MAP.keys())}")

        # Single batched request
        request = StockBarsRequest(
            symbol_or_symbols=list(tickers),
            timeframe=_TIMEFRAME_MAP[timeframe],
            limit=limit,
        )
        bars_df = self.client.get_stock_bars(request).df.reset_index()

        all_features = []

        for ticker in tickers:
            ticker_bars = bars_df[bars_df["symbol"] == ticker].copy()

            if ticker_bars.empty:
                raise ValueError(f"No data for {ticker}")

            df = pd.DataFrame({
                "open": ticker_bars["open"].astype(float).values,
                "high": ticker_bars["high"].astype(float).values,
                "low": ticker_bars["low"].astype(float).values,
                "close": ticker_bars["close"].astype(float).values,
                "volume": ticker_bars["volume"].astype(float).values,
            })

            stock = Sdf.retype(df)
            features = []

            for indicator in indicators:
                try:
                    value = stock[indicator].values[-1]
                    value = 0.0 if np.isnan(value) else float(value)
                except Exception as e:
                    raise ValueError(f"Failed to compute '{indicator}' for {ticker}: {e}")
                features.append(value)

            all_features.append(features)

        return np.array(all_features, dtype=np.float32)  # (N, F)


    def get_latest_vix(self) -> float:

        request = StockLatestTradeRequest(symbol_or_symbols=["VIX"])
        trades = self.client.get_stock_latest_trade(request)

        trade = trades.get("VIX")

        if trade is None:
            raise ValueError("No VIX data available")

        return float(trade.price)


    def get_price_history(
        self,
        tickers: Sequence[str],
        timeframe: str = "1d",
        limit: int = 100,
    ) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray shape (T, N) of close prices, ordered by time.
        """
        timeframe = timeframe.lower()
        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'."
                f"Supported timeframes: {list(_TIMEFRAME_MAP.keys())}"
            )

        request = StockBarsRequest(
            symbol_or_symbols=list(tickers),
            timeframe=_TIMEFRAME_MAP[timeframe],
            limit=limit,
        )
        bars_df = self.client.get_stock_bars(request).df.reset_index()

        prices = np.zeros((limit, len(tickers)), dtype=np.float32)

        for i, ticker in enumerate(tickers):
            ticker_bars = bars_df[bars_df["symbol"] == ticker].copy()

            if ticker_bars.empty:
                raise ValueError(f"No data for {ticker}")

            ticker_bars = ticker_bars.sort_values("timestamp")

            if len(ticker_bars) < limit:
                raise ValueError(
                    f"Not enough bars for {ticker}: {len(ticker_bars)} < {limit}"
                )

            prices[:, i] = ticker_bars["close"].astype(float).values[-limit:]

        return prices
