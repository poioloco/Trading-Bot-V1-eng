"""Base class for all market data providers.

Every provider returns daily OHLCV bars as a pandas DataFrame with:
  - index: DatetimeIndex (tz-naive, daily)
  - columns: open, high, low, close, volume (floats)
sorted ascending by date.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataProviderError(Exception):
    """Raised when a provider cannot return usable data."""


class DataProvider(ABC):
    """Abstract market-data provider."""

    name: str = "base"

    @abstractmethod
    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return daily OHLCV bars for `symbol` in [start, end]."""

    # ----------------------------------------------------------------- #
    # Shared validation/normalisation helpers                           #
    # ----------------------------------------------------------------- #
    @staticmethod
    def _normalize(df: pd.DataFrame, symbol: str, provider: str) -> pd.DataFrame:
        if df is None or df.empty:
            raise DataProviderError(f"{provider}: empty data for {symbol}")

        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataProviderError(
                f"{provider}: missing columns {missing} for {symbol}"
            )

        df = df[REQUIRED_COLUMNS].astype(float)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep="last")].sort_index()

        if df["close"].isna().all():
            raise DataProviderError(f"{provider}: all-NaN closes for {symbol}")
        return df
