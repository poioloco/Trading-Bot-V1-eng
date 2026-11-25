"""Synthetic providers for tests: deterministic, no network."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.providers.base import DataProvider, DataProviderError


class FakeProvider(DataProvider):
    """Generates deterministic daily bars.

    Trend per symbol is controlled via `trends`: symbol -> 'up'|'down'|'flat'
    (default 'up').
    """

    name = "fake"

    def __init__(self, trends: dict[str, str] | None = None) -> None:
        self.trends = trends or {}

    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        idx = pd.bdate_range(start=start, end=end)
        n = len(idx)
        trend = self.trends.get(symbol, "up")
        if trend == "up":
            close = np.linspace(100.0, 200.0, n)
        elif trend == "down":
            close = np.linspace(200.0, 100.0, n)
        else:  # flat with a tiny deterministic ripple
            close = 100.0 + 0.01 * np.sin(np.arange(n))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(n, 1_000.0),
            },
            index=idx,
        )
        return self._normalize(df, symbol, self.name)


class FailingProvider(DataProvider):
    name = "failing"

    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        raise DataProviderError(f"failing: simulated outage for {symbol}")
