"""Technical indicator primitives.

Wilder RSI implementation matches the Excel model exactly:
  - first average gain/loss = simple mean over `period`
  - subsequent values = Wilder smoothing: avg = (prev*(period-1) + current)/period
Default period 21, as in the Toy Model (confirmed spec).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rsi(close: pd.Series, period: int = 21) -> pd.Series:
    """Wilder's RSI. Returns a Series aligned with `close` (NaN warm-up)."""
    if period < 1:
        raise ValueError("period must be >= 1")
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = pd.Series(np.nan, index=close.index)
    avg_loss = pd.Series(np.nan, index=close.index)
    if len(close) <= period:
        return pd.Series(np.nan, index=close.index, name="rsi")

    # Seed: simple average of the first `period` changes
    avg_gain.iloc[period] = gain.iloc[1 : period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1 : period + 1].mean()

    # Wilder smoothing
    g = avg_gain.iloc[period]
    l = avg_loss.iloc[period]
    for i in range(period + 1, len(close)):
        g = (g * (period - 1) + gain.iloc[i]) / period
        l = (l * (period - 1) + loss.iloc[i]) / period
        avg_gain.iloc[i] = g
        avg_loss.iloc[i] = l

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss == 0 and avg_gain > 0 -> RSI = 100; both 0 -> 50
    rsi = rsi.where(avg_loss != 0.0, np.where(avg_gain > 0.0, 100.0, 50.0))
    rsi[avg_gain.isna()] = np.nan
    return rsi.rename("rsi")


def simple_ma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()
