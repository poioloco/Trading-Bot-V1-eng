"""PointInTimeStore: the source of truth for the rest of the system.

It makes a single promise: *"give me the data for this symbol up to date T"*
and **never** returns a row dated after T, nor uses a corporate action not
yet known at T. It is the anti-lookahead guarantee enforced by the lowest
layer: the backtest cannot cheat even if it wanted to.

It implements the `DataProvider` interface (`get_daily_bars(symbol, start, end)`)
so it is a transparent replacement for the network provider: the exact same
indicator and signal code runs both live (end = today) and in the backtest
(end = every day since 2005), reading from the database instead of the
network. So "same code live and in backtest" is guaranteed by construction.

The dividend and split adjustment is computed AT READ TIME from the raw
prices and only the actions with ex-date <= T (back-adjusted: the most
recent price stays the raw one).
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from ..providers.base import DataProvider, DataProviderError
from .database import Database

logger = logging.getLogger(__name__)


def _adjustment_factor(
    dates: pd.DatetimeIndex, raw_close: np.ndarray, actions: list[tuple]
) -> np.ndarray:
    """Back-adjusted adjustment factor for each date.

    `actions`: list of (ex_date: Timestamp, type: str, value: float), already
    filtered to ex_date <= T. Rows dated < ex_date are scaled.
    """
    factor = np.ones(len(dates), dtype=float)
    for ex_date, atype, value in actions:
        mask = dates < ex_date
        if not mask.any():
            continue
        if atype == "split":
            if value and value > 0:
                factor[mask] *= 1.0 / value
        elif atype == "dividend":
            prior = dates[dates < ex_date]
            if len(prior) == 0:
                continue
            c_prev = raw_close[dates.get_loc(prior[-1])]
            if c_prev and c_prev > 0:
                m = 1.0 - value / c_prev
                if m > 0:
                    factor[mask] *= m
    return factor


class PointInTimeStore(DataProvider):
    name = "duckdb"

    def __init__(self, db: Database, adjusted: bool = True) -> None:
        self.db = db
        self.adjusted = adjusted

    # ------------------------------------------------------------------ #
    def get_bars_asof(
        self, symbol: str, as_of: date, adjusted: bool | None = None
    ) -> pd.DataFrame:
        """OHLCV bars of `symbol` dated <= `as_of`. Adjusted (default)
        using only the actions known by `as_of`."""
        use_adj = self.adjusted if adjusted is None else adjusted
        raw = self.db.con.execute(
            "SELECT date, open, high, low, close, volume FROM prices_raw "
            "WHERE symbol = ? AND date <= ? ORDER BY date",
            [symbol, as_of],
        ).fetchdf()
        if raw.empty:
            raise DataProviderError(f"store: no data for {symbol} by {as_of}")
        raw["date"] = pd.to_datetime(raw["date"]).astype("datetime64[ns]")
        raw = raw.set_index("date").sort_index()

        if not use_adj:
            return raw[["open", "high", "low", "close", "volume"]]

        acts = self.db.con.execute(
            "SELECT date, action_type, value FROM corporate_actions "
            "WHERE symbol = ? AND date <= ? ORDER BY date",
            [symbol, as_of],
        ).fetchdf()
        if acts.empty:
            return raw[["open", "high", "low", "close", "volume"]]

        actions = [
            (pd.Timestamp(r.date), str(r.action_type), float(r.value))
            for r in acts.itertuples()
        ]
        dates = raw.index
        factor = _adjustment_factor(dates, raw["close"].to_numpy(), actions)
        out = raw.copy()
        for c in ("open", "high", "low", "close"):
            out[c] = out[c] * factor
        # volume: more shares after a split -> scale by the inverse of the
        # price factor for splits only (dividends do not change the share count)
        split_only = [a for a in actions if a[1] == "split"]
        if split_only:
            vfac = _adjustment_factor(dates, raw["close"].to_numpy(), split_only)
            out["volume"] = out["volume"] / np.where(vfac == 0, 1.0, vfac)
        return out[["open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------ #
    # DataProvider interface: transparent replacement for the network one #
    # ------------------------------------------------------------------ #
    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Bars in [start, end]. By construction no row after `end` is ever
        returned: calling with end=T gives exactly the as-of-T view."""
        bars = self.get_bars_asof(symbol, end)
        bars = bars[bars.index >= pd.Timestamp(start)]
        if bars.empty:
            raise DataProviderError(
                f"store: no data for {symbol} in [{start}, {end}]"
            )
        return bars

    # ------------------------------------------------------------------ #
    def tdex_series(self, as_of: date) -> pd.Series:
        """TDEX series (single value/day) up to `as_of`, from the raw close."""
        df = self.db.con.execute(
            "SELECT date, close FROM prices_raw WHERE symbol = 'TDEX' "
            "AND date <= ? ORDER BY date",
            [as_of],
        ).fetchdf()
        if df.empty:
            raise DataProviderError(f"store: no TDEX data by {as_of}")
        return pd.Series(
            df["close"].to_numpy(),
            index=pd.to_datetime(df["date"]).astype("datetime64[ns]"),
            name="tdex",
        )

    def available_range(self, symbol: str) -> tuple[date, date] | None:
        row = self.db.con.execute(
            "SELECT MIN(date), MAX(date) FROM prices_raw WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if not row or row[0] is None:
            return None
        return row[0], row[1]
