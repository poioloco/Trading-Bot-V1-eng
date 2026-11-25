"""Staleness guard for forward-filled series (TDEX from FRED).

TDEX comes from FRED with normal gaps (holidays, calendar misalignments):
a 1-3 day forward-fill is correct, it is a level series. But if the feed
silently stops, the ffill would project the last value forever and the bot
would vote on dead data.

`ffill_with_guard` fills the gaps as before, but ONLY as long as the last
real observation is not older than `max_lag_days` (calendar days). Beyond
that limit the series goes back to NaN: in majority mode the NaN makes the
TDEX leg ABSTAIN (same semantics as pre-2013), so the other legs still
decide. With `max_lag_days=None` the guard is off and the behaviour stays
the plain ffill used before.
"""
from __future__ import annotations

import pandas as pd


def ffill_with_guard(
    series: pd.Series,
    index: pd.Index,
    max_lag_days: int | None = 10,
) -> pd.Series:
    """Re-aligns `series` on `index` and forward-fills the gaps, but values
    older than `max_lag_days` calendar days become NaN (the leg abstains
    instead of voting on stale data)."""
    aligned = series.reindex(index)
    filled = aligned.ffill()
    if max_lag_days is None:
        return filled
    # date of the last REAL observation for each row (NaT before the first)
    idx = pd.Series(pd.DatetimeIndex(index), index=index)
    last_obs = idx.where(aligned.notna()).ffill()
    lag = (idx - last_obs).dt.days
    return filled.where(lag <= max_lag_days)


def last_data_lag(series: pd.Series, as_of) -> int | None:
    """Calendar days between `as_of` and the last real observation of the
    series (None if the series is empty). Used for diagnostic messages."""
    s = series.dropna()
    if s.empty:
        return None
    return int((pd.Timestamp(as_of) - pd.Timestamp(s.index.max())).days)
