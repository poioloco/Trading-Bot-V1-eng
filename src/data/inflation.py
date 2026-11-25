"""Price index (inflation) from FRED — ONLY for the dashboard chart: the
dashed line «cash net of inflation». It does not enter any signal
calculation.

The series is monthly and is downloaded from FRED's public CSV (no key,
like TDEX). A local copy is kept (data/inflation_<SERIES>.csv) so the
dashboard also works without internet; the copy is refreshed at most once
every CACHE_DAYS days.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_SERIES = "CPIAUCSL"
# short name shown in the chart
SERIES_NAMES = {
    "CPIAUCSL": "USA",
    "CP0000EZ19M086NEST": "euro area",
    "CP0000ITM086NEST": "Italy",
}
CACHE_DAYS = 7


def series_name(series: str) -> str:
    return SERIES_NAMES.get(series.upper(), series)


def _parse(text: str) -> pd.Series:
    """FRED CSV (date, value) -> sorted series; '.' entries are missing values."""
    if not text or text.lstrip().lower().startswith("<"):
        raise ValueError("response is not CSV (HTML/error page)")
    df = pd.read_csv(io.StringIO(text))
    ser = pd.Series(pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy(),
                    index=pd.to_datetime(df.iloc[:, 0])).dropna().sort_index()
    if ser.empty:
        raise ValueError("empty series")
    return ser


def price_index(series: str = DEFAULT_SERIES, folder: str | Path = "data",
                timeout: int = 15) -> pd.Series | None:
    """Monthly price index. Tries FRED if the local copy is missing or old;
    if FRED does not answer it uses the local copy; if there is none either
    it returns None (the dashboard goes without the dashed line)."""
    series = (series or DEFAULT_SERIES).strip().upper()
    cache = Path(folder) / f"inflation_{series}.csv"
    fresh = (cache.exists() and
             time.time() - cache.stat().st_mtime < CACHE_DAYS * 86400)
    if not fresh:
        try:
            resp = requests.get(FRED_CSV_URL, params={"id": series},
                                timeout=timeout)
            resp.raise_for_status()
            ser = _parse(resp.text)
            cache.parent.mkdir(parents=True, exist_ok=True)
            ser.rename(series).to_csv(cache, index_label="date")
            return ser
        except Exception as exc:
            logger.warning("Inflation %s not downloaded from FRED: %s", series, exc)
    if cache.exists():
        try:
            return _parse(cache.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Local copy %s unreadable: %s", cache, exc)
    return None


def purchasing_power(index: pd.Series,
                     days: pd.DatetimeIndex) -> pd.Series | None:
    """What 1 held in cash since the first day of `days` is worth in real
    terms: index(first day) / index(day).

    The monthly figure is the month's average: it is placed mid-month and
    interpolated day by day. After the last published value (FRED publishes
    with a ~2-6 week delay) the value stays flat at the last one."""
    ind = index.dropna().sort_index()
    if ind.empty or len(days) == 0:
        return None
    ind.index = ind.index + pd.Timedelta(days=14)
    every = ind.index.union(days)
    daily = (ind.reindex(every).interpolate(method="time", limit_area="inside")
             .ffill().reindex(days))
    valid = daily.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[0]) / daily
