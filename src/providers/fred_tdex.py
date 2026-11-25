"""TDEX daily series from FRED (St. Louis Fed), series NASDAQTDEX.

Validated empirically against the Excel's RTD feed: 15/15 exact matches
on scattered dates 2018-2024 (including extremes: 94.48 on 2018-02-05,
56.42 on 2020-03-20) and 4/5 on Mar-2026; the single difference was an
intraday RTD snapshot vs the official close. The series is the official
Nasdaq GIDS dissemination of the Nations TailDex, updated nightly
(~11:30 PM US Central), i.e. the prior session's close.

Endpoints (no API key needed):
  primary : https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQTDEX
  fallback: FRED API (requires free api key) if configured

Licensing note: FRED terms allow personal, non-commercial use; the
series is Nasdaq-copyrighted ("Pre-Approval Required" for redistribution).
"""
from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
import requests

from .base import DataProviderError

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES_ID = "NASDAQTDEX"


class FredTdexProvider:
    def __init__(
        self,
        series_id: str = SERIES_ID,
        api_key: str | None = None,
        session: requests.Session | None = None,
        timeout: int = 30,
    ) -> None:
        self.series_id = series_id
        self.api_key = api_key
        self._session = session or requests.Session()
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    def get_series(
        self, start: date | None = None, end: date | None = None
    ) -> pd.Series:
        """Daily TDEX closes (holidays dropped). Index: DatetimeIndex.

        Returns ONLY the days in [start, end]. FRED sometimes ignores the
        date filter and sends the whole series: without this cut the
        incremental update used to rewrite the whole history (including data
        imported by hand). If FRED answers but has not published new days
        yet, the series comes back EMPTY (it is not an error)."""
        errors = []
        try:
            return self._clip(self._from_csv(start, end), start, end)
        except (requests.RequestException, DataProviderError, ValueError) as exc:
            logger.warning("FRED CSV endpoint failed: %s", exc)
            errors.append(exc)
        if self.api_key:
            try:
                return self._clip(self._from_api(start, end), start, end)
            except (requests.RequestException, DataProviderError, ValueError) as exc:
                logger.warning("FRED API endpoint failed: %s", exc)
                errors.append(exc)
        raise DataProviderError(f"FRED TDEX: all endpoints failed: {errors}")

    @staticmethod
    def _clip(ser: pd.Series, start: date | None, end: date | None) -> pd.Series:
        """Keeps only the observations with start <= date <= end."""
        if start is not None:
            ser = ser[ser.index >= pd.Timestamp(start)]
        if end is not None:
            ser = ser[ser.index <= pd.Timestamp(end)]
        return ser

    # ------------------------------------------------------------------ #
    def _from_csv(self, start, end) -> pd.Series:
        params = {"id": self.series_id}
        if start:
            params["cosd"] = start.isoformat()
        if end:
            params["coed"] = end.isoformat()
        resp = self._session.get(FRED_CSV_URL, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return self.parse_csv(resp.text, self.series_id)

    def _from_api(self, start, end) -> pd.Series:
        params = {
            "series_id": self.series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start.isoformat()
        if end:
            params["observation_end"] = end.isoformat()
        resp = self._session.get(FRED_API_URL, params=params, timeout=self._timeout)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if not obs:
            raise DataProviderError("FRED API: no observations")
        idx, vals = [], []
        for o in obs:
            if o.get("value") in (".", "", None):
                continue
            idx.append(pd.Timestamp(o["date"]))
            vals.append(float(o["value"]))
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="tdex")

    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_csv(text: str, series_id: str = SERIES_ID) -> pd.Series:
        """FRED CSV: header DATE,<SERIES>; market holidays carry '.'."""
        if not text or text.lstrip().lower().startswith("<html"):
            raise DataProviderError("FRED CSV: HTML response (blocked/error)")
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip().upper() for c in df.columns]
        date_col = "DATE" if "DATE" in df.columns else df.columns[0]
        val_col = series_id.upper() if series_id.upper() in df.columns else df.columns[-1]
        ser = (
            pd.Series(
                pd.to_numeric(df[val_col], errors="coerce").to_numpy(),
                index=pd.to_datetime(df[date_col]),
                name="tdex",
            )
            .dropna()
            .sort_index()
        )
        if ser.empty:
            raise DataProviderError("FRED CSV: empty series after parsing")
        return ser
