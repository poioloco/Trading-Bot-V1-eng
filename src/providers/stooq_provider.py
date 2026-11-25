"""Stooq provider (fallback source).

Stooq exposes free daily history as CSV:
    https://stooq.com/q/d/l/?s=<symbol>.us&i=d&d1=YYYYMMDD&d2=YYYYMMDD
US tickers must carry the `.us` suffix (added automatically when absent).
"""
from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
import requests

from .base import DataProvider, DataProviderError

logger = logging.getLogger(__name__)


class StooqProvider(DataProvider):
    name = "stooq"

    BASE_URL = "https://stooq.com/q/d/l/"

    def __init__(
        self, session: requests.Session | None = None, timeout: int = 20
    ) -> None:
        self._session = session or requests.Session()
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        stooq_symbol = self.to_stooq_symbol(symbol)
        params = {
            "s": stooq_symbol,
            "i": "d",
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
        }
        try:
            resp = self._session.get(
                self.BASE_URL, params=params, timeout=self._timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise DataProviderError(f"stooq: request failed for {symbol}: {exc}")

        df = self.parse_csv(resp.text, symbol)
        return self._normalize(df, symbol, self.name)

    # ------------------------------------------------------------------ #
    @staticmethod
    def to_stooq_symbol(symbol: str) -> str:
        s = symbol.lower()
        # Indices like ^VIX keep their own notation; plain US tickers get .us
        if s.startswith("^") or "." in s:
            return s
        return f"{s}.us"

    @staticmethod
    def parse_csv(text: str, symbol: str) -> pd.DataFrame:
        if not text or text.strip().lower().startswith(("no data", "<html")):
            raise DataProviderError(f"stooq: no data for {symbol}")
        df = pd.read_csv(io.StringIO(text))
        if "Date" not in df.columns or df.empty:
            raise DataProviderError(f"stooq: malformed CSV for {symbol}")
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        if "volume" not in df.columns:  # indices have no volume on stooq
            df["volume"] = 0.0
        return df.set_index("date")
