"""Closes of the INDICES (^SPX, ^VIX) from FRED (St. Louis Fed).

Why it is needed: Alpaca does not provide indices (the /v2/stocks endpoint
answers 400 for ^SPX and ^VIX) and Stooq's CSV endpoint has become
unreliable (404 on programmatic requests). FRED publishes the official
closes without an API key:

  ^SPX → series SP500   (S&P 500, last ~10 years)
  ^VIX → series VIXCLS  (CBOE VIX, since 1990)

FRED only gives the close: open/high/low are set equal to the close and
volume to 0 — the bot's engines use closes exclusively, so nothing changes
in the calculations.

Licence: personal, non-commercial use (same terms as the TDEX series).
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from .base import REQUIRED_COLUMNS, DataProvider, DataProviderError
from .fred_tdex import FredTdexProvider

logger = logging.getLogger(__name__)

# bot symbol -> FRED series id
FRED_SERIES = {"^SPX": "SP500", "^VIX": "VIXCLS"}


class FredIndexProvider(DataProvider):
    name = "fred"

    def __init__(self, mapping: dict[str, str] | None = None,
                 timeout: int = 30) -> None:
        self._mapping = {k.upper(): v for k, v in (mapping or FRED_SERIES).items()}
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    def supports(self, symbol: str) -> bool:
        """True if the symbol has a matching FRED series."""
        return symbol.upper() in self._mapping

    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        series = self._mapping.get(symbol.upper())
        if not series:
            raise DataProviderError(f"fred: no FRED series for {symbol}")
        closes = FredTdexProvider(series_id=series,
                                  timeout=self._timeout).get_series(start, end)
        if closes.empty:
            # FRED answered but has not published the requested days yet
            # (e.g. yesterday's VIX only comes out around 09:40 New York time
            # of the next business day). It is not a failure: nothing to
            # write, and no fallback to another source.
            return pd.DataFrame(columns=REQUIRED_COLUMNS,
                                index=pd.DatetimeIndex([]), dtype=float)
        df = pd.DataFrame({
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": 0.0,
        })
        return self._normalize(df, symbol, self.name)
