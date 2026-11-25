"""Alpaca Market Data provider (primary source).

Uses the plain REST API (no SDK dependency) with the free-tier IEX feed.
Credentials are read from environment variables defined in config.yaml.
"""
from __future__ import annotations

import logging
import os
from datetime import date

import pandas as pd
import requests

from .base import DataProvider, DataProviderError

logger = logging.getLogger(__name__)


class AlpacaProvider(DataProvider):
    name = "alpaca"

    def __init__(
        self,
        base_url: str = "https://data.alpaca.markets",
        feed: str = "iex",
        api_key_env: str = "ALPACA_API_KEY",
        api_secret_env: str = "ALPACA_API_SECRET",
        session: requests.Session | None = None,
        timeout: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.feed = feed
        self._key = os.environ.get(api_key_env, "")
        self._secret = os.environ.get(api_secret_env, "")
        self._session = session or requests.Session()
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if not self._key or not self._secret:
            raise DataProviderError("alpaca: missing API credentials in environment")

        url = f"{self.base_url}/v2/stocks/{symbol}/bars"
        params: dict = {
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": "split",
            "feed": self.feed,
            "limit": 10000,
        }
        headers = {
            "APCA-API-KEY-ID": self._key,
            "APCA-API-SECRET-KEY": self._secret,
        }

        rows: list[dict] = []
        page_token: str | None = None
        while True:
            if page_token:
                params["page_token"] = page_token
            try:
                resp = self._session.get(
                    url, params=params, headers=headers, timeout=self._timeout
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                raise DataProviderError(f"alpaca: request failed for {symbol}: {exc}")

            payload = resp.json()
            rows.extend(self.parse_bars(payload))
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        if not rows:
            raise DataProviderError(f"alpaca: no bars returned for {symbol}")

        df = pd.DataFrame(rows).set_index("t")
        return self._normalize(df, symbol, self.name)

    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_bars(payload: dict) -> list[dict]:
        """Convert an Alpaca /bars JSON payload page into flat row dicts."""
        out = []
        for bar in payload.get("bars") or []:
            out.append(
                {
                    "t": bar["t"],
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"],
                }
            )
        return out
