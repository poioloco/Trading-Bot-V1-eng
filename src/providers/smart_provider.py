"""SmartProvider: tries the primary provider, falls back automatically.

This is the only provider the rest of the codebase should instantiate.

Indices (^SPX, ^VIX): Alpaca does not provide them (400 Bad Request), so if
an `index_provider` is configured (FRED, the default) the index symbols start
from that one and use the fallback (stooq) as backup, without querying Alpaca.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from .base import DataProvider, DataProviderError

logger = logging.getLogger(__name__)


class SmartProvider(DataProvider):
    name = "smart"

    def __init__(self, primary: DataProvider, fallback: DataProvider,
                 index_provider: DataProvider | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.index_provider = index_provider

    def _chain_for(self, symbol: str) -> list[DataProvider]:
        ip = self.index_provider
        if ip is not None and getattr(ip, "supports", lambda s: False)(symbol):
            return [ip, self.fallback]
        return [self.primary, self.fallback]

    def get_daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        chain = self._chain_for(symbol)
        last_exc: DataProviderError | None = None
        for i, prov in enumerate(chain):
            try:
                return prov.get_daily_bars(symbol, start, end)
            except DataProviderError as exc:
                last_exc = exc
                if i < len(chain) - 1:
                    logger.warning(
                        "Primary provider '%s' failed for %s (%s); "
                        "falling back to '%s'",
                        prov.name, symbol, exc, chain[i + 1].name,
                    )
        raise DataProviderError(
            f"smart: both providers failed for {symbol}: {last_exc}"
        ) from last_exc


def build_smart_provider(config: dict) -> SmartProvider:
    """Factory: builds the SmartProvider from the `data` section of config.yaml."""
    from .alpaca_provider import AlpacaProvider
    from .fred_index_provider import FredIndexProvider
    from .stooq_provider import StooqProvider

    data_cfg = config.get("data", {})
    alpaca_cfg = data_cfg.get("alpaca", {})

    registry = {
        "alpaca": lambda: AlpacaProvider(
            base_url=alpaca_cfg.get("base_url", "https://data.alpaca.markets"),
            feed=alpaca_cfg.get("feed", "iex"),
            api_key_env=alpaca_cfg.get("api_key_env", "ALPACA_API_KEY"),
            api_secret_env=alpaca_cfg.get("api_secret_env", "ALPACA_API_SECRET"),
        ),
        "stooq": lambda: StooqProvider(),
    }
    primary = registry[data_cfg.get("primary_provider", "alpaca")]()
    fallback = registry[data_cfg.get("fallback_provider", "stooq")]()
    # 'fred' (default) = ^SPX/^VIX from FRED · anything else (e.g. 'none') = old behaviour
    index_provider = (
        FredIndexProvider()
        if data_cfg.get("indices_provider", "fred") == "fred"
        else None
    )
    return SmartProvider(primary=primary, fallback=fallback,
                         index_provider=index_provider)
