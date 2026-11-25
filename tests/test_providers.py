"""Provider-layer tests (synthetic data, no network)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.providers.alpaca_provider import AlpacaProvider
from src.providers.base import DataProvider, DataProviderError
from src.providers.smart_provider import SmartProvider
from src.providers.stooq_provider import StooqProvider
from tests.fakes import FakeProvider, FailingProvider


def test_normalize_orders_and_validates():
    df = pd.DataFrame(
        {
            "Open": [2.0, 1.0],
            "High": [2.5, 1.5],
            "Low": [1.8, 0.9],
            "Close": [2.2, 1.2],
            "Volume": [10, 20],
        },
        index=["2026-01-05", "2026-01-02"],
    )
    out = DataProvider._normalize(df, "TEST", "unit")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.is_monotonic_increasing
    assert out.iloc[0]["close"] == 1.2

    with pytest.raises(DataProviderError):
        DataProvider._normalize(df.drop(columns=["Close"]), "TEST", "unit")


def test_alpaca_parse_bars():
    payload = {
        "bars": [
            {"t": "2026-01-02T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100},
            {"t": "2026-01-03T05:00:00Z", "o": 1.5, "h": 2.5, "l": 1, "c": 2, "v": 150},
        ],
        "next_page_token": None,
    }
    rows = AlpacaProvider.parse_bars(payload)
    assert len(rows) == 2
    assert rows[0]["close"] == 1.5 and rows[1]["volume"] == 150


def test_stooq_symbol_and_csv():
    assert StooqProvider.to_stooq_symbol("SPY") == "spy.us"
    assert StooqProvider.to_stooq_symbol("^VIX") == "^vix"

    csv = "Date,Open,High,Low,Close,Volume\n2026-01-02,1,2,0.5,1.5,100\n"
    df = StooqProvider.parse_csv(csv, "SPY")
    assert df.loc["2026-01-02", "close"] == 1.5

    with pytest.raises(DataProviderError):
        StooqProvider.parse_csv("No data", "SPY")


def test_smart_provider_falls_back():
    smart = SmartProvider(primary=FailingProvider(), fallback=FakeProvider())
    df = smart.get_daily_bars("SPY", date(2025, 1, 1), date(2026, 1, 1))
    assert not df.empty


def test_smart_provider_both_fail():
    smart = SmartProvider(primary=FailingProvider(), fallback=FailingProvider())
    with pytest.raises(DataProviderError):
        smart.get_daily_bars("SPY", date(2025, 1, 1), date(2026, 1, 1))


# ------------------------------------------------------------------ #
# Indices via FRED                                                    #
# ------------------------------------------------------------------ #
def test_fred_index_mapping_and_bars(monkeypatch):
    from src.providers import fred_index_provider as fip

    prov = fip.FredIndexProvider()
    assert prov.supports("^SPX") and prov.supports("^vix")
    assert not prov.supports("SPY")

    closes = pd.Series(
        [20.0, 21.5], index=pd.to_datetime(["2026-01-02", "2026-01-05"])
    )
    monkeypatch.setattr(
        fip.FredTdexProvider, "get_series", lambda self, s, e: closes
    )
    df = prov.get_daily_bars("^VIX", date(2026, 1, 1), date(2026, 1, 6))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [20.0, 21.5]
    assert (df["open"] == df["close"]).all()
    assert (df["volume"] == 0).all()

    with pytest.raises(DataProviderError):
        prov.get_daily_bars("SPY", date(2026, 1, 1), date(2026, 1, 6))


def test_smart_provider_routes_indices():
    from src.providers.fred_index_provider import FredIndexProvider

    class FakeIndexProvider(FakeProvider):
        name = "fred"
        supports = staticmethod(lambda s: s.startswith("^"))

    class FailingIndexProvider(FailingProvider):
        name = "fred"
        supports = staticmethod(lambda s: s.startswith("^"))

    # index: uses the index_provider, NOT the primary (which would fail here)
    smart = SmartProvider(primary=FailingProvider(), fallback=FailingProvider(),
                          index_provider=FakeIndexProvider())
    df = smart.get_daily_bars("^SPX", date(2025, 1, 1), date(2025, 2, 1))
    assert not df.empty

    # index with a broken index_provider: falls back to the fallback
    smart = SmartProvider(primary=FailingProvider(), fallback=FakeProvider(),
                          index_provider=FailingIndexProvider())
    df = smart.get_daily_bars("^VIX", date(2025, 1, 1), date(2025, 2, 1))
    assert not df.empty

    # non-index: classic primary->fallback path, index_provider ignored
    smart = SmartProvider(primary=FakeProvider(), fallback=FailingProvider(),
                          index_provider=FredIndexProvider())
    df = smart.get_daily_bars("SPY", date(2025, 1, 1), date(2025, 2, 1))
    assert not df.empty
