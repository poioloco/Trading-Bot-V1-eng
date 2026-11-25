"""Reproduces the two problems seen in November 2026 with FRED.

1. FRED ignored the date filter and sent the whole series: the incremental
   update rewrote the entire VIX/SPX/TDEX history (including hand-imported
   data). Now only the new days are written.
2. A day's VIX only comes out on the following business day (~09:40 New
   York time): before then the signal computation stopped with "no data
   for ^VIX". Now it computes on the last complete day and warns.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.app import (build_database, build_pipeline, build_store,
                     etf_symbols, load_params)
from src.data.ingest import incremental_update, seed_offline_demo
from src.engine.pipeline import date_with_indices
from src.providers import fred_tdex
from src.providers.fred_index_provider import FredIndexProvider
from src.providers.fred_tdex import FredTdexProvider


def _whole_series_csv(until: str) -> str:
    """CSV like the FRED one: the whole series, holidays with '.'."""
    days = pd.bdate_range("2026-10-01", until)
    rows = ["observation_date,VIXCLS"]
    for i, d in enumerate(days):
        rows.append(f"{d:%Y-%m-%d},{20 + i * 0.1:.2f}")
    rows.append("2026-11-27,.")
    return "\n".join(rows)


class _Response:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


# ------------------------------------------------------------------ #
# 1. No rewriting of the history                                      #
# ------------------------------------------------------------------ #
def test_fred_returns_only_the_requested_days(monkeypatch):
    text = _whole_series_csv("2026-11-19")
    prov = FredTdexProvider(series_id="VIXCLS")
    monkeypatch.setattr(prov._session, "get", lambda *a, **k: _Response(text))

    ser = prov.get_series(date(2026, 11, 18), date(2026, 11, 20))
    assert [d.date() for d in ser.index] == [date(2026, 11, 18), date(2026, 11, 19)]

    # day not published yet: empty series, NOT an error
    empty = prov.get_series(date(2026, 11, 20), date(2026, 11, 20))
    assert empty.empty


def test_index_not_published_yet_is_not_a_failure(monkeypatch):
    text = _whole_series_csv("2026-11-19")
    monkeypatch.setattr(fred_tdex.requests.Session, "get",
                        lambda self, *a, **k: _Response(text))
    df = FredIndexProvider().get_daily_bars("^VIX", date(2026, 11, 20),
                                            date(2026, 11, 20))
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


@pytest.fixture()
def env(tmp_path):
    p = load_params("params.yaml")
    p["data"]["database"] = str(tmp_path / "m.duckdb")
    db = build_database(p)
    seed_offline_demo(db, etf_symbols(p), date(2019, 1, 2), date(2021, 6, 30))
    yield p, db, build_store(db)
    db.close()


class _WholeSeriesSource:
    """Source that ignores the dates and sends the whole history (like FRED)."""
    name = "fred"

    def __init__(self, df):
        self.df = df

    def get_daily_bars(self, symbol, start, end):
        return self.df.copy()


def test_incremental_does_not_rewrite_the_history(env):
    p, db, store = env
    before = db.con.execute(
        "SELECT COUNT(*), SUM(close) FROM prices_raw "
        "WHERE symbol = '^VIX' AND source = 'demo'").fetchone()

    # the source sends the whole series with different values + 1 new day
    days = pd.bdate_range("2019-01-02", "2021-07-01")
    whole = pd.DataFrame({c: 99.0 for c in ("open", "high", "low", "close")} |
                         {"volume": 0.0}, index=days)
    res = incremental_update(db, _WholeSeriesSource(whole), ["^VIX"],
                             date(2021, 7, 1))
    assert res["written"] == 1                      # only 1 July
    after = db.con.execute(
        "SELECT COUNT(*), SUM(close) FROM prices_raw "
        "WHERE symbol = '^VIX' AND source = 'demo'").fetchone()
    assert after == before                          # history untouched


def test_incremental_reports_not_yet_published(env):
    p, db, store = env
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                         index=pd.DatetimeIndex([]), dtype=float)
    res = incremental_update(db, _WholeSeriesSource(empty), ["^VIX"],
                             date(2021, 7, 1))
    assert res["pending"] == ["^VIX"]
    assert res["failed"] == []
    msg = db.con.execute(
        "SELECT message FROM runs_log ORDER BY started_at DESC LIMIT 1").fetchone()[0]
    assert "not yet published" in msg


# ------------------------------------------------------------------ #
# 2. Signal on the last complete day                                  #
# ------------------------------------------------------------------ #
def test_signal_uses_last_day_with_vix(env):
    p, db, store = env
    rep_ok = build_pipeline(p, db, store).run(date(2021, 6, 29))

    # the ETFs of 30/6 arrive but not VIX and SPX yet
    db.con.execute("DELETE FROM prices_raw WHERE symbol IN ('^VIX', '^SPX') "
                   "AND date = '2021-06-30'")
    rep = build_pipeline(p, db, store).run(date(2021, 6, 30))

    assert rep.as_of.date() == date(2021, 6, 29)
    assert "have not been published yet" in rep.warning
    assert "Tuesday 29/06" in rep.warning
    assert "Thursday 01/07" in rep.warning
    assert rep.quad == rep_ok.quad
    assert rep.asset_signals == rep_ok.asset_signals
    last = db.con.execute(
        "SELECT status, as_of FROM runs_log WHERE kind = 'signals' "
        "ORDER BY started_at DESC LIMIT 1").fetchone()
    assert last[0] == "ok" and last[1] == date(2021, 6, 29)


def test_when_the_vix_arrives_it_realigns(env):
    p, db, store = env
    vix_30 = db.con.execute("SELECT * FROM prices_raw WHERE symbol = '^VIX' "
                            "AND date = '2021-06-30'").fetchdf()
    db.con.execute("DELETE FROM prices_raw WHERE symbol = '^VIX' "
                   "AND date = '2021-06-30'")
    rep1 = build_pipeline(p, db, store).run(date(2021, 6, 30))
    assert rep1.as_of.date() == date(2021, 6, 29)
    assert rep1.warning.startswith("The VIX for 30/06")

    # the VIX comes out in the afternoon: the next run reaches 30/6
    db.con.register("v30", vix_30)
    db.con.execute("INSERT INTO prices_raw SELECT * FROM v30")
    rep2 = build_pipeline(p, db, store).run(date(2021, 6, 30))
    assert rep2.as_of.date() == date(2021, 6, 30)
    assert rep2.warning == ""

    # and it matches the full recompute
    rep3 = build_pipeline(p, db, store).run(date(2021, 6, 30), force_rebuild=True)
    assert rep2.asset_signals == rep3.asset_signals
    assert rep2.quad == rep3.quad


def test_no_warning_if_the_data_is_there(env):
    p, db, store = env
    d, warning = date_with_indices(db, date(2021, 6, 30))
    assert d == date(2021, 6, 30) and warning == ""
