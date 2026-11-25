"""Tests of the V1 data layer.

The most important test is the POINT-IN-TIME GUARANTEE: the as-of-T view
must not change if data after T is added later. It is the proof, from the
bottom up, that the backtest cannot use data from the future.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.calendar import (
    current_trading_date,
    is_session,
    last_session_on_or_before,
    missing_sessions,
    sessions_between,
)
from src.data.database import Database
from src.data.ingest import seed_offline_demo
from src.data.store import PointInTimeStore
from src.data.validation import validate_bars


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "t.duckdb")
    yield d
    d.close()


def _bars(start_px, n, idx):
    close = np.linspace(start_px, start_px * 1.1, n)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000.0},
        index=idx,
    )


# --------------------------------------------------------------------- #
# Calendar                                                              #
# --------------------------------------------------------------------- #
def test_calendar_holidays_and_weekends():
    assert is_session(date(2025, 12, 25)) is False        # Christmas
    assert is_session(date(2025, 7, 4)) is False          # Independence Day
    assert is_session(date(2025, 6, 14)) is False         # Saturday
    assert is_session(date(2025, 6, 13)) is True          # Friday


def test_last_session_and_missing():
    assert last_session_on_or_before(date(2025, 12, 25)) == date(2025, 12, 24)
    s = sessions_between(date(2025, 12, 22), date(2025, 12, 26))
    assert date(2025, 12, 25) not in s and date(2025, 12, 24) in s
    miss = missing_sessions([date(2025, 12, 24)], date(2025, 12, 22), date(2025, 12, 26))
    assert date(2025, 12, 24) not in miss and date(2025, 12, 23) in miss


def test_current_trading_date_after_close_vs_before():
    # 22:00 UTC = 17:00 NY (after the close) on a market Friday
    after = current_trading_date(pd.Timestamp("2025-06-13 22:00", tz="UTC"))
    assert after == date(2025, 6, 13)
    # 13:00 UTC = 09:00 NY (market not closed yet) -> previous session
    before = current_trading_date(pd.Timestamp("2025-06-13 13:00", tz="UTC"))
    assert before == date(2025, 6, 12)


# --------------------------------------------------------------------- #
# Validation                                                            #
# --------------------------------------------------------------------- #
def test_validation_drops_bad_and_incoherent():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    df = _bars(100, 5, idx)
    df.loc[idx[1], "close"] = -1.0          # invalid price
    df.loc[idx[3], "high"] = 1.0            # high < low: inconsistent
    res = validate_bars(df, "TEST")
    assert res.dropped == 2
    assert idx[1] not in res.clean.index and idx[3] not in res.clean.index


# --------------------------------------------------------------------- #
# Corporate action adjustment (at read time)                          #
# --------------------------------------------------------------------- #
def test_split_adjustment_backadjusted(db):
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    df = _bars(100, 6, idx)
    db.upsert_prices(df, "ABC", "test")
    # 2:1 split in the middle: the prices BEFORE are halved, the last one stays raw
    ca = pd.DataFrame([{"symbol": "ABC", "date": idx[3].date(),
                        "action_type": "split", "value": 2.0}])
    db.upsert_corporate_actions(ca, "test")
    store = PointInTimeStore(db)
    adj = store.get_bars_asof("ABC", idx[-1].date())
    raw = store.get_bars_asof("ABC", idx[-1].date(), adjusted=False)
    # last close: unchanged; first close: halved
    assert adj["close"].iloc[-1] == pytest.approx(raw["close"].iloc[-1])
    assert adj["close"].iloc[0] == pytest.approx(raw["close"].iloc[0] / 2.0)


# --------------------------------------------------------------------- #
# POINT-IN-TIME GUARANTEE (the anti-lookahead core)                     #
# --------------------------------------------------------------------- #
def test_point_in_time_no_future_rows(db):
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    db.upsert_prices(_bars(50, 10, idx), "XYZ", "test")
    store = PointInTimeStore(db)
    t = idx[4].date()
    asof = store.get_bars_asof("XYZ", t)
    assert asof.index.max().date() == t           # never rows after T
    assert len(asof) == 5


def test_asof_view_is_stable_when_future_data_arrives(db):
    """The as-of-T view must NOT change if data after T is added."""
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    db.upsert_prices(_bars(50, 6, idx), "XYZ", "test")
    store = PointInTimeStore(db)
    t = idx[3].date()
    before = store.get_bars_asof("XYZ", t).copy()
    # new days arrive AFTER T
    new_idx = pd.date_range("2025-01-07", periods=4, freq="D")
    db.upsert_prices(_bars(60, 4, new_idx), "XYZ", "test")
    after = store.get_bars_asof("XYZ", t)
    pd.testing.assert_frame_equal(before, after)


def test_future_corporate_action_not_applied_before_exdate(db):
    """A corporate action with ex-date > T must not touch the as-of-T view."""
    idx = pd.date_range("2025-01-01", periods=8, freq="D")
    db.upsert_prices(_bars(100, 8, idx), "ABC", "test")
    ca = pd.DataFrame([{"symbol": "ABC", "date": idx[6].date(),
                        "action_type": "split", "value": 2.0}])
    db.upsert_corporate_actions(ca, "test")
    store = PointInTimeStore(db)
    # T before the split: no adjustment
    early = store.get_bars_asof("ABC", idx[4].date())
    raw_early = store.get_bars_asof("ABC", idx[4].date(), adjusted=False)
    pd.testing.assert_series_equal(early["close"], raw_early["close"])


# --------------------------------------------------------------------- #
# Write idempotency                                                     #
# --------------------------------------------------------------------- #
def test_upsert_is_idempotent(db):
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    df = _bars(100, 5, idx)
    db.upsert_prices(df, "ABC", "test")
    db.upsert_prices(df, "ABC", "test")  # same range twice
    cnt = db.con.execute("SELECT COUNT(*) FROM prices_raw WHERE symbol='ABC'").fetchone()[0]
    assert cnt == 5


def test_signal_dedup(db):
    assert db.write_signal(date(2025, 6, 13), "TQQQ", "BUY", 2) is True
    assert db.write_signal(date(2025, 6, 13), "TQQQ", "BUY", 2) is False  # already written


# --------------------------------------------------------------------- #
# Demo seed end-to-end + store as DataProvider                          #
# --------------------------------------------------------------------- #
def test_demo_seed_and_store_as_provider(db):
    etfs = ["AAA", "BBB", "CCC"]
    info = seed_offline_demo(db, etfs, date(2020, 1, 1), date(2021, 1, 1))
    assert info["sessions"] > 200
    store = PointInTimeStore(db)
    bars = store.get_daily_bars("AAA", date(2020, 1, 1), date(2020, 6, 1))
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.max() <= pd.Timestamp("2020-06-01")
    assert db.last_successful_run("demo_seed") is not None


# --------------------------------------------------------------------- #
# User history importer + incremental update                          #
# --------------------------------------------------------------------- #
def test_importer_folder_with_aliases(db, tmp_path):
    from src.data.importer import import_path
    idx = pd.bdate_range("2005-01-03", periods=50)
    c = np.linspace(50, 60, 50)
    pd.DataFrame({"Date": idx, "Open": c, "High": c * 1.01, "Low": c * 0.99,
                  "Close": c, "Volume": 1e6}).to_csv(tmp_path / "TLT.csv", index=False)
    pd.DataFrame({"Date": idx, "Close": c}).to_csv(tmp_path / "SPX.csv", index=False)
    info = import_path(db, tmp_path)
    assert "^SPX" in db.symbols() and "TLT" in db.symbols()   # alias normalised
    assert info["total"] == 100
    src = db.con.execute("SELECT DISTINCT source FROM prices_raw").fetchdf()
    assert src["source"].tolist() == ["import"]


def test_demo_seed_blocked_when_real_data_present(db, tmp_path):
    """The DEMO seed must not overwrite imported real data."""
    from src.data.importer import import_path
    from src.data.ingest import seed_offline_demo
    idx = pd.bdate_range("2005-01-03", periods=30)
    pd.DataFrame({"Date": idx, "Close": np.linspace(10, 12, 30)}).to_csv(
        tmp_path / "GLD.csv", index=False)
    import_path(db, tmp_path)
    with pytest.raises(RuntimeError):
        seed_offline_demo(db, ["GLD"], date(2005, 1, 1), date(2005, 3, 1))


def test_importer_excel_with_banner_row(db, tmp_path):
    """Excel with a 'banner' row above the real column names (like the regime
    workbook: row1=ticker, row2=Date/Open/HOD/LOD/Close)."""
    from src.data.importer import import_path
    idx = pd.bdate_range("2005-01-03", periods=80)
    c = np.linspace(80, 90, 80)
    body = pd.DataFrame({"Date": idx, "Open": c, "HOD": c * 1.01,
                         "LOD": c * 0.99, "Close": c})
    xlsx = tmp_path / "model.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        # write a banner row, then the real header, then the data
        banner = pd.DataFrame([["TLT", "TLT", "TLT", "TLT", "TLT"]])
        banner.to_excel(w, sheet_name="TLT", header=False, index=False, startrow=0)
        body.to_excel(w, sheet_name="TLT", index=False, startrow=1)
        body.to_excel(w, sheet_name="Macro", index=False)   # unwanted sheet
    info = import_path(db, xlsx, only_symbols={"TLT"})
    assert "TLT" in db.symbols() and "MACRO" not in db.symbols()
    assert info["symbols"]["TLT"] == 80
