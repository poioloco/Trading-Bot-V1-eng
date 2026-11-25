"""TDEX staleness guard.

The TDEX forward-fill must cover short gaps (FRED holidays) but must NOT
project a dead value forever if the feed stops: beyond
`tdex_max_staleness_days` the series goes back to NaN and the leg abstains.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.app import (build_store, etf_symbols, load_params, quads_from_params,
                     regime_params, signal_config, vix_quad_params)
from src.data.database import Database
from src.data.ingest import seed_offline_demo
from src.data.staleness import ffill_with_guard, last_data_lag
from src.engine.regime_engine import StatefulRegime
from src.engine.vix_engine import StatefulVix

START, END = date(2021, 1, 1), date(2022, 12, 30)


# ---------------------------------------------------------------------- #
# unit: the guard function                                                #
# ---------------------------------------------------------------------- #
def _days(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def test_guard_fills_short_gaps():
    idx = _days(10)
    series = pd.Series(np.arange(10.0), index=idx)
    with_gap = series.drop(idx[4:6])          # 2-day gap
    out = ffill_with_guard(with_gap, idx, max_lag_days=10)
    assert out.notna().all()
    assert out.loc[idx[4]] == 3.0 and out.loc[idx[5]] == 3.0   # ffill


def test_guard_abstains_beyond_the_limit():
    idx = _days(30)
    series = pd.Series(1.0, index=idx[:10])   # feed stopped after 10 days
    out = ffill_with_guard(series, idx, max_lag_days=7)
    assert out.iloc[:10].notna().all()
    limit = idx[9] + pd.Timedelta(days=7)
    assert out[out.index > limit].isna().all()
    between = out[(out.index > idx[9]) & (out.index <= limit)]
    assert between.notna().all()


def test_guard_off_is_plain_ffill():
    idx = _days(30)
    series = pd.Series(1.0, index=idx[:5])
    out = ffill_with_guard(series, idx, max_lag_days=None)
    assert out.notna().all()                  # old behaviour


def test_guard_before_the_first_observation_stays_nan():
    idx = _days(10)
    series = pd.Series(1.0, index=idx[5:])
    out = ffill_with_guard(series, idx, max_lag_days=10)
    assert out.iloc[:5].isna().all()          # warm-up: never invented


def test_last_data_lag():
    idx = _days(10)
    series = pd.Series(1.0, index=idx)
    assert last_data_lag(series, idx[-1]) == 0
    assert last_data_lag(series, idx[-1] + pd.Timedelta(days=9)) == 9
    assert last_data_lag(pd.Series(dtype=float), idx[-1]) is None


# ---------------------------------------------------------------------- #
# integration: TDEX feed interrupted -> the leg abstains                  #
# ---------------------------------------------------------------------- #
@pytest.fixture
def env(tmp_path):
    p = load_params("params.yaml")
    db = Database(tmp_path / "stale.duckdb")
    seed_offline_demo(db, etf_symbols(p), START, END)
    yield p, db
    db.close()


def test_tdex_interrupted_abstains_in_warmup(env):
    p, db = env
    # simulate a silent interruption of the FRED feed: no TDEX from October
    db.con.execute("DELETE FROM prices_raw WHERE symbol='TDEX' AND date > '2022-10-01'")
    store = build_store(db)
    reg = StatefulRegime(store, quads_from_params(p), regime_params(p),
                         p["regime"]["ma_windows"], p["regime"]["vote_threshold"])
    vix = StatefulVix(store, vix_quad_params(p), signal_config(p),
                      p["vix"]["spx_symbol"], p["vix"]["vix_symbol"])
    w, _ = reg.warm_up(END)
    ind, sig, _ = vix.warm_up(w, END)
    # within the limit after the last data point: still filled
    assert pd.notna(ind["tdex_smooth"].loc["2022-09-15"])
    # beyond the limit (14 calendar days > 10): abstention (NaN)
    assert pd.isna(ind["tdex_smooth"].loc["2022-10-14"])
    # and it stays that way until the end
    assert pd.isna(ind["tdex_smooth"].iloc[-1])
    # the engine keeps producing signals with the other 2 legs
    assert sig.iloc[-1][["buy_vix", "sell_vix", "buy_spx", "sell_spx"]].notna().all()


def test_tdex_guard_off_projects(env):
    p, db = env
    db.con.execute("DELETE FROM prices_raw WHERE symbol='TDEX' AND date > '2022-10-01'")
    p["vix"]["tdex_max_staleness_days"] = ""          # guard off
    cfg = signal_config(p)
    assert cfg.tdex_stale_days is None
    store = build_store(db)
    reg = StatefulRegime(store, quads_from_params(p), regime_params(p),
                         p["regime"]["ma_windows"], p["regime"]["vote_threshold"])
    vix = StatefulVix(store, vix_quad_params(p), cfg,
                      p["vix"]["spx_symbol"], p["vix"]["vix_symbol"])
    w, _ = reg.warm_up(END)
    ind, _, _ = vix.warm_up(w, END)
    # without the guard the stale value is projected and still votes
    # (after ~21 days the z window becomes constant and goes NaN by itself,
    # but 14 days after the feed stops the stale vote is still there:
    # that is exactly what the guard prevents)
    assert pd.notna(ind["tdex_smooth"].loc["2022-10-14"])
