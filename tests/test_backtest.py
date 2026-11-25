"""Backtest tests.

Checks that the backtest (a) really uses the same code as live, (b) does
not look into the future (walk-forward), (c) produces a consistent
timeline. Offline synthetic data.
"""
from __future__ import annotations

from datetime import date

import pytest

from src.app import (build_store, etf_symbols, load_params, quads_from_params,
                     regime_params, signal_config, vix_quad_params)
from src.backtest import Backtester
from src.data.database import Database
from src.data.ingest import seed_offline_demo
from src.engine.pipeline import DailyPipeline

START, END = date(2019, 1, 1), date(2022, 12, 30)


@pytest.fixture
def bt_env(tmp_path):
    p = load_params("params.yaml")
    db = Database(tmp_path / "bt.duckdb")
    seed_offline_demo(db, etf_symbols(p), START, END)
    store = build_store(db)
    bt = Backtester(
        store=store, quads=quads_from_params(p), regime_params=regime_params(p),
        ma_windows=p["regime"]["ma_windows"], vote_threshold=p["regime"]["vote_threshold"],
        params_by_quad=vix_quad_params(p), cfg=signal_config(p),
        long_asset=p["vix"]["market_asset"], vol_asset=p["vix"]["volatility_asset"],
        spx_symbol=p["vix"]["spx_symbol"], vix_symbol=p["vix"]["vix_symbol"],
    )
    yield p, db, store, bt
    db.close()


def test_backtest_runs_and_has_timeline(bt_env):
    _, _, _, bt = bt_env
    res = bt.run(END)
    assert len(res.timeline) > 200
    assert set(res.timeline["quad"].unique()).issubset({1, 2, 3, 4})
    # positions are 0/1
    assert set(res.timeline["pos_market"].unique()).issubset({0.0, 1.0})


def test_backtest_no_lookahead(bt_env):
    _, _, _, bt = bt_env
    res = bt.run(END)
    idx = res.timeline.index
    sample = [idx[int(len(idx) * f)].date() for f in (0.4, 0.6, 0.8)]
    wf = bt.walk_forward_check(sample, res.timeline)
    assert wf["dates_checked"] >= 2
    assert wf["max_diff_quad"] == 0
    assert wf["mismatched_events"] == 0


def test_backtest_matches_live_pipeline_last_day(bt_env):
    """Same code: the last day of the backtest == the live signal."""
    p, db, store, bt = bt_env
    res = bt.run(END)
    last = res.timeline.iloc[-1]
    pipe = DailyPipeline(
        db=db, store=store, quads=quads_from_params(p), regime_params=regime_params(p),
        ma_windows=p["regime"]["ma_windows"], vote_threshold=p["regime"]["vote_threshold"],
        params_by_quad=vix_quad_params(p), cfg=signal_config(p),
        long_asset=p["vix"]["market_asset"], vol_asset=p["vix"]["volatility_asset"],
        spx_symbol=p["vix"]["spx_symbol"], vix_symbol=p["vix"]["vix_symbol"],
    )
    rep = pipe.run(END)
    assert int(last["quad"]) == rep.quad
    assert abs(float(last["rsi_smooth"]) - rep.metrics["rsi_smooth"]) < 1e-9
    assert abs(float(last["vrp_smooth"]) - rep.metrics["vrp_smooth"]) < 1e-9
