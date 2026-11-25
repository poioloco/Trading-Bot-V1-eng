"""Equivalence tests: the incremental update (warm-up + steps) must give the
SAME numbers as the full recompute from the first day.

They use offline synthetic data (no network), but exercise exactly the
same code as the real path.
"""
from __future__ import annotations

from datetime import date

import pytest

from src.app import (build_store, etf_symbols, load_params, quads_from_params,
                     regime_params, signal_config, vix_quad_params)
from src.data.database import Database
from src.data.ingest import seed_offline_demo
from src.engine.pipeline import DailyPipeline
from src.engine.regime_engine import StatefulRegime
from src.engine.vix_engine import StatefulVix

START, MID, END = date(2021, 1, 1), date(2022, 6, 30), date(2022, 12, 30)


@pytest.fixture
def env(tmp_path):
    p = load_params("params.yaml")
    db = Database(tmp_path / "eq.duckdb")
    seed_offline_demo(db, etf_symbols(p), START, END)
    store = build_store(db)
    yield p, db, store
    db.close()


def test_regime_incremental_matches_full(env):
    p, db, store = env
    reg = StatefulRegime(store, quads_from_params(p), regime_params(p),
                         p["regime"]["ma_windows"], p["regime"]["vote_threshold"])
    full, _ = reg.warm_up(END)
    _, state = reg.warm_up(MID)
    new, _ = reg.step(MID, state, END)
    common = full.index.intersection(new.index)
    assert len(common) > 20
    # winner is an integer 1..4: it must match exactly
    assert (full.loc[common] - new.loc[common]).abs().max() == 0


def test_vix_incremental_matches_full(env):
    p, db, store = env
    reg = StatefulRegime(store, quads_from_params(p), regime_params(p),
                         p["regime"]["ma_windows"], p["regime"]["vote_threshold"])
    vix = StatefulVix(store, vix_quad_params(p), signal_config(p),
                      p["vix"]["spx_symbol"], p["vix"]["vix_symbol"])
    wf, _ = reg.warm_up(END)
    ind_full, sig_full, _ = vix.warm_up(wf, END)

    wm, _ = reg.warm_up(MID)
    _, _, vstate = vix.warm_up(wm, MID)
    wn, _ = reg.step(MID, reg.warm_up(MID)[1], END)
    ind_step, sig_step, _ = vix.step(MID, vstate, wn, END)

    common = ind_full.index.intersection(ind_step.index)
    assert len(common) > 20
    for col in ("rsi_smooth", "vrp_smooth", "tdex_smooth"):
        assert (ind_full.loc[common, col] - ind_step.loc[common, col]).abs().max() < 1e-6
    for col in ("buy_vix", "sell_vix", "buy_spx", "sell_spx",
                "pos_vix", "pos_spx"):
        a = sig_full.loc[common, col].astype(bool)
        b = sig_step.loc[common, col].astype(bool)
        assert (a != b).sum() == 0


def test_pipeline_warmup_then_latest_is_stable(env):
    p, db, store = env
    pipe = DailyPipeline(
        db=db, store=store, quads=quads_from_params(p), regime_params=regime_params(p),
        ma_windows=p["regime"]["ma_windows"], vote_threshold=p["regime"]["vote_threshold"],
        params_by_quad=vix_quad_params(p), cfg=signal_config(p),
        long_asset=p["vix"]["market_asset"], vol_asset=p["vix"]["volatility_asset"],
        spx_symbol=p["vix"]["spx_symbol"], vix_symbol=p["vix"]["vix_symbol"],
    )
    r1 = pipe.run(END)                 # warm-up
    r2 = pipe.run(END)                 # from state, no new day
    assert r1.mode == "warm-up" and r2.mode == "up to date"
    assert r1.quad == r2.quad
    assert r1.asset_signals == r2.asset_signals
    assert abs(r1.metrics["rsi_smooth"] - r2.metrics["rsi_smooth"]) < 1e-9
    assert abs(r1.metrics["vrp_smooth"] - r2.metrics["vrp_smooth"]) < 1e-9


def test_state_is_persisted_and_small(env):
    p, db, store = env
    pipe = DailyPipeline(
        db=db, store=store, quads=quads_from_params(p), regime_params=regime_params(p),
        ma_windows=p["regime"]["ma_windows"], vote_threshold=p["regime"]["vote_threshold"],
        params_by_quad=vix_quad_params(p), cfg=signal_config(p),
        long_asset=p["vix"]["market_asset"], vol_asset=p["vix"]["volatility_asset"],
    )
    pipe.run(END)
    rows = db.con.execute(
        "SELECT engine, COUNT(*) FROM indicator_state GROUP BY engine ORDER BY engine"
    ).fetchall()
    engines = {r[0]: r[1] for r in rows}
    assert engines == {"regime": 1, "vix": 1}   # a single state (the latest) per engine
