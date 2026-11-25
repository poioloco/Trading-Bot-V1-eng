"""Tests of the market-leg calibration (override, sweep, Monte Carlo)."""
from __future__ import annotations

import copy
from datetime import date

import pytest

from src.app import (build_database, build_longvol_calibrator, build_pipeline,
                     build_store, etf_symbols, load_params, signal_config,
                     vix_quad_params)
from src.data.ingest import seed_offline_demo


@pytest.fixture()
def env(tmp_path):
    p = load_params("params.yaml")
    p["data"]["database"] = str(tmp_path / "m.duckdb")
    p["vix"]["voting_method"] = "majority"
    # these tests are about the CLASSIC mode: independent of params.yaml
    p["vix"]["market_mode"] = "classic"
    db = build_database(p)
    seed_offline_demo(db, etf_symbols(p), date(2018, 1, 2), date(2021, 6, 30))
    yield p, db, build_store(db)
    db.close()


def _prepare(p, store):
    cal = build_longvol_calibrator(p, store)
    cal.prepare(date(2021, 6, 30))
    return cal


def test_lower_exits_reduce_churn(env):
    """Very negative exit thresholds => it almost never exits => time in
    the market and average trade duration rise compared to the Excel zeros."""
    p, db, store = env
    cal = _prepare(p, store)
    zeros = cal.evaluate_market(None, "zeros")
    wide = cal.evaluate_market(
        {q: {"rsi_exit": -99.0, "vrp_exit": -99.0} for q in (1, 2, 3, 4)},
        "wide")
    assert (wide.stats["time_in_market_pct"]
            > zeros.stats["time_in_market_pct"])
    assert wide.stats["avg_duration_days"] > zeros.stats["avg_duration_days"]


def test_calibrated_gate_honours_exits(env):
    """CALIBRATED GATE: with market_per_quad the gate honours the EXIT
    thresholds (anchored to the position); without buy keys the entry and
    the volatility leg stay the Excel replica."""
    from src.vix.signal_generator import generate_signals
    from dataclasses import replace
    p, db, store = env
    cal = _prepare(p, store)
    cfg_gate = replace(signal_config(p), method="gate",
                       market_per_quad=None)
    # "always true" exits (very high thresholds): every position closes at once
    cfg_tight = replace(cfg_gate, market_per_quad={
        q: {"rsi_exit": 99.0, "vrp_exit": 99.0} for q in (1, 2, 3, 4)})
    a = generate_signals(cal.ind, vix_quad_params(p), cfg_gate)
    b = generate_signals(cal.ind, vix_quad_params(p), cfg_tight)
    # market entry and volatility leg: replica unchanged (no buy keys)
    for col in ("buy_spx_state", "buy_vix", "sell_vix", "pos_vix"):
        assert (a[col] == b[col]).all(), f"gate changed on {col}"
    # the exit instead really acts: much less time in the market
    assert a["pos_spx"].sum() > 0
    assert b["pos_spx"].sum() < a["pos_spx"].sum()


def test_calibrated_gate_honours_entries(env):
    """With rsi_buy/vrp_buy keys in the block, the gate ENTRY is calibrated
    too: very permissive thresholds => entry state true more often."""
    from src.vix.signal_generator import generate_signals
    from dataclasses import replace
    p, db, store = env
    cal = _prepare(p, store)
    cfg_gate = replace(signal_config(p), method="gate", market_per_quad=None)
    cfg_easy = replace(cfg_gate, market_per_quad={
        q: {"rsi_buy": -99.0, "vrp_buy": -99.0} for q in (1, 2, 3, 4)})
    a = generate_signals(cal.ind, vix_quad_params(p), cfg_gate)
    b = generate_signals(cal.ind, vix_quad_params(p), cfg_easy)
    assert b["buy_spx_state"].sum() > a["buy_spx_state"].sum()
    # the volatility leg stays intact anyway
    for col in ("buy_vix", "sell_vix", "pos_vix"):
        assert (a[col] == b[col]).all(), col


def test_montecarlo_samples_the_entries(gate_env):
    """include_entries=True: legs extended with rsi_buy/vrp_buy, quantiles in
    the entry range and overrides with the buy keys in the market_per_quad
    block."""
    p, db, store = gate_env
    cal = _prepare(p, store)
    res = cal.montecarlo_market(n=10, objective="calmar", seed=11,
                                include_entries=True)
    assert res["legs"][-2:] == ["rsi_buy", "vrp_buy"]
    c0 = res["candidates"][0]
    for q in (1, 2, 3, 4):
        assert {"rsi_buy", "vrp_buy", "rsi_exit", "vrp_exit"} <= set(c0["overrides"][q])
        assert 10.0 <= c0["quantiles"][q]["rsi_buy"] <= 70.0
        assert 1.0 <= c0["quantiles"][q]["rsi"] <= 30.0
    sens = res["sensitivity"]
    assert len(sens) == 4 * len(res["legs"])


def test_gate_without_override_stays_replica(env):
    """Without a market_per_quad block the gate does NOT change (exact replica)."""
    from src.vix.signal_generator import generate_signals
    from dataclasses import replace
    p, db, store = env
    cal = _prepare(p, store)
    cfg_gate = replace(signal_config(p), method="gate", market_per_quad=None)
    a = generate_signals(cal.ind, vix_quad_params(p), cfg_gate)
    b = generate_signals(cal.ind, vix_quad_params(p), replace(cfg_gate))
    for col in ("buy_spx", "sell_spx", "buy_vix", "sell_vix", "pos_spx"):
        assert (a[col] == b[col]).all(), col


@pytest.fixture()
def gate_env(tmp_path):
    p = load_params("params.yaml")
    p["data"]["database"] = str(tmp_path / "g.duckdb")
    p["vix"]["voting_method"] = "gate"
    p["vix"]["market_per_quad"] = None
    p["vix"]["market_mode"] = "classic"   # tests of the classic mode
    db = build_database(p)
    seed_offline_demo(db, etf_symbols(p), date(2018, 1, 2), date(2021, 6, 30))
    yield p, db, build_store(db)
    db.close()


def test_montecarlo_works_in_gate(gate_env):
    """The Monte Carlo of the exit thresholds also runs in gate mode: the
    'current' reference is the pure gate (Excel replica, zero overrides)."""
    p, db, store = gate_env
    cal = _prepare(p, store)
    res = cal.montecarlo_market(n=10, objective="calmar", seed=7)
    cands = res["candidates"]
    assert len(cands) == 10
    key = res["objective"]
    scores = [c["train"][key] for c in cands]
    assert scores == sorted(scores, reverse=True)
    assert res["current"]["overrides"] == {}
    # the sampled overrides really matter in gate too: at least one
    # candidate must differ from the reference (no "empty" calibration)
    cur = res["current"]["train"]
    assert any(c["train"] != cur for c in cands)
    # reproducibility
    res2 = cal.montecarlo_market(n=10, objective="calmar", seed=7)
    assert res2["candidates"][0]["overrides"] == cands[0]["overrides"]


def test_sweep_market_works_in_gate(gate_env):
    p, db, store = gate_env
    cal = _prepare(p, store)
    cands = cal.sweep_market(["rsi", "vrp"], [5, 20], [1, 2, 3, 4])
    assert [c.label for c in cands] == ["current", "q=5%", "q=20%"]
    # lower quantile = rarer exits = more time in the market
    assert (cands[1].stats["time_in_market_pct"]
            >= cands[2].stats["time_in_market_pct"])


def test_sweep_market_produces_candidates(env):
    p, db, store = env
    cal = _prepare(p, store)
    cands = cal.sweep_market(["rsi", "vrp"], [5, 20], [1, 2, 3, 4])
    assert [c.label for c in cands] == ["current", "q=5%", "q=20%"]
    for c in cands[1:]:
        for q in (1, 2, 3, 4):
            assert set(c.overrides[q]) == {"rsi_exit", "vrp_exit"}
    # lower quantile = rarer exits = more time in the market
    assert (cands[1].stats["time_in_market_pct"]
            >= cands[2].stats["time_in_market_pct"])


def test_montecarlo_sorts_and_validates(env):
    p, db, store = env
    cal = _prepare(p, store)
    res = cal.montecarlo_market(n=12, objective="calmar", seed=7)
    cands = res["candidates"]
    assert len(cands) == 12
    key = res["objective"]
    scores = [c["train"][key] for c in cands]
    assert scores == sorted(scores, reverse=True)   # best first
    # every candidate has disjoint train and test and complete metrics
    for c in cands[:3]:
        for phase in ("train", "test"):
            for k in ("multiple", "max_dd_pct", "calmar", "sharpe",
                      "time_in_market_pct"):
                assert k in c[phase]
    assert res["stability"]["worst_neighbour"] is not None
    assert res["split"]["train_to"] < res["split"]["test_from"]
    # reproducibility: same seed, same winner
    res2 = cal.montecarlo_market(n=12, objective="calmar", seed=7)
    assert res2["candidates"][0]["overrides"] == cands[0]["overrides"]


def test_montecarlo_sensitivity_and_recommended(env):
    """The sensitivity gives a number for every threshold (4 quads x sampled
    legs: rsi, vrp and possibly tdex), sorted by effect, and the recommended
    config is evaluated on train+test."""
    p, db, store = env
    cal = _prepare(p, store)
    res = cal.montecarlo_market(n=30, objective="calmar", seed=3)
    legs = res["legs"]
    assert legs[:2] == ["rsi", "vrp"]
    sens = res["sensitivity"]
    assert len(sens) == 4 * len(legs)
    effects = [s["effect"] for s in sens]
    assert effects == sorted(effects, reverse=True)   # most important first
    for s in sens:
        assert s["leg"] in legs
        assert s["strength"] in {"STRONG", "medium", "weak"}
        assert [f["name"] for f in s["bands"]] == ["low", "mid", "high"]
        assert sum(f["n"] for f in s["bands"]) == 30
        c = s["recommended"]
        assert 1.0 <= c["p25"] <= c["median"] <= c["p75"] <= 30.0
    rec = res["recommended"]
    assert set(rec) == {"quantiles", "overrides", "tdex_sell", "train", "test"}
    for phase in ("train", "test"):
        for k in ("multiple", "cagr_pct", "max_dd_pct", "calmar", "sharpe"):
            assert k in rec[phase]
    # the recommended one uses exactly the sensitivity medians
    for s in sens:
        assert rec["quantiles"][s["quad"]][s["leg"]] == s["recommended"]["median"]
    # if tdex is sampled, every candidate carries its 4 sell_spx
    if "tdex" in legs:
        assert set(rec["tdex_sell"]) == {1, 2, 3, 4}
        assert set(res["candidates"][0]["tdex_sell"]) == {1, 2, 3, 4}
    else:
        assert rec["tdex_sell"] == {}


def test_apply_tdex_sell_yaml(tmp_path):
    """Updates only the sell_spx: longvol/buy_spx of the block stay intact."""
    import shutil
    import yaml
    from cli import _apply_tdex_sell_yaml

    dst = tmp_path / "params.yaml"
    shutil.copyfile("params.yaml", dst)
    before = yaml.safe_load(dst.read_text(encoding="utf-8"))["vix"]["tdex_per_quad"]
    _apply_tdex_sell_yaml(str(dst), {1: -5.5, 3: -2.2})
    after = yaml.safe_load(dst.read_text(encoding="utf-8"))["vix"]["tdex_per_quad"]
    assert after[1]["sell_spx"] == -5.5 and after[3]["sell_spx"] == -2.2
    assert after[2]["sell_spx"] == before[2]["sell_spx"]
    for q in (1, 2, 3, 4):
        assert after[q]["longvol"] == before[q]["longvol"]
        assert after[q]["buy_spx"] == before[q]["buy_spx"]
    # idempotent on the structure: a second application does not duplicate the block
    _apply_tdex_sell_yaml(str(dst), {2: -9.0})
    text = dst.read_text(encoding="utf-8")
    assert text.count("tdex_per_quad:") == 1


def test_apply_market_yaml(tmp_path):
    import shutil
    import yaml
    from cli import _apply_market_yaml
    from src.backtest.calibrate import yaml_block_market

    dst = tmp_path / "params.yaml"
    shutil.copyfile("params.yaml", dst)
    ovr = {2: {"rsi_exit": -1.5, "vrp_exit": -3.2}, 4: {"rsi_exit": -0.8}}
    b1 = yaml_block_market(ovr, "montecarlo #1", "2026-07-16")
    _apply_market_yaml(str(dst), b1)
    p1 = yaml.safe_load(dst.read_text(encoding="utf-8"))
    assert p1["vix"]["market_per_quad"] == ovr
    # second application: REPLACES the block, does not duplicate it
    b2 = yaml_block_market({1: {"rsi_exit": -9.9}}, "montecarlo #2",
                           "2026-07-16")
    _apply_market_yaml(str(dst), b2)
    p2 = yaml.safe_load(dst.read_text(encoding="utf-8"))
    assert p2["vix"]["market_per_quad"] == {1: {"rsi_exit": -9.9}}
    # the longvol and the workbook parameters are not touched
    assert p2["vix"].get("params_per_quad") == p1["vix"].get("params_per_quad")


def test_pipeline_rebuilds_after_config_change(env):
    """After an applied calibration (config changed) the signals run must
    throw away the incremental state and rebuild, without manual action."""
    p, db, store = env
    rep1 = build_pipeline(p, db, store).run(date(2021, 6, 30))
    assert rep1.mode == "warm-up"

    q = copy.deepcopy(p)
    q["vix"]["market_per_quad"] = {1: {"rsi_exit": -2.0}}
    rep2 = build_pipeline(q, db, store).run(date(2021, 6, 30))
    assert rep2.mode == "rebuild"

    # same config again: no rebuild
    rep3 = build_pipeline(q, db, store).run(date(2021, 6, 30))
    assert rep3.mode == "up to date"
