"""Tests of the long-vol recalibration and of the per-quadrant override.

As for majority, the LOGIC tests use FIXED synthetic parameters (TEST_QP),
decoupled from the production set that changes during calibrations. The
INTEGRATION tests use the offline synthetic demo DB.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import shutil

import pandas as pd
import pytest
import yaml

from src.app import (build_longvol_calibrator, build_store, etf_symbols,
                     load_params)
from src.data.database import Database
from src.data.ingest import seed_offline_demo
from src.vix.quad_params import QuadParams
from src.vix.signal_generator import SignalConfig, generate_signals

TEST_QP = {q: QuadParams(7, 8, 19, 2, -0.25, 4.25, 3.5, -3.0)
           for q in (1, 2, 3, 4)}


def _cfg(**kw):
    base = dict(method="majority", tdex_on_thr=-0.5, tdex_lv_thr=-3.0,
                tdex_off_thr=-7.5, use_tdex=True)
    base.update(kw)
    return SignalConfig(**base)


# --------------------------------------------------------------------- #
# Logic of the longvol_per_quad override                                 #
# --------------------------------------------------------------------- #
def test_override_changes_only_the_given_quad():
    # rsi_smooth=-1: with the workbook threshold (-0.25) the RSI long-vol vote is True.
    # The override on Q2 (rsi_long=-2) turns it off ONLY on Q2 days.
    ind = pd.DataFrame([
        {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},
        {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 2},
    ])
    cfg = _cfg(longvol_per_quad={2: {"rsi_long": -2.0}})
    sig = generate_signals(ind, TEST_QP, cfg)
    assert list(sig["vote_rsi_longvol"]) == [True, False]
    # and indeed in Q2 the majority falls (only the VRP is left: 1/3)
    assert list(sig["buy_vix_state"]) == [True, False]


def test_override_does_not_touch_the_market_leg():
    ind = pd.DataFrame([
        {"vrp_smooth": v, "rsi_smooth": r, "tdex_smooth": t, "quad": q}
        for v, r, t, q in [(1, -1, 0, 1), (5, 4, -5, 2), (-4, -2, -1, 3),
                           (0, 4, -1, 4), (2, 1, -4, 2)]
    ])
    base = generate_signals(ind, TEST_QP, _cfg())
    ovr = generate_signals(ind, TEST_QP, _cfg(
        longvol_per_quad={q: {"rsi_long": -9.0, "vrp_long": -9.0}
                          for q in (1, 2, 3, 4)}))
    for col in ("buy_spx_state", "sell_spx_raw", "spx_vrp", "spx_rsi"):
        assert list(base[col]) == list(ovr[col]), col


def test_gate_ignores_the_override():
    # The gate stays the Excel replica: longvol_per_quad must not touch it.
    ind = pd.DataFrame([
        {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 2},
    ])
    gate_base = generate_signals(ind, TEST_QP, SignalConfig(tdex_lv_thr=4.0))
    gate_ovr = generate_signals(ind, TEST_QP, SignalConfig(
        tdex_lv_thr=4.0, longvol_per_quad={2: {"rsi_long": -9.0}}))
    for col in ("buy_vix_state", "sell_vix_raw", "buy_spx_state", "sell_spx_raw"):
        assert list(gate_base[col]) == list(gate_ovr[col]), col


def test_partial_override_uses_workbook_for_the_rest():
    # Override of vrp_long only: rsi_long stays the workbook one.
    ind = pd.DataFrame([
        {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},
    ])
    sig = generate_signals(ind, TEST_QP, _cfg(
        longvol_per_quad={1: {"vrp_long": 0.5}}))
    assert bool(sig["vote_vrp_longvol"].iloc[0]) is False   # 1 < 0.5 no
    assert bool(sig["vote_rsi_longvol"].iloc[0]) is True    # -1 < -0.25 yes


# --------------------------------------------------------------------- #
# Exit semantics in majority (fix for the "hanging position")            #
# --------------------------------------------------------------------- #
def test_consensus_exit_closes_when_the_majority_decays():
    # d0: vrp+tdex vote (rsi dissents) -> 2/3 enters WITH per-leg sell
    # already true. d1: stays 2/3 -> held. d2: only tdex -> 1/3 -> exits.
    ind = pd.DataFrame([
        {"vrp_smooth": 1.0, "rsi_smooth": 1.0, "tdex_smooth": -5.0, "quad": 1},
        {"vrp_smooth": 1.0, "rsi_smooth": 1.0, "tdex_smooth": -5.0, "quad": 1},
        {"vrp_smooth": 9.0, "rsi_smooth": 1.0, "tdex_smooth": -5.0, "quad": 1},
    ])
    sig = generate_signals(ind, TEST_QP, _cfg(vol_exit="consensus"))
    assert list(sig["buy_vix"]) == [True, False, False]
    assert list(sig["sell_vix"]) == [False, False, True]


def test_legs_exit_does_not_leave_hanging_positions():
    # Old defect: 2/3 entry with a dissenting leg => sell_raw (OR) already
    # true => the Excel transition filter never triggered and the position
    # stayed open forever. With the anchoring to the position the exit
    # comes on the next day.
    ind = pd.DataFrame([
        {"vrp_smooth": 1.0, "rsi_smooth": 1.0, "tdex_smooth": -5.0, "quad": 1},
        {"vrp_smooth": 1.0, "rsi_smooth": 1.0, "tdex_smooth": -5.0, "quad": 1},
        {"vrp_smooth": 1.0, "rsi_smooth": 1.0, "tdex_smooth": -5.0, "quad": 1},
    ])
    sig = generate_signals(ind, TEST_QP, _cfg(vol_exit="legs"))
    assert list(sig["sell_vix_raw"]) == [True, True, True]   # rsi always dissents
    assert bool(sig["buy_vix"].iloc[0]) is True
    assert bool(sig["sell_vix"].iloc[1]) is True             # BEFORE: never


def test_state_reentry_reenters_without_new_edge():
    """market_reentry: state — after an exit, if the entry state stays true
    and the exit clears, it RE-ENTERS at once. With the default 'edge' it
    stays out until the state produces a new edge (historical behaviour).
    d0: 3/3 enters. d1: rsi<0 -> exits (the state stays 2/3, no new edge).
    d2: rsi goes back up -> state: re-enters; edge: stays flat."""
    ind = pd.DataFrame([
        {"vrp_smooth": 1.0, "rsi_smooth": 4.0, "tdex_smooth": 0.0, "quad": 1},
        {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},
        {"vrp_smooth": 1.0, "rsi_smooth": 4.0, "tdex_smooth": 0.0, "quad": 1},
    ])
    edge = generate_signals(ind, TEST_QP, _cfg())
    state = generate_signals(ind, TEST_QP, _cfg(market_reentry="state"))
    assert list(edge["pos_spx"]) == [True, False, False]
    assert list(state["pos_spx"]) == [True, False, True]
    assert bool(state["buy_spx"].iloc[2]) is True


def test_state_reentry_does_not_touch_the_pure_gate():
    """The gate WITHOUT market_per_quad stays the Excel replica even with
    market_reentry: state configured."""
    ind = pd.DataFrame([
        {"vrp_smooth": 1.0, "rsi_smooth": 4.0, "tdex_smooth": 0.0, "quad": 1},
        {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},
        {"vrp_smooth": 1.0, "rsi_smooth": 4.0, "tdex_smooth": 0.0, "quad": 1},
    ])
    a = generate_signals(ind, TEST_QP, SignalConfig(tdex_lv_thr=4.0))
    b = generate_signals(ind, TEST_QP, SignalConfig(tdex_lv_thr=4.0,
                                                    market_reentry="state"))
    for col in ("buy_spx", "sell_spx", "pos_spx", "buy_vix", "sell_vix"):
        assert list(a[col]) == list(b[col]), col


def test_gate_excel_events_unchanged():
    # The gate keeps EXACTLY the workbook's transition filter, including the
    # double look-back of the Sell VIX (column L).
    ind = pd.DataFrame([
        {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},  # enters
        {"vrp_smooth": 9.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},  # svr T
        {"vrp_smooth": 9.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},  # svr T
        {"vrp_smooth": 9.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},  # svr T
    ])
    sig = generate_signals(ind, TEST_QP, SignalConfig(tdex_lv_thr=4.0))
    # L = svr & (~svr[-1] | ~svr[-2]): True on the first TWO days of svr
    assert list(sig["sell_vix"]) == [False, True, True, False]



START, END = date(2019, 1, 1), date(2022, 12, 30)


@pytest.fixture(scope="module")
def cal_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cal")
    p = load_params("params.yaml")
    p["vix"]["voting_method"] = "majority"
    db = Database(tmp / "cal.duckdb")
    seed_offline_demo(db, etf_symbols(p), START, END)
    store = build_store(db)
    cal = build_longvol_calibrator(p, store)
    cal.prepare(END)
    yield p, cal
    db.close()


def test_quantile_centres_the_frequency(cal_env):
    _, cal = cal_env
    for f in (15.0, 30.0):
        thr = cal.thresholds_for_frequency("rsi", f, [1, 2, 3, 4])
        q_series = cal.ind["quad"].astype(int)
        for q, t in thr.items():
            vals = cal.ind.loc[q_series == q, "rsi_smooth"].dropna()
            freq = float((vals < t).mean() * 100)
            # rounding to 1 decimal + finite sample: wide tolerance
            assert abs(freq - f) < 12.0, (q, f, freq, t)


def test_sweep_reduces_the_time_in_vol(cal_env):
    _, cal = cal_env
    cands = cal.sweep(["rsi"], [5.0, 60.0], [1, 2, 3, 4])
    labels = [c.label for c in cands]
    assert labels[0] == "current"
    by = {c.label: c.stats for c in cands}
    assert by["f=5%"]["time_in_vol_pct"] <= by["f=60%"]["time_in_vol_pct"]
    for c in cands:
        s = c.stats
        assert 0.0 <= s["time_in_vol_pct"] <= 100.0
        assert s["entries"] >= 0


def test_evaluated_and_configured_match(cal_env):
    """What the calibrator measures = what ends up in the YAML: the same
    evaluation repeated with the explicit overrides gives identical stats."""
    _, cal = cal_env
    ovr = {q: {"rsi_long": v} for q, v in
           cal.thresholds_for_frequency("rsi", 20.0, [2, 4]).items()}
    a = cal.evaluate(ovr, label="x").stats
    b = cal.evaluate(dict(ovr), label="y").stats
    assert a == b


def test_longvol_rejects_the_gate_but_the_calibrator_is_built(cal_env):
    """The constructor accepts the gate (needed by the MARKET leg:
    sweep_market/montecarlo_market = 'calibrated gate'), but the LONG-VOL
    calibration stays majority-only: sweep rejects it."""
    p, _ = cal_env
    import copy
    p2 = copy.deepcopy(p)
    p2["vix"]["voting_method"] = "gate"
    db2 = Database(":memory:")
    store2 = build_store(db2)
    cal2 = build_longvol_calibrator(p2, store2)   # no ValueError here anymore
    with pytest.raises(ValueError):
        cal2.sweep(["rsi"], [10.0], [1, 2, 3, 4])
    db2.close()


# --------------------------------------------------------------------- #
# Surgical writing of the YAML block                                     #
# --------------------------------------------------------------------- #
def test_apply_yaml_inserts_and_replaces(tmp_path):
    from cli import _apply_longvol_yaml
    from src.backtest.calibrate import yaml_block

    dst = tmp_path / "params.yaml"
    shutil.copyfile("params.yaml", dst)

    b1 = yaml_block({2: {"rsi_long": -1.2}, 4: {"rsi_long": -0.8}},
                    "f=20%", "2026-07-04")
    _apply_longvol_yaml(str(dst), b1)
    p1 = yaml.safe_load(dst.read_text(encoding="utf-8"))
    assert p1["vix"]["longvol_per_quad"] == {2: {"rsi_long": -1.2},
                                             4: {"rsi_long": -0.8}}
    # the workbook parameters are NOT touched
    assert p1["vix"]["params_per_quad"][2]["rsi_long"] == 9
    assert (tmp_path / "params.yaml.bak").exists()

    # second application: REPLACES the block, does not duplicate it
    b2 = yaml_block({2: {"rsi_long": -2.0, "vrp_long": 1.5}}, "f=15%",
                    "2026-07-04")
    _apply_longvol_yaml(str(dst), b2)
    text = dst.read_text(encoding="utf-8")
    assert text.count("\n  longvol_per_quad:") == 1
    p2 = yaml.safe_load(text)
    assert p2["vix"]["longvol_per_quad"] == {2: {"rsi_long": -2.0,
                                                 "vrp_long": 1.5}}
    # and the rest of the file stays intact (comments included)
    assert "TDEX THRESHOLDS PER QUADRANT" in text
