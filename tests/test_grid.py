"""Trend mode of the market leg and grid calibration."""
from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.app import (build_database, build_longvol_calibrator, build_store,
                     etf_symbols, load_params, quads_from_params, regime_params,
                     signal_config, vix_quad_params)
from src.backtest.grid import GRID, NG, GridCalibrator, apply_yaml
from src.data.ingest import seed_offline_demo
from src.vix.signal_generator import TREND, generate_signals

START, TRAIN = "2018-11-01", "2020-06-01"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("grid")
    p = load_params("params.yaml")
    p["data"]["database"] = str(tmp / "m.duckdb")
    db = build_database(p)
    seed_offline_demo(db, etf_symbols(p), date(2018, 1, 2), date(2021, 6, 30))
    store = build_store(db)
    cal = build_longvol_calibrator(p, store)
    cal.prepare(date(2021, 6, 30))
    yield p, cal
    db.close()


def _grid_cal(p, cal):
    return GridCalibrator(cal.ind, cal.params_by_quad, signal_config(p),
                          start=START, train_start=TRAIN)


def _random(rng):
    ent = {q: {"rsi_buy": float(rng.choice(GRID)), "vrp_buy": float(rng.choice(GRID)),
               "tdex_buy": float(rng.choice(GRID))} for q in (1, 2, 3, 4)}
    ext = {q: {"rsi": float(rng.choice(GRID)), "vrp": float(rng.choice(GRID))}
           for q in (1, 2, 3, 4)}
    return ent, ext


def test_classic_mode_ignores_the_trend_blocks(env):
    """In classic mode the trend-mode blocks change NOTHING."""
    p, cal = env
    cfg = replace(signal_config(p), market_mode="classic",
                  trend_entries_per_quad=None, trend_exits_per_quad=None)
    rng = np.random.default_rng(0)
    ent, ext = _random(rng)
    cfg2 = replace(cfg, trend_entries_per_quad=ent, trend_exits_per_quad=ext)
    a = generate_signals(cal.ind, vix_quad_params(p), cfg)
    b = generate_signals(cal.ind, vix_quad_params(p), cfg2)
    pd.testing.assert_frame_equal(a, b)


def test_trend_mode_logic_by_hand(env):
    """Enters with E and no veto; exits on the veto; does NOT re-enter by
    itself when the veto ends: it waits for the entry rule to be true again."""
    p, _ = env
    idx = pd.bdate_range("2024-01-01", periods=6)
    ind = pd.DataFrame({
        "quad":        [1, 1, 1, 1, 1, 1],
        "rsi_smooth":  [2.0, -1.0, -1.0, 2.0, 2.0, 2.0],
        "vrp_smooth":  [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "tdex_smooth": [0.0, 0.0, 0.0, 0.0, 0.0, -3.0],
        "spx_close":   [100, 90, 101, 102, 103, 104.0],
        "spx_sma":     [95, 95, 95, 95, 95, 95.0],
    }, index=idx)
    cfg = replace(signal_config(p), market_mode=TREND, trend_enabled=True,
                  trend_entries_per_quad={1: {"rsi_buy": 0, "vrp_buy": 0, "tdex_buy": -1}},
                  trend_exits_per_quad={1: {"rsi": 0, "vrp": 0}})
    s = generate_signals(ind, vix_quad_params(p), cfg)
    assert s["pos_spx"].tolist() == [True, False, False, True, True, True]
    assert s["buy_spx"].tolist() == [True, False, False, True, False, False]
    assert s["sell_spx"].tolist() == [False, True, False, False, False, False]
    # with the same history but an RSI confirmation threshold of -2, day 1 is not a veto
    cfg2 = replace(cfg, trend_exits_per_quad={1: {"rsi": -2, "vrp": 0}})
    assert generate_signals(ind, vix_quad_params(p), cfg2)["pos_spx"].all()


def test_fast_engine_equals_the_real_engine(env):
    p, cal = env
    g = _grid_cal(p, cal)
    rng = np.random.default_rng(1)
    for _ in range(25):
        ent, ext = _random(rng)
        E, V = g._ev(ent, ext)
        fast = g._positions(E, V)[0]
        real = generate_signals(g.ind, g.pq, g.cfg_with(ent, ext))["pos_spx"].to_numpy(dtype=bool)
        assert np.array_equal(fast, real)


@pytest.mark.parametrize("part", ["entries", "exits"])
def test_calibrate_improves_and_stays_on_the_grid(env, part):
    p, cal = env
    g = _grid_cal(p, cal)
    res = g.calibrate(part, max_rounds=4)
    k = res["key"]
    assert (res["references"]["calibrated"]["train"][k]
            >= res["references"]["start"]["train"][k] - 1e-12)
    for q, d in res["final"][part].items():
        for v in d.values():
            assert -10.0 <= v <= 10.0 and abs(v * 10 - round(v * 10)) < 1e-9
    tot = NG ** 3 if part == "entries" else NG ** 2
    for q in (1, 2, 3, 4):
        assert int(res["per_quad"][q]["table"]["combinations"].sum()) == tot
        assert res["per_quad"][q]["table"]["chosen"].sum() == 1
    assert res["verification"]["engine_cases"] >= 1 and res["verification"]["dedup_cases"] > 0
    # the final thresholds, put back into the real engine, give the declared metrics
    m = g.config_metrics(res["final"]["entries"], res["final"]["exits"])
    assert m["train"][k] == pytest.approx(
        res["references"]["calibrated"]["train"][k])


def test_apply_yaml(tmp_path):
    dst = tmp_path / "params.yaml"
    shutil.copyfile("params.yaml", dst)
    ent = {q: {"rsi_buy": 1.5, "vrp_buy": -0.5, "tdex_buy": -2.0} for q in (1, 2, 3, 4)}
    apply_yaml(str(dst), "entries", ent, "test", "2026-10-14")
    assert (tmp_path / "params.yaml.bak").exists()
    p = load_params(dst)
    cfg = signal_config(p)
    assert cfg.market_mode == "trend" and cfg.trend_enabled
    assert {int(k): v for k, v in cfg.trend_entries_per_quad.items()} == ent
    ent2 = {q: {"rsi_buy": 0.1, "vrp_buy": 0.2, "tdex_buy": 0.3} for q in (1, 2, 3, 4)}
    apply_yaml(str(dst), "entries", ent2, "test 2", "2026-10-14")
    ext = {q: {"rsi": -1.0, "vrp": 2.5} for q in (1, 2, 3, 4)}
    apply_yaml(str(dst), "exits", ext, "test", "2026-10-14")
    text = dst.read_text(encoding="utf-8")
    assert text.count("trend_entries_per_quad:") == 1
    assert text.count("trend_exits_per_quad:") == 1
    import re
    assert len(re.findall(r"^[ \t]*market_mode:", text, re.MULTILINE)) == 1
    cfg = signal_config(load_params(dst))
    assert {int(k): v for k, v in cfg.trend_entries_per_quad.items()} == ent2
    assert {int(k): v for k, v in cfg.trend_exits_per_quad.items()} == ext


def test_incremental_equals_full_in_trend_mode(env):
    """The live engine (daily steps) must give the same positions as the
    full recompute in trend mode too."""
    from src.engine.regime_engine import StatefulRegime
    from src.engine.vix_engine import StatefulVix
    p, cal = env
    store = cal.vix.store
    p2 = dict(p)
    p2["vix"] = dict(p["vix"], market_mode="trend",
                     trend_entries_per_quad={q: {"rsi_buy": 0.5, "vrp_buy": 0.0, "tdex_buy": -1.0}
                                             for q in (1, 2, 3, 4)},
                     trend_exits_per_quad={q: {"rsi": 1.0, "vrp": 0.5} for q in (1, 2, 3, 4)})
    cfg = signal_config(p2)
    reg = StatefulRegime(store, quads_from_params(p2), regime_params(p2),
                         p2["regime"]["ma_windows"], p2["regime"]["vote_threshold"])
    vix = StatefulVix(store, vix_quad_params(p2), cfg, p2["vix"]["spx_symbol"],
                      p2["vix"]["vix_symbol"])
    end, mid = date(2021, 6, 30), date(2020, 9, 30)
    wf, _ = reg.warm_up(end)
    _, sig_full, _ = vix.warm_up(wf, end)
    wm, rstate = reg.warm_up(mid)
    _, _, vstate = vix.warm_up(wm, mid)
    wn, _ = reg.step(mid, rstate, end)
    _, sig_step, _ = vix.step(mid, vstate, wn, end)
    common = sig_full.index.intersection(sig_step.index)
    assert len(common) > 20
    assert sig_full["pos_spx"].any() and (~sig_full["pos_spx"].astype(bool)).any()
    for col in ("buy_spx", "sell_spx", "pos_spx"):
        a = sig_full.loc[common, col].astype(bool)
        b = sig_step.loc[common, col].astype(bool)
        assert (a != b).sum() == 0, col


def test_calibrate_grid_command(env, tmp_path, monkeypatch):
    """The menu command runs end to end and writes CSV + report."""
    import cli
    p, _ = env
    dst = tmp_path / "params.yaml"
    shutil.copyfile("params.yaml", dst)
    monkeypatch.chdir(tmp_path)
    rc = cli.cmd_calibrate_grid(p, "2021-06-30", "exits", "yes", str(dst),
                                START, TRAIN, open_browser=False)
    assert rc == 0
    assert (tmp_path / "output" / "grid_exits.html").stat().st_size > 10_000
    assert (tmp_path / "output" / "grid_exits.csv").exists()
    assert signal_config(load_params(dst)).market_mode == "trend"


# ---------------------------------------------------------------------- #
# Tie-break among equivalent thresholds                                   #
# ---------------------------------------------------------------------- #
from src.backtest.grid import (_axis_margins, _innermost_point,  # noqa: E402
                               _depth_of, TIEBREAKS)


def _box(*axes):
    """All the points of the Cartesian product of the given ranges."""
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


def test_innermost_point_synthetic_shapes():
    # 5x5x5 cube: the centre, margin 2
    assert _innermost_point(_box(range(10, 15), range(20, 25), range(30, 35))) == ((12, 22, 32), 2)
    # 4x4x4 cube (even side): 8 equivalent centres -> the current threshold decides
    cube4 = _box(range(0 + 60, 64), range(60, 64), range(60, 64))
    assert _innermost_point(cube4, (63, 63, 63)) == ((62, 62, 62), 1)
    assert _innermost_point(cube4, (60, 60, 60)) == ((61, 61, 61), 1)
    # L 3 wide: the inner points are on the centre line; the one closest to
    # the barycentre wins (the inner corner), margin 1
    L = np.array([(i, j) for i in range(9) for j in range(3)]
                 + [(i, j) for i in range(3) for j in range(3, 9)])
    pt, m = _innermost_point(L)
    assert m == 1 and pt == (2, 2) and _depth_of(L, pt) == 1
    # isolated point: margin 0
    assert _innermost_point(np.array([[7, 8]])) == ((7, 8), 0)
    # the edge of the grid counts as an edge: free from -10 to 0 -> -5
    R = _box(range(0, 101), range(95, 106))
    assert _innermost_point(R) == ((50, 100), 5)
    assert GRID[50] == -5.0
    inside = np.zeros((NG, NG), dtype=bool)
    inside[tuple(R.T)] = True
    assert _axis_margins(inside, (50, 100)) == [(50, 50), (5, 5)]
    assert _axis_margins(inside, (0, 95)) == [(0, 100), (0, 10)]
    # zero width on one axis (fixed VRP): depth 0 everywhere, the centre of
    # the rectangle on the other two axes is chosen
    Q1 = _box(range(0, 101), [103], range(0, 82))
    pt, m = _innermost_point(Q1, (63, 103, 58))
    assert m == 0 and pt[0] == 50 and pt[1] == 103 and pt[2] in (40, 41)


def _explorations(g, part):
    ent, ext = g.current_thresholds()
    for q in (1, 2, 3, 4):
        exp = (g._explore_entries(q, ent, ext) if part == "entries"
               else g._explore_exits(q, ent, ext))
        yield q, ent, ext, exp


@pytest.mark.parametrize("part", ["entries", "exits"])
def test_recentring_leaves_the_metrics_identical(env, part):
    """Starting from the EDGE of the winning plateau, the robust tie-break
    recentres without changing even a single day of position."""
    p, cal = env
    g = _grid_cal(p, cal)
    assert g.tiebreak == "robust"
    recentred = 0
    for q, ent, ext, exp in _explorations(g, part):
        obj = exp["metrics"]["train"][g.key]
        ties = np.flatnonzero(obj >= np.nanmax(obj) - 1e-12)
        u = g._deepest(exp, ties, (100,) * len(exp["shape"]))
        members = np.flatnonzero(exp["inv"] == u)
        coords = np.stack(np.unravel_index(members, exp["shape"]), axis=1)
        edge = coords[0]                        # first in order: it is on the edge
        cur = {k: float(GRID[i]) for k, i in zip(exp["keys"], edge)}
        e2 = {k: dict(v) for k, v in ent.items()}
        x2 = {k: dict(v) for k, v in ext.items()}
        (e2 if part == "entries" else x2)[q] = cur
        before = g.config_metrics(e2, x2)
        chosen, action = g._choose(exp, cur, before["train"][g.key])
        assert action in ("recentre", "keep"), (q, action)
        (e2 if part == "entries" else x2)[q] = chosen
        after = g.config_metrics(e2, x2)
        for f in before:
            for m in before[f]:
                assert after[f][m] == pytest.approx(before[f][m], rel=0, abs=1e-12), (q, f, m)
        if action == "recentre":
            recentred += 1
    assert recentred >= 1


@pytest.mark.parametrize("part", ["entries", "exits"])
def test_choice_among_ties_does_not_depend_on_the_start(env, part):
    """Whatever training-tied outcome it starts from, robust chooses the
    same plateau (the deepest) and the training does not change."""
    p, cal = env
    g = _grid_cal(p, cal)
    for q, ent, ext, exp in _explorations(g, part):
        obj = exp["metrics"]["train"][g.key]
        ties = np.flatnonzero(obj >= np.nanmax(obj) - 1e-12)
        choices = set()
        for u in ties[:6]:
            first = np.flatnonzero(exp["inv"] == u)[0]
            cur = {k: float(GRID[i]) for k, i in
                   zip(exp["keys"], np.unravel_index(first, exp["shape"]))}
            e2 = {k: dict(v) for k, v in ent.items()}
            x2 = {k: dict(v) for k, v in ext.items()}
            (e2 if part == "entries" else x2)[q] = cur
            before = g.config_metrics(e2, x2)
            chosen, action = g._choose(exp, cur, before["train"][g.key])
            assert action in ("keep", "recentre", "change_tie")
            (e2 if part == "entries" else x2)[q] = chosen
            after = g.config_metrics(e2, x2)
            assert after["train"][g.key] == pytest.approx(
                before["train"][g.key], rel=0, abs=1e-12)
            idx = tuple(int(round((chosen[k] + 10) * 10)) for k in exp["keys"])
            choices.add(int(exp["inv"].reshape(exp["shape"])[idx]))
        if len(choices) > 1:    # allowed only on a perfect tie of depth and size
            info = {(_innermost_point(np.stack(np.unravel_index(
                np.flatnonzero(exp["inv"] == e), exp["shape"]), axis=1))[1],
                int((exp["inv"] == e).sum())) for e in choices}
            assert len(info) == 1, (q, choices, info)


@pytest.mark.parametrize("part", ["entries", "exits"])
def test_rerun_after_recentring_changes_nothing(env, part):
    p, cal = env
    g = _grid_cal(p, cal)
    res = g.calibrate(part, max_rounds=8)
    assert res["converged"]
    cfg2 = g.cfg_with(res["final"]["entries"], res["final"]["exits"])
    g2 = GridCalibrator(cal.ind, cal.params_by_quad, cfg2, start=START, train_start=TRAIN)
    res2 = g2.calibrate(part, max_rounds=2)
    assert res2["rounds"] == 1 and res2["converged"]
    assert all(s["action"] == "keep" for s in res2["history"])
    assert res2["final"] == res["final"]
    assert set(res2["actions"].values()) == {"kept"}


def test_robust_does_not_look_at_the_unseen_years(env):
    """The robust choice does not change if the numbers of the unseen years
    are scrambled; the 'unseen' one instead uses them."""
    p, cal = env
    g = _grid_cal(p, cal)
    rng = np.random.default_rng(3)
    for q, ent, ext, exp in _explorations(g, "exits"):
        obj_cur = g._obj(g.config_metrics(ent, ext))
        a = g._choose(exp, ext[q], obj_cur)
        fake = {f: {m: (rng.normal(size=len(v)) if f == "unseen" else v)
                    for m, v in d.items()} for f, d in exp["metrics"].items()}
        b = g._choose(dict(exp, metrics=fake), ext[q], obj_cur)
        assert a == b


def test_unseen_tiebreak_declared_and_flagged(env, tmp_path):
    from src.backtest.report_grid import write_html
    p, cal = env
    g = GridCalibrator(cal.ind, cal.params_by_quad, signal_config(p),
                       start=START, train_start=TRAIN, tiebreak="unseen")
    res = g.calibrate("exits", max_rounds=4)
    assert res["tiebreak"] == "unseen"
    for q in (1, 2, 3, 4):
        rows = res["per_quad"][q]["ties"]["rows"]
        chosen = next(r for r in rows if r["chosen"])
        assert all(chosen["unseen_calmar"] >= r["unseen_calmar"] - 1e-12 for r in rows)
    page = write_html(res, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "are no longer a clean test" in page


def test_invalid_tiebreak(env):
    p, cal = env
    assert TIEBREAKS == ("robust", "nearest", "unseen")
    with pytest.raises(ValueError):
        GridCalibrator(cal.ind, cal.params_by_quad, signal_config(p),
                       start=START, train_start=TRAIN, tiebreak="whatever")
    import cli
    assert cli.cmd_calibrate_grid(p, "2021-06-30", "exits", "", "params.yaml",
                                  START, TRAIN, open_browser=False,
                                  tiebreak="whatever") == 1


def test_report_shows_margin_and_ties(env, tmp_path):
    from src.backtest.report_grid import write_csv, write_html
    p, cal = env
    g = _grid_cal(p, cal)
    res = g.calibrate("exits", max_rounds=4)
    for q in (1, 2, 3, 4):
        d = res["per_quad"][q]
        assert d["margin"] >= 0 and d["margin"] <= d["max_margin"]
        assert set(d["margins"]) == {"rsi", "vrp"}
        assert res["actions"][q] in ("changed", "changed_among_ties", "recentred", "kept")
        assert d["ties"]["outcomes"] >= 1
        # robust: the chosen point is the innermost of its plateau and no
        # threshold sits on the edge if the plateau is wider than one step
        assert d["margin"] == d["max_margin"]
        for k, m in d["margins"].items():
            if m["width"] >= 2 and d["max_margin"] > 0:
                assert m["min"] >= 1, (q, k, m)
    page = write_html(res, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "informational only" in page and "margin" in page
    assert "are no longer a clean test" not in page
    csv = pd.read_csv(write_csv(res, tmp_path / "r.csv"))
    assert {"tie_in_train", "depth"} <= set(csv.columns)
    assert int(csv["chosen"].sum()) == 4
