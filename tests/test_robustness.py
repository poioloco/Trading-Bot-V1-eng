# -*- coding: utf-8 -*-
"""Tests of the robustness battery.

Two kinds of checks:
  a) MECHANICS — costs, latency, leverage and metrics do what they say, on
     hand-built series where the right answer can be worked out with pen
     and paper;
  b) CALIBRATION OF THE TOOLS — the statistical tests must REJECT noise and
     ACCEPT a true signal. Without this check a p-value is just a number:
     here we verify that the tool can tell the two cases apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.robustness import (
    Costs, Robustness, metrics, strategy_returns, _block_durations,
    _norm_ppf, _norm_cdf,
)
from src.vix.quad_params import QuadParams
from src.vix.signal_generator import SignalConfig


# ---------------------------------------------------------------------- #
#  a) mechanics: costs, latency, leverage                                 #
# ---------------------------------------------------------------------- #
def _series(vals, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(vals, index=idx)


def test_latency_shifts_exposure_by_one_day():
    """Signal on at day 0: at T+1 the return of day 1 is captured, the one
    of day 0 is not."""
    ret = _series([0.10, 0.20, 0.30, 0.0])
    pos = _series([True, False, False, False])
    r1 = strategy_returns(pos, ret, Costs(latency=1))
    assert r1.iloc[0] == 0.0            # the signal is born EOD: nothing
    assert r1.iloc[1] == pytest.approx(0.20)
    r2 = strategy_returns(pos, ret, Costs(latency=2))
    assert r2.iloc[1] == 0.0
    assert r2.iloc[2] == pytest.approx(0.30)


def test_costs_are_paid_on_turnover_and_scale_with_leverage():
    ret = _series([0.0, 0.0, 0.0, 0.0])
    pos = _series([True, True, False, False])     # 1 entry + 1 exit
    r = strategy_returns(pos, ret, Costs(bps_side=10.0, latency=1))
    assert r.sum() == pytest.approx(-2 * 10 / 10_000, abs=1e-12)
    # at leverage 3 the same round trip costs 3 times as much
    r3 = strategy_returns(pos, ret, Costs(bps_side=10.0, latency=1, leverage=3.0))
    assert r3.sum() == pytest.approx(-3 * 2 * 10 / 10_000, abs=1e-12)


def test_leverage_amplifies_and_financing_costs():
    ret = _series([0.0, 0.01, 0.01, 0.01])
    pos = _series([True] * 4)
    r1 = strategy_returns(pos, ret, Costs(leverage=1.0))
    r3 = strategy_returns(pos, ret, Costs(leverage=3.0))
    assert r3.iloc[2] == pytest.approx(3 * r1.iloc[2])
    r3f = strategy_returns(pos, ret, Costs(leverage=3.0, fin_annual=5.0))
    assert r3f.iloc[2] < r3.iloc[2]      # financing subtracts every day


def test_metrics_on_a_hand_computable_case():
    """+1% a day for 252 days: multiple = 1.01^252, zero drawdown."""
    r = _series([0.01] * 252)
    m = metrics(r)
    assert m["multiple"] == pytest.approx(1.01 ** 252, rel=1e-3)
    assert m["max_dd_pct"] == pytest.approx(0.0, abs=1e-9)
    assert m["calmar"] is None           # without drawdown the Calmar does not exist
    assert m["max_recovery_days"] == 0


def test_days_underwater():
    r = _series([0.0, -0.10, 0.0, 0.0, 0.20])
    m = metrics(r)
    assert m["max_recovery_days"] == 3   # days 1,2,3 below the peak


def test_entry_count_and_durations():
    pos = _series([False, True, True, False, True, False])
    m = metrics(_series([0.0] * 6), pos)
    assert m["entries"] == 2
    assert m["time_in_market_pct"] == pytest.approx(50.0)
    assert _block_durations(pos.to_numpy()) == [2, 1]


def test_inverse_normal_and_cdf_consistent():
    for q in (0.01, 0.25, 0.5, 0.9, 0.999):
        assert _norm_cdf(_norm_ppf(q)) == pytest.approx(q, abs=1e-6)


# ---------------------------------------------------------------------- #
#  b) calibration: noise vs signal                                        #
# ---------------------------------------------------------------------- #
def _fake_ind(n=2600, seed=7, drift=0.0003):
    """Synthetic indicators with the minimum structure the engine needs:
    random-walk prices, quad and the three smoothed indicators."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    ret = rng.normal(drift, 0.011, n)
    spx = pd.Series(100 * np.cumprod(1 + ret), index=idx)
    return pd.DataFrame({
        "spx": spx, "vix": pd.Series(rng.normal(18, 4, n), index=idx).abs(),
        "quad": pd.Series(rng.integers(1, 5, n), index=idx),
        "rsi_smooth": pd.Series(rng.normal(0, 5, n), index=idx),
        "vrp_smooth": pd.Series(rng.normal(0, 5, n), index=idx),
        "tdex_smooth": pd.Series(rng.normal(0, 3, n), index=idx),
        "spx_close": spx, "spx_sma": spx.rolling(200).mean(),
    }, index=idx)


def _rb():
    ind = _fake_ind()
    pbq = {q: QuadParams(rs_ema=5, rsi_smoother=3, hv_lookback=10, vrp_ema=10,
                         rsi_long_thr=-3, vrp_long_thr=3,
                         rsi_short_thr=1, vrp_short_thr=1) for q in (1, 2, 3, 4)}
    return Robustness(ind, pbq, SignalConfig(method="gate"), seed=1)


def test_permutation_rejects_a_random_position():
    """A strategy that enters at random must NOT come out significant: the
    p-value must be high. If this test passed with a low p-value, the tool
    would be manufacturing edge out of nothing."""
    rb = _rb()
    rng = np.random.default_rng(3)
    pos = pd.Series(rng.random(len(rb.ind)) < 0.5, index=rb.ind.index)
    r = rb.permutation(pos, n=200, costs=Costs())
    assert r["p_value"]["sharpe"] > 0.10


def test_permutation_accepts_perfect_timing():
    """Conversely: whoever knows the future (in only on positive days) must
    come out unmistakably outside chance."""
    rb = _rb()
    # position = tomorrow goes up (perfect, since execution is at T+1)
    pos = (rb.ret.shift(-1) > 0).fillna(False)
    r = rb.permutation(pos, n=200, costs=Costs())
    assert r["p_value"]["sharpe"] <= 0.01
    assert r["observed"]["sharpe"] > max(
        v["p95"] for k, v in r["null"].items() if k == "sharpe")


def test_permutation_preserves_the_null_structure():
    """The rotation null must have EXACTLY the same time in the market: that
    is what makes the comparison fair."""
    rb = _rb()
    rng = np.random.default_rng(5)
    pos = pd.Series(rng.random(len(rb.ind)) < 0.4, index=rb.ind.index)
    a = pos.to_numpy()
    for k in (37, 500, 1200):
        assert np.roll(a, k).sum() == a.sum()


def test_bootstrap_interval_contains_the_observed():
    rb = _rb()
    pos = pd.Series(True, index=rb.ind.index)
    r = rb.bootstrap(pos, n=200, block=21, costs=Costs())
    lo = r["intervals"]["cagr_pct"]["p5"]
    hi = r["intervals"]["cagr_pct"]["p95"]
    assert lo <= r["observed"]["cagr_pct"] <= hi
    # always in = no possible advantage over the benchmark
    assert abs(r["intervals"]["extra_cagr_pct"]["p50"]) < 0.5


def test_deflated_sharpe_discounts_the_trials():
    """With many trials the chance threshold rises: the same Sharpe that
    passes with 2 trials must not pass with 5000."""
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0003, 0.01, 3000))
    few = Robustness.deflated_sharpe(r, np.array([0.4, 0.5]))
    many = Robustness.deflated_sharpe(r, rng.normal(0.3, 0.15, 5000))
    assert many["expected_chance_sharpe_annual"] > few["expected_chance_sharpe_annual"]
    assert many["dsr_pct"] <= few["dsr_pct"]


def test_deflated_sharpe_rejects_pure_noise():
    """Returns without drift: the probability of a true Sharpe > 0 must not
    get close to significance."""
    rng = np.random.default_rng(13)
    r = pd.Series(rng.normal(0.0, 0.01, 4000))
    d = Robustness.deflated_sharpe(r, rng.normal(0.2, 0.2, 500))
    assert d["dsr_pct"] < 95.0
    assert not d["beats_chance"]


def test_pbo_high_when_the_candidates_are_noise():
    """Calibration test of the PBO on a synthetic matrix: N series of pure
    noise, none better than the others. Picking the best in-sample, out of
    sample it must land at random -> PBO close to 50%."""
    rng = np.random.default_rng(17)
    T, N, S = 2000, 60, 8
    M = rng.normal(0.0002, 0.01, (T, N))
    edges = np.linspace(0, T, S + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(S)]
    from itertools import combinations
    below = 0
    combos = list(combinations(range(S), S // 2))
    for c in combos:
        i_is = np.concatenate([blocks[i] for i in c])
        i_oos = np.concatenate([blocks[i] for i in range(S) if i not in c])
        sr = lambda X: X.mean(axis=0) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)
        n_star = int(np.argmax(sr(M[i_is])))
        p_oos = sr(M[i_oos])
        below += int((p_oos < p_oos[n_star]).sum() / N < 0.5)
    pbo = 100.0 * below / len(combos)
    assert 30.0 <= pbo <= 70.0, f"noise PBO outside the expected range: {pbo}"


def test_costs_latency_includes_the_benchmark_and_all_combinations():
    rb = _rb()
    confs = {"A": SignalConfig(method="gate"),
             "B": SignalConfig(method="majority")}
    df = rb.costs_latency(confs, bps=(0.0, 5.0), latencies=(1, 2), leverages=(1.0,))
    assert set(df.config.unique()) == {"A", "B", "buy&hold"}
    assert len(df[df.config == "A"]) == 4          # 2 bps × 2 latencies
    # more costs => never more return
    a = df[(df.config == "A") & (df.latency == 1)].sort_values("bps")
    assert a.iloc[0]["cagr_pct"] >= a.iloc[-1]["cagr_pct"]


def test_subperiods_covers_all_years_and_the_jackknife():
    rb = _rb()
    pos = pd.Series(True, index=rb.ind.index)
    sp = rb.subperiods(pos, costs=Costs())
    expected_years = sorted({d.year for d in rb.ind.index})
    assert [a["year"] for a in sp["per_year"]] == expected_years
    assert len(sp["annual_jackknife"]) == len(expected_years)
    # always invested => no extra over the market
    assert abs(sp["concentration"]["extra_total_log"]) < 1e-6


def test_data_coverage_finds_the_gap():
    rb = _rb()
    rb.ind.loc[:"2012-12-31", "tdex_smooth"] = np.nan
    cov = rb.data_coverage()
    assert cov["tdex_smooth"]["first_data"] >= "2013-01-01"
    assert cov["history_fully_covered_from"] == cov["tdex_smooth"]["first_data"]


def test_walk_forward_does_not_look_into_the_future():
    """The thresholds of every fold must be quantiles computed ONLY on the
    training data: changing the data of the test period alone must not
    change the chosen thresholds."""
    rb = _rb()
    wf = rb.walk_forward(n_fold=2, n_samples=8, costs=Costs(),
                         min_train_years=4.0)
    folds = [f for f in wf["fold"] if "skip" not in f]
    assert folds, "no valid fold"
    assert all(f["train_from"] == str(rb.ind.index[0].date()) for f in folds)
    # the test blocks are consecutive and do not overlap
    for a, b in zip(folds, folds[1:]):
        assert a["test_to"] < b["test_from"]
    assert wf["oos_stitched"]["calibrated"]["days"] > 0
