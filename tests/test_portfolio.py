"""Tests of the consensus portfolio.

Checks votes, consensus, hysteresis (once in a position you don't leave
without the opposite consensus) and deposit accounting. All on hand-built
data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import (CASH, OFF, ON, PortfolioSimulator, apply_hysteresis,
                           consensus_target, leg_votes)


def _timeline(pos_m, pos_v):
    idx = pd.date_range("2020-01-01", periods=len(pos_m), freq="B")
    return pd.DataFrame({
        "pos_market": pos_m, "pos_volatility": pos_v,
        "regime": ["reflation"] * len(pos_m),
        "rsi_smooth": [50.0] * len(pos_m), "vrp_smooth": [5.0] * len(pos_m),
    }, index=idx)


def test_votes_and_raw_consensus():
    tl = _timeline([1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0])
    v = leg_votes(tl, vol_inverted=True)
    assert list(v["market"]) == [1, 1, -1, 1, -1, -1]
    assert list(v["volatility"]) == [1, 1, 1, 1, -1, 1]
    raw = consensus_target(v, threshold=0.5)
    assert list(raw) == [ON, ON, CASH, ON, OFF, CASH]


def test_hysteresis_holds_until_opposite():
    raw = pd.Series([CASH, ON, CASH, CASH, OFF, CASH, ON],
                    index=pd.date_range("2020-01-01", periods=7, freq="B"))
    held = apply_hysteresis(raw)
    # cash while flat; then holds ON on days without consensus; flips to OFF; etc.
    assert list(held) == [CASH, ON, ON, ON, OFF, OFF, ON]


def test_simulation_accounts_deposits_and_pnl():
    tl = _timeline([1, 1, 1, 1, 1], [0, 0, 0, 0, 0])      # always long consensus
    idx = tl.index
    price_on = pd.Series([100, 110, 121, 133.1, 146.41], index=idx)   # +10%/day
    price_off = pd.Series([100, 100, 100, 100, 100], index=idx)
    contributions = pd.Series([1000.0], index=[idx[0]])
    sim = PortfolioSimulator("^SPX", "TLT", threshold=0.5, vol_inverted=True, hysteresis=True)
    res = sim.run(tl, price_on, price_off, contributions)
    assert res.stats["total_deposited"] == 1000.0
    # 1000€ invested at 100 -> 10 units -> at 146.41 they are worth ~1464
    assert abs(res.stats["final_value"] - 1464.1) < 1.0
    assert res.stats["total_pnl"] > 400
    # position still open at the end of the period
    assert bool(res.trades.iloc[-1]["is_open"]) is True


def test_no_consensus_stays_cash_when_flat():
    tl = _timeline([1, 0, 1, 0], [1, 0, 1, 0])           # always mixed votes
    idx = tl.index
    price = pd.Series([100, 101, 102, 103], index=idx)
    res = PortfolioSimulator("^SPX", "TLT").run(tl, price, price, pd.Series([500.0], index=[idx[0]]))
    # never a consensus -> always cash -> no trades, value = deposited
    assert res.stats["time_cash_pct"] == 100.0
    assert res.trades.empty
    assert res.stats["final_value"] == 500.0


def _timeline_tdex(pos_m, pos_v, tdex):
    idx = pd.date_range("2020-01-01", periods=len(pos_m), freq="B")
    return pd.DataFrame({
        "pos_market": pos_m, "pos_volatility": pos_v, "tdex_smooth": tdex,
        "regime": ["reflation"] * len(pos_m),
        "rsi_smooth": [50.0] * len(pos_m), "vrp_smooth": [5.0] * len(pos_m),
    }, index=idx)


LEGS3 = {"market": True, "volatility": True, "tdex": True}


def test_tdex_third_voter_abstains_and_votes_by_threshold():
    # tdex_smooth: NaN -> abstains(0), 2.0(<=4) -> risk-off(-1), 9.0(>4) -> risk-on(+1)
    tl = _timeline_tdex(pos_m=[1, 1, 1, 0], pos_v=[0, 0, 0, 1],
                        tdex=[np.nan, 9.0, 2.0, 2.0])
    v = leg_votes(tl, LEGS3, tdex_threshold=4.0, vol_inverted=True)
    assert list(v["tdex"]) == [0, 1, -1, -1]
    raw = consensus_target(v, threshold=0.5)
    # r0: +1,+1,abstains -> 2/2 long ON ; r1: all long ON ;
    # r2: +1,+1,-1 -> 2/3 long ON (majority) ; r3: -1,-1,-1 -> OFF
    assert list(raw) == [ON, ON, ON, OFF]


def test_tdex_breaks_a_two_leg_tie():
    tl = _timeline_tdex(pos_m=[1], pos_v=[1], tdex=[2.0])   # market +1, vol -1
    without = consensus_target(leg_votes(
        tl, {"market": True, "volatility": True, "tdex": False}), 0.5)
    assert list(without) == [CASH]                         # 1-1 -> cash
    with_tdex = consensus_target(leg_votes(tl, LEGS3, 4.0, True), 0.5)
    assert list(with_tdex) == [OFF]                        # +1,-1,-1 -> 2/3 short


def test_disabling_all_legs_is_always_cash():
    tl = _timeline_tdex([1, 0], [0, 1], [9.0, 2.0])
    v = leg_votes(tl, {"market": False, "volatility": False, "tdex": False})
    assert v.shape[1] == 0
    assert list(consensus_target(v)) == [CASH, CASH]
