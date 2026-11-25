"""Tests of the MAJORITY voting mode of the VIX engine.

FIXED test parameters (TEST_QP): the logic tests do not depend on the
production set in params.yaml/quad_params.py, which follows the midterm
workbook and may change. Thresholds for every quad: vrp_long=4.25,
rsi_long=-0.25, vrp_short=-3, rsi_short=3.5. Test TDEX thresholds:
on=-0.5, lv=-3.0.
"""
from __future__ import annotations

import pandas as pd

from src.vix.quad_params import QuadParams
from src.vix.signal_generator import SignalConfig, generate_signals

TEST_QP = {q: QuadParams(7, 8, 19, 2, -0.25, 4.25, 3.5, -3.0)
           for q in (1, 2, 3, 4)}


def _cfg(**kw):
    base = dict(method="majority", tdex_on_thr=-0.5, tdex_lv_thr=-3.0,
                tdex_off_thr=-7.5, use_tdex=True)
    base.update(kw)
    return SignalConfig(**base)


def _ind(rows):
    return pd.DataFrame(rows)


def test_tdex_does_not_veto_buy_vix():
    # VRP and RSI vote long-vol, TDEX does not -> 2/3 = majority -> BUY.
    # In gate mode the false tdex_lv vetoed: not here.
    ind = _ind([{"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1}])
    sig = generate_signals(ind, TEST_QP, _cfg())
    assert bool(sig["buy_vix_state"].iloc[0]) is True
    # counter-check: in gate the same input does NOT buy (false tdex_lv vetoes)
    gate = generate_signals(ind, TEST_QP, SignalConfig(tdex_lv_thr=-3.0))
    assert bool(gate["buy_vix_state"].iloc[0]) is False


def test_tdex_does_not_veto_buy_spx():
    # VRP and RSI say "buy the market", TDEX does not -> 2/3 -> BUY.
    ind = _ind([{"vrp_smooth": 0.0, "rsi_smooth": 4.0, "tdex_smooth": -1.0, "quad": 1}])
    sig = generate_signals(ind, TEST_QP, _cfg())
    assert bool(sig["buy_spx_state"].iloc[0]) is True


def test_tdex_abstains_on_nan_and_unanimity_is_needed():
    # TDEX NaN -> 2 active legs -> VRP+RSI unanimity is needed.
    ind = _ind([
        {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": float("nan"), "quad": 1},  # 2/2
        {"vrp_smooth": 1.0, "rsi_smooth":  1.0, "tdex_smooth": float("nan"), "quad": 1},  # 1/2
    ])
    sig = generate_signals(ind, TEST_QP, _cfg())
    assert list(sig["buy_vix_state"]) == [True, False]


def test_tdex_votes_and_can_trigger_long_vol():
    # 3 legs: VRP yes, RSI no, TDEX yes (fear) -> 2/3 -> BUY.
    ind = _ind([{"vrp_smooth": 1.0, "rsi_smooth": 1.0, "tdex_smooth": -5.0, "quad": 1}])
    sig = generate_signals(ind, TEST_QP, _cfg())
    # vrp_lv: 1<4.25 True ; rsi_lv: 1<-0.25 False ; tdex_lv: -5<=-3 True -> 2/3
    assert bool(sig["buy_vix_state"].iloc[0]) is True


def test_disabling_tdex_leg_goes_back_to_two_votes():
    # tdex off -> 2 legs -> VRP+RSI unanimity (like the historical logic).
    legs = {"vrp": True, "rsi": True, "tdex": False}
    ind = _ind([
        {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": -5.0, "quad": 1},  # 2/2 -> True
        {"vrp_smooth": 1.0, "rsi_smooth":  1.0, "tdex_smooth": -5.0, "quad": 1},  # 1/2 -> False
    ])
    sig = generate_signals(ind, TEST_QP, _cfg(legs=legs))
    assert list(sig["buy_vix_state"]) == [True, False]
    assert sig["tdex_lv"].isna().all()   # a disabled leg casts no vote


def test_disabling_rsi_tdex_weighs_half():
    # rsi off -> active legs = VRP + TDEX -> both are needed.
    legs = {"vrp": True, "rsi": False, "tdex": True}
    ind = _ind([
        {"vrp_smooth": 1.0, "rsi_smooth": 9.0, "tdex_smooth": -5.0, "quad": 1},  # vrp T, tdex T -> 2/2
        {"vrp_smooth": 1.0, "rsi_smooth": 9.0, "tdex_smooth":  0.0, "quad": 1},  # vrp T, tdex F -> 1/2
    ])
    sig = generate_signals(ind, TEST_QP, _cfg(legs=legs))
    assert list(sig["buy_vix_state"]) == [True, False]


def test_sell_spx_tdex_triggers_on_deep_fear():
    # CORRECT direction of the Sell SPX branch in majority: TDEX triggers the
    # exit when there is deep fear (tdex_smooth < tdex_sell_thr), NOT when the
    # market is calm. Here rsi/vrp would not exit: only TDEX decides.
    cfg = _cfg(tdex_sell_thr=-4.0)
    # day0 calm (tdex high) -> no exit ; day1 fear (tdex low) -> exit
    ind = _ind([
        {"vrp_smooth": 5.0, "rsi_smooth": 5.0, "tdex_smooth":  2.0, "quad": 1},  # calm
        {"vrp_smooth": 5.0, "rsi_smooth": 5.0, "tdex_smooth": -6.0, "quad": 1},  # fear
    ])
    sig = generate_signals(ind, TEST_QP, cfg)
    # raw state: calm=False, fear=True ; the EVENT triggers on the rising edge
    assert list(sig["tdex_off"]) == [False, True]
    assert bool(sig["sell_spx"].iloc[1]) is True
    assert bool(sig["sell_spx"].iloc[0]) is False


def test_sell_spx_tdex_abstains_on_nan():
    # TDEX missing (pre-2013): the TDEX branch neither votes nor exits.
    cfg = _cfg(tdex_sell_thr=-4.0)
    ind = _ind([{"vrp_smooth": 5.0, "rsi_smooth": 5.0,
                 "tdex_smooth": float("nan"), "quad": 1}])
    sig = generate_signals(ind, TEST_QP, cfg)
    assert pd.isna(sig["tdex_off"].iloc[0])
    assert bool(sig["sell_spx_raw"].iloc[0]) is False


def test_tdex_thresholds_per_quad():
    # Same fear value (-4.2): in Disinflation (threshold -3.5) it exits,
    # in Deflation (threshold -4.7) it does NOT, because there deep fear is
    # normal and a deeper confirmation is needed (chosen direction)
    pq = {1: {"sell_spx": -3.5, "longvol": -2.5, "buy_spx": 0.0},
          2: {"sell_spx": -3.9, "longvol": -3.0, "buy_spx": -0.9},
          3: {"sell_spx": -4.0, "longvol": -3.0, "buy_spx": -0.1},
          4: {"sell_spx": -4.7, "longvol": -3.6, "buy_spx": -0.7}}
    cfg = _cfg(tdex_per_quad=pq)
    ind = _ind([
        {"vrp_smooth": 5.0, "rsi_smooth": 5.0, "tdex_smooth": -4.2, "quad": 1},
        {"vrp_smooth": 5.0, "rsi_smooth": 5.0, "tdex_smooth": -4.2, "quad": 4},
    ])
    sig = generate_signals(ind, TEST_QP, cfg)
    assert bool(sig["tdex_off"].iloc[0]) is True    # quad 1: -4.2 < -3.5
    assert bool(sig["tdex_off"].iloc[1]) is False   # quad 4: -4.2 > -4.7
    # and the long-vol vote follows the same scale: -3.4 votes in quad 2 (<=-3.0)
    # but not in quad 4 (<=-3.6 no... -3.4 > -3.6)
    ind2 = _ind([
        {"vrp_smooth": 5.0, "rsi_smooth": 5.0, "tdex_smooth": -3.4, "quad": 2},
        {"vrp_smooth": 5.0, "rsi_smooth": 5.0, "tdex_smooth": -3.4, "quad": 4},
    ])
    sig2 = generate_signals(ind2, TEST_QP, cfg)
    assert bool(sig2["tdex_lv"].iloc[0]) is True
    assert bool(sig2["tdex_lv"].iloc[1]) is False


def test_tdex_per_quad_missing_uses_globals():
    # without the per-quad block the global scalar thresholds are used
    cfg = _cfg(tdex_sell_thr=-4.0, tdex_per_quad=None)
    ind = _ind([{"vrp_smooth": 5.0, "rsi_smooth": 5.0,
                 "tdex_smooth": -4.5, "quad": 3}])
    sig = generate_signals(ind, TEST_QP, cfg)
    assert bool(sig["tdex_off"].iloc[0]) is True


# ----------------------------------------------------------------------- #
# pos0: position continuity at the incremental engine's boundary          #
# (StatefulVix regenerates the signals on a short tail: without pos0 the  #
# position-anchored exits would not trigger for entries before the tail)  #
# ----------------------------------------------------------------------- #
LV = {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1}   # 2/3 long-vol
NO = {"vrp_smooth": 9.0, "rsi_smooth": 9.0, "tdex_smooth": 0.0, "quad": 1}    # 0/3


def test_pos0_triggers_exit_on_position_opened_before_the_tail():
    # Without pos0: no BUY in the window -> no SELL possible.
    ind = _ind([NO, NO])
    sig = generate_signals(ind, TEST_QP, _cfg())
    assert not sig["sell_vix"].any() and not sig["pos_vix"].any()
    # With pos0 (vol position open before the tail): the SELL triggers at once.
    sig = generate_signals(ind, TEST_QP, _cfg(), pos0={"vix": True})
    assert list(sig["sell_vix"]) == [True, False]
    assert list(sig["pos_vix"]) == [False, False]


def test_state0_avoids_false_buy_edge_on_first_row():
    # SPX leg: state already true BEFORE the tail but position closed
    # (possible: the SPX sell can trigger with the state still true).
    # Without spx_state the edge at row 0 would be spurious and reopen.
    BSPX = {"vrp_smooth": 0.0, "rsi_smooth": 4.0, "tdex_smooth": 0.0, "quad": 1}
    ind = _ind([BSPX, BSPX])
    sig = generate_signals(ind, TEST_QP, _cfg(),
                           pos0={"spx": False, "spx_state": True})
    assert not sig["buy_spx"].any()          # no edge: state already true
    assert not sig["pos_spx"].any()          # and therefore no reopening
    # counter-check: if the state was NOT true yesterday, the edge is real
    sig = generate_signals(ind, TEST_QP, _cfg(),
                           pos0={"spx": False, "spx_state": False})
    assert list(sig["buy_spx"]) == [True, False]
    assert list(sig["pos_spx"]) == [True, True]


def test_pos0_vix_already_in_position_does_not_reopen_and_closes_on_decay():
    # In a vol position AND state still true at the boundary: no new BUY,
    # the (consensus) exit comes when the majority decays.
    ind = _ind([LV, NO])
    sig = generate_signals(ind, TEST_QP, _cfg(vol_exit="consensus"),
                           pos0={"vix": True, "vix_state": True})
    assert not sig["buy_vix"].any()
    assert list(sig["sell_vix"]) == [False, True]
    assert list(sig["pos_vix"]) == [True, False]


def test_position_columns_also_in_gate_without_touching_events():
    # The gate stays the Excel replica: pos_vix/pos_spx are only tracking
    # downstream of the events (here: long-vol entry then exit).
    G_IN = {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": -5.0, "quad": 1}
    ind = _ind([G_IN, G_IN, NO])
    gate = generate_signals(ind, TEST_QP, SignalConfig(tdex_lv_thr=-3.0))
    assert list(gate["buy_vix"]) == [True, False, False]
    assert list(gate["sell_vix"]) == [False, False, True]
    assert list(gate["pos_vix"]) == [True, True, False]
