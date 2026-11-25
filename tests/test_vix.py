"""Phase-2 unit tests: CBOE parser, contango, signal events, TDEX toggle."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.providers.base import DataProviderError
from src.vix.cboe import CboeTermStructure
from src.vix.indicators import contango_metrics
from src.vix.quad_params import (
    DEFAULT_QUAD_PARAMS,
    QuadParams,
    quad_params_from_config,
)

# fixed test parameters, independent of the production set (midterm)
TEST_QP = {q: QuadParams(7, 8, 19, 2, -0.25, 4.25, 3.5, -3.0)
           for q in (1, 2, 3, 4)}
from src.vix.signal_generator import (
    SignalConfig,
    generate_signals,
    to_asset_signals,
)

SETTLEMENT_CSV = """Product,Symbol,Expiration Date,Price
VX,VX/M6,2026-06-17,18.45
VX,VX/N6,2026-07-22,19.10
VX,VX/Q6,2026-08-19,19.85
VX,VX01/K6,2026-06-03,18.10
ES,ES/M6,2026-06-19,6600.0
VX,VX/J6,2026-04-15,17.00
"""


def test_cboe_parse_and_front_months():
    quotes = CboeTermStructure.parse_settlement_csv(SETTLEMENT_CSV)
    syms = {q.symbol for q in quotes}
    assert "ES/M6" not in syms          # other product: excluded
    assert "VX01/K6" not in syms        # weekly: excluded
    assert len(quotes) == 4

    class Stub(CboeTermStructure):
        def fetch(self):
            return quotes

    front = Stub().front_months(n=3, as_of=date(2026, 6, 1))
    assert front == [18.45, 19.10, 19.85]   # sorted by expiry, VXF1..VXF3

    with pytest.raises(DataProviderError):
        Stub().front_months(n=5, as_of=date(2026, 6, 1))


def test_contango_metrics():
    m = contango_metrics(vxf1=18.45, vxf2=19.10, vxf3=19.85, vix=17.5)
    assert m["contango_f1_f2"] == pytest.approx(19.10 / 18.45 - 1)
    assert m["contango_next"] == pytest.approx(19.85 / 19.10 - 1)  # Alerts!E14
    assert m["roll_yield_spot"] == pytest.approx(18.45 / 17.5 - 1)


def _ind_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["quad"] = df.get("quad", 1)
    return df


def test_buy_vix_event_fires_on_transition_only():
    # quad 1: vrp_long_thr=4.25, rsi_long_thr=-0.25 ; tdex_lv: <= 4
    ind = _ind_frame(
        [
            {"vrp_smooth": 9.0, "rsi_smooth": 1.0, "tdex_smooth": 0.0, "quad": 1},
            {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},
            {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},
        ]
    )
    sig = generate_signals(ind, TEST_QP)
    assert list(sig["buy_vix_state"]) == [False, True, True]
    assert list(sig["buy_vix"]) == [False, True, False]   # only on the transition


def test_sell_vix_two_day_gate():
    # raw sell active 4 days in a row: the event is emitted on the first two
    # (Excel: L = K if K_{t-1} or K_{t-2} were Do Nothing)
    ind = _ind_frame(
        [
            {"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": 0.0, "quad": 1},
            {"vrp_smooth": 9.0, "rsi_smooth": 1.0, "tdex_smooth": 0.0, "quad": 1},
            {"vrp_smooth": 9.0, "rsi_smooth": 1.0, "tdex_smooth": 0.0, "quad": 1},
            {"vrp_smooth": 9.0, "rsi_smooth": 1.0, "tdex_smooth": 0.0, "quad": 1},
            {"vrp_smooth": 9.0, "rsi_smooth": 1.0, "tdex_smooth": 0.0, "quad": 1},
        ]
    )
    sig = generate_signals(ind, TEST_QP)
    assert list(sig["sell_vix_raw"]) == [False, True, True, True, True]
    assert list(sig["sell_vix"]) == [False, True, True, False, False]


def test_buy_spx_requires_tdex_gate():
    # quad 1: vrp_short_thr=-3, rsi_short_thr=3.5
    base = {"vrp_smooth": 5.0, "rsi_smooth": 4.0, "quad": 1}
    ind = _ind_frame(
        [
            {**base, "tdex_smooth": -1.0},   # tdex_on false (-1 <= -0.5)
            {**base, "tdex_smooth": 0.5},    # gate opens -> BUY
        ]
    )
    sig = generate_signals(ind, TEST_QP)
    assert list(sig["buy_spx_state"]) == [False, True]
    assert list(sig["buy_spx"]) == [False, True]

    assets = to_asset_signals(sig.iloc[1], long_asset="TQQQ", vol_asset="UVIX")
    assert assets["TQQQ"] == "BUY"


def test_use_tdex_false_reverts_to_two_vote_logic():
    ind = _ind_frame(
        [{"vrp_smooth": 1.0, "rsi_smooth": -1.0, "tdex_smooth": float("nan"), "quad": 1}]
    )
    sig = generate_signals(ind, TEST_QP, SignalConfig(use_tdex=False))
    assert bool(sig["buy_vix_state"].iloc[0]) is True     # 2 votes are enough
    sig2 = generate_signals(ind, TEST_QP, SignalConfig(use_tdex=True))
    assert bool(sig2["buy_vix_state"].iloc[0]) is False   # TDEX gate NaN -> no


def test_quad_params_from_config_roundtrip():
    cfg = {
        str(q): {
            "rs_ema": p.rs_ema, "rsi_smoother": p.rsi_smoother,
            "hv_lookback": p.hv_lookback, "vrp_ema": p.vrp_ema,
            "rsi_long_thr": p.rsi_long_thr, "vrp_long_thr": p.vrp_long_thr,
            "rsi_short_thr": p.rsi_short_thr, "vrp_short_thr": p.vrp_short_thr,
        }
        for q, p in DEFAULT_QUAD_PARAMS.items()
    }
    assert quad_params_from_config(cfg) == DEFAULT_QUAD_PARAMS
    with pytest.raises(ValueError):
        quad_params_from_config({"1": cfg["1"]})
