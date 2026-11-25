"""The live runner must apply the quad DAY BY DAY (series of the 40-ETF
model's winners), not stamp today's quad over the whole historical window.
Live == backtest fidelity (replaces the retired Excel golden test)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.runner.signal_runner import SignalRunner
from src.vix.quad_params import DEFAULT_QUAD_PARAMS
from src.vix.signal_generator import SignalConfig


class _FakeProvider:
    """400 synthetic SPX/VIX bars ending at `end`."""

    def __init__(self, end: date, n: int = 400):
        idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
        rng = np.random.default_rng(7)
        self._spx = pd.DataFrame(
            {"close": 4000 + np.cumsum(rng.normal(0, 10, n))}, index=idx
        )
        self._vix = pd.DataFrame(
            {"close": 16 + np.abs(rng.normal(0, 2, n))}, index=idx
        )

    def get_daily_bars(self, symbol, start, end):
        df = self._vix if "VIX" in symbol else self._spx
        return df.loc[pd.Timestamp(start):pd.Timestamp(end)]


def _mk_runner(end: date, quad_hist: pd.Series) -> SignalRunner:
    return SignalRunner(
        provider=_FakeProvider(end),
        params_by_quad=DEFAULT_QUAD_PARAMS,
        signal_cfg=SignalConfig(method="majority", use_tdex=False,
                                legs={"vrp": True, "rsi": True, "tdex": False}),
        tdex_provider=None,
        regime_history_fn=lambda: quad_hist,
        history_days=700,
    )


def test_runner_uses_daily_quad_not_constant():
    end = date(2026, 6, 30)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=400)
    # half of the history in quad 2, half in quad 4: if the runner stamped
    # today's quad (4) over everything, the first half would be wrong
    winners = pd.Series([2.0] * 200 + [4.0] * 200, index=idx)
    rep = _mk_runner(end, winners).run(as_of=end)
    assert rep.quad == 4
    assert "daily series" in " ".join(rep.notes)


def test_runner_drops_days_without_regime():
    end = date(2026, 6, 30)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=400)
    # regime available only on the last 350 days (40-ETF warm-up):
    # the first 50 must be dropped, not filled with today's quad
    winners = pd.Series([3.0] * 350, index=idx[-350:])
    rep = _mk_runner(end, winners).run(as_of=end)
    assert rep.quad == 3


def test_runner_manual_quad_fallback():
    end = date(2026, 6, 30)
    runner = SignalRunner(
        provider=_FakeProvider(end),
        params_by_quad=DEFAULT_QUAD_PARAMS,
        signal_cfg=SignalConfig(method="majority", use_tdex=False,
                                legs={"vrp": True, "rsi": True, "tdex": False}),
        tdex_provider=None,
        manual_quad=2,
        history_days=700,
    )
    rep = runner.run(as_of=end)
    assert rep.quad == 2
    assert any("manual" in n for n in rep.notes)
