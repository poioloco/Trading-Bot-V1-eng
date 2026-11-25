"""Indicator + regime pipeline tests (synthetic data)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.regime.etf_signal import EtfSignal
from src.regime.regime_detector import RegimeDetector
from src.utils.indicators import wilder_rsi
from tests.fakes import FakeProvider

QUADS = {
    "disinflation": ["D1", "D2"],
    "reflation": ["R1", "R2"],
    "stagflation": ["S1", "S2"],
    "deflation": ["F1", "F2"],
}
TRENDS = {
    "D1": "up", "D2": "up",
    "R1": "flat", "R2": "flat",
    "S1": "flat", "S2": "flat",
    "F1": "down", "F2": "down",
}


def test_wilder_rsi_hand_computed():
    # closes: 10,11,12,11,12 ; period 3
    # seed avg_gain=(1+1+0)/3, avg_loss=(0+0+1)/3 -> RS=2 -> RSI=66.667
    # next: g=(0.6667*2+1)/3, l=(0.3333*2+0)/3 -> RS=3.5 -> RSI=77.778
    s = pd.Series([10.0, 11.0, 12.0, 11.0, 12.0])
    rsi = wilder_rsi(s, period=3)
    assert rsi.iloc[:3].isna().all()
    assert rsi.iloc[3] == pytest.approx(66.6667, abs=1e-3)
    assert rsi.iloc[4] == pytest.approx(77.7778, abs=1e-3)


def test_wilder_rsi_extremes():
    up = pd.Series(np.arange(1.0, 60.0))
    assert wilder_rsi(up, 21).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert wilder_rsi(down, 21).iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_etf_signal_votes_follow_trend():
    fake = FakeProvider(trends={"UP": "up", "DOWN": "down"})
    start, end = date(2024, 12, 1), date(2026, 6, 1)
    up_votes = EtfSignal("UP").compute(fake.get_daily_bars("UP", start, end))
    down_votes = EtfSignal("DOWN").compute(fake.get_daily_bars("DOWN", start, end))
    assert up_votes["vote"].iloc[-1] == 1
    assert down_votes["vote"].iloc[-1] == -1


def test_regime_detector_end_to_end():
    detector = RegimeDetector(
        provider=FakeProvider(trends=TRENDS),
        quads=QUADS,
        lookback_days=500,
    )
    result = detector.run(as_of=date(2026, 6, 1))
    # both disinflation ETFs rise -> 2 bullish votes; deflation falls -> 0
    assert result.quad in (1, 2, 3, 4)
    assert result.regime == ["disinflation", "reflation", "stagflation", "deflation"][result.quad - 1]
    assert set(result.scores) == set(QUADS)
    assert result.scores["disinflation"] == pytest.approx(2.0)
    assert result.scores["deflation"] == pytest.approx(0.0)
    assert not result.history.empty
    assert result.history["winner"].iloc[-1] == result.quad
