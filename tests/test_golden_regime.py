"""Golden regression tests against the 40-ETF regime workbook.

Fixtures (cached values extracted from the .xlsm):
- golden_etf_tlt.csv    : TLT sheet rows 4980-5344 (close + MA chain + vote)
- golden_regime_agg.csv : Regime sheet rows 5080-5344; the aggregation
  chain is intact through row 5284 (afterwards the external-indicator
  block feeding the adaptive EMA period is broken with #REF!), so
  comparisons stop there. The Excel's per-day EMA period is read from
  the IF column and replayed row by row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.regime.etf_signal import EtfSignal
from src.regime.quad_aggregator import QUAD_NAMES, QuadAggregator
from src.regime.winner import RegimeSeeds, compute_winner

FIX = Path(__file__).parent / "fixtures"
ERR = {"#REF!", "#DIV/0!", "#VALUE!", "#N/A"}

QUADS = {
    "disinflation": ["QUAL", "USMV", "SPHQ", "SCHD", "DGRO", "XLK", "AMLP", "BKLN", "NEAR", "JPST"],
    "reflation": ["VOO", "VT", "QQQM", "MTUM", "IMTM", "JEPI", "PDBC", "COMT", "GUNR", "LIT"],
    "stagflation": ["USDU", "HRL", "BTU", "NTR", "AMR", "UNG", "BOIL", "DOG", "GIS", "PSX"],
    "deflation": ["RWM", "SO", "SDS", "GOVT", "SCHZ", "ISTB", "VGSH", "BAR", "GLD", "TLT"],
}
SYMBOLS = [s for q in QUAD_NAMES for s in QUADS[q]]


# --------------------------------------------------------------------- #
# Per-ETF chain (TLT sheet)                                             #
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tlt() -> pd.DataFrame:
    return pd.read_csv(FIX / "golden_etf_tlt.csv").astype({"close": float})


def test_etf_chain_matches_excel(tlt):
    out = EtfSignal(symbol="TLT").compute(tlt)
    tail = slice(70, None)  # warm-up: MA63 + 3 days of u3
    for mine, theirs, tol in [
        ("ma_5", "ma5", 1e-9), ("ma_8", "ma8", 1e-9), ("ma_13", "ma13", 1e-9),
        ("ma_20", "ma20", 1e-9), ("ma_41", "ma41", 1e-9), ("ma_63", "ma63", 1e-9),
        ("summed", "summed", 1e-9), ("treatment", "treatment", 1e-7),
        ("u3", "u3", 1e-7),
    ]:
        np.testing.assert_allclose(
            out[mine].to_numpy()[tail],
            tlt[theirs].astype(float).to_numpy()[tail],
            atol=tol, rtol=0, err_msg=mine,
        )


def test_etf_vote_matches_excel(tlt):
    out = EtfSignal(symbol="TLT").compute(tlt)
    got = out["vote"].to_numpy()[70:]
    want = tlt["vote_ap"].astype(float).to_numpy()[70:]
    mismatch = np.flatnonzero(got != want)
    assert mismatch.size == 0, (
        f"{mismatch.size} different votes, first at idx {mismatch[:5]} "
        f"(xl_row {tlt['xl_row'].to_numpy()[70:][mismatch[:5]]})"
    )


def test_effective_ma_windows_not_labels(tlt):
    """The effective windows are 20/41, not the labels 21/42."""
    close = tlt["close"]
    excel = tlt["ma20"].astype(float).to_numpy()[100:110]
    assert not np.allclose(close.rolling(21).mean().to_numpy()[100:110], excel, atol=1e-9)
    np.testing.assert_allclose(close.rolling(20).mean().to_numpy()[100:110], excel, atol=1e-9)
    excel41 = tlt["ma41"].astype(float).to_numpy()[100:110]
    np.testing.assert_allclose(close.rolling(41).mean().to_numpy()[100:110], excel41, atol=1e-9)


# --------------------------------------------------------------------- #
# Aggregation and Winner (Regime sheet, HL..MH)                        #
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def regime() -> pd.DataFrame:
    df = pd.read_csv(FIX / "golden_regime_agg.csv", dtype=str)
    good = ~df[["IF", "HZ", "KN", "LM"]].isin(ERR).any(axis=1)
    good &= df[["IF", "HZ", "KN", "LM"]].notna().all(axis=1)
    df = df[good].reset_index(drop=True)
    assert len(df) >= 150, f"only {len(df)} intact rows"
    return df


def _votes(df) -> dict[str, pd.Series]:
    return {
        sym: pd.to_numeric(df[f"v_{i+1:02d}"], errors="coerce")
        for i, sym in enumerate(SYMBOLS)
    }


def test_scores_and_fractions_match_excel(regime):
    agg = QuadAggregator(QUADS).aggregate(_votes(regime))
    for quad, xl in zip(QUAD_NAMES, ["HL", "HM", "HN", "HO"]):
        np.testing.assert_allclose(
            agg[f"score_{quad}"].to_numpy(), pd.to_numeric(regime[xl]).to_numpy(),
            atol=0, err_msg=quad,
        )
    np.testing.assert_allclose(agg["total"], pd.to_numeric(regime["HP"]), atol=0)
    for quad, xl in zip(QUAD_NAMES, ["HQ", "HR", "HS", "HT"]):
        np.testing.assert_allclose(
            agg[f"frac_{quad}"].to_numpy(), pd.to_numeric(regime[xl]).to_numpy(),
            atol=1e-12, err_msg=quad,
        )


@pytest.fixture(scope="module")
def winner_run(regime):
    """compute_winner seeded from the first row of the fixture, with the
    per-row EMA period read from the Excel's IF column."""
    agg = QuadAggregator(QUADS).aggregate(_votes(regime))
    seeds = RegimeSeeds(
        ema=[float(regime[c].iloc[0]) for c in ("HZ", "IA", "IB", "IC")],
        avg_gain=[float(regime[c].iloc[0]) for c in ("KK", "KR", "KY", "LF")],
        avg_loss=[float(regime[c].iloc[0]) for c in ("KL", "KS", "KZ", "LG")],
        final=[float(regime[c].iloc[0]) for c in ("LO", "LP", "LQ", "LR")],
    )
    periods = pd.to_numeric(regime["IF"]).to_numpy()[1:]
    win = compute_winner(agg.iloc[1:], seeds=seeds, ema_periods=periods)
    return regime.iloc[1:].reset_index(drop=True), win.reset_index(drop=True)


def test_ema_and_rsi_chains_match_excel(winner_run):
    xl, win = winner_run
    for quad, (e_col, r_col) in zip(
        QUAD_NAMES, [("HZ", "KN"), ("IA", "KU"), ("IB", "LB"), ("IC", "LI")]
    ):
        np.testing.assert_allclose(
            win[f"ema_{quad}"].to_numpy(), pd.to_numeric(xl[e_col]).to_numpy(),
            atol=1e-10, err_msg=f"ema {quad}",
        )
        np.testing.assert_allclose(
            win[f"rsi_{quad}"].to_numpy(), pd.to_numeric(xl[r_col]).to_numpy(),
            atol=1e-7, err_msg=f"rsi {quad}",
        )


def test_selectors_and_winner_match_excel(winner_run):
    xl, win = winner_run
    np.testing.assert_allclose(
        win["bifurcation"].to_numpy(), pd.to_numeric(xl["LK"]).to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        win["reverse"].to_numpy(), pd.to_numeric(xl["LL"]).to_numpy(), atol=1e-7
    )
    for mine, theirs in [("rsi_winner", "LJ"), ("ema_winner", "IE")]:
        np.testing.assert_array_equal(
            win[mine].to_numpy(), pd.to_numeric(xl[theirs]).to_numpy(), err_msg=mine
        )
    # roc5_winner and final winner: compared after the ROC warm-up (5 rows)
    tail = slice(6, None)
    np.testing.assert_array_equal(
        win["roc5_winner"].to_numpy()[tail],
        pd.to_numeric(xl["MH"]).to_numpy()[tail], err_msg="MH",
    )
    got = win["winner"].to_numpy()[tail]
    want = pd.to_numeric(xl["LM"]).to_numpy()[tail]
    mismatch = np.flatnonzero(got != want)
    assert mismatch.size == 0, (
        f"{mismatch.size} different winners, first idx {mismatch[:5]} "
        f"(xl_row {xl['xl_row'].to_numpy()[tail][mismatch[:5]]})"
    )
