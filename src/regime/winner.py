"""Regime winner - exact replica of the Regime sheet, columns HZ..MH.

Decoded formulas (shared-formula anchors verified at rows ~4936-5062):

  Adaptive EMA of the quad fractions:
    HZ..IC ema_q(t) = (frac_q - ema_q(t-1)) * alpha + ema_q(t-1)
    alpha = 2/(period+1); in the Excel the period (IF) adapts daily via a
    lookup on the count of 12 manually-fed confirmation indicators
    (DIX, GEX, PCR, VVIX:VIX, NPD, VGR, ...):
      confirms 1->0, 2->3, 3->5, 4->8, 5->13, 6->16, 7->18, 8->20,
               9->21, 10->33, 11->39, 12->42
    Those indicators are external data the bot does not receive, so the
    period is a fixed config value here (default 16 = the workbook's
    current effective value, 6 confirms).

  RSI(21) per quad on the DAILY CHANGES of the EMA'd fractions
  (EMA-style averages, alpha = 2/22 - same style as the VIX model):
    change = ema_q diff; gain/loss split
    avg_gain(t) = (gain - avg_gain(t-1)) * 2/22 + avg_gain(t-1)   (idem loss)
    rsi_q = 100 if avg_loss == 0 else 100 - 100/(1 + avg_gain/avg_loss)

  Selection:
    LJ rsi_winner  = argmax(rsi_1..rsi_4)            (ties -> lowest quad)
    IE ema_winner  = argmax(ema_1..ema_4)
    LK bifurcation = max(ema) - min(ema)              threshold $LK$3 = 0.215
    LL reverse     = max(rsi) - min(rsi)              threshold $LL$3 = 92
    LO..LR final_q = rsi_q if LK < 0.215 else ema_q*100
    LT..LW roc_q   = final_q / final_q(t-1) - 1
    LY..MB roc3_q  = AVERAGE(roc_q, last 3)
    MD..MG roc5_q  = AVERAGE(roc_q, last 5)
    MH roc5_winner = argmax(roc5)
    LM WINNER      = roc5_winner  if LK < 0.215 and LL > 92
                     rsi_winner   if LK < 0.215
                     ema_winner   otherwise
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .quad_aggregator import QUAD_NAMES

# adaptive EMA period lookup: number of confirms -> period (Regime!JG..JR)
CONFIRMS_TO_EMA_PERIOD = {1: 0, 2: 3, 3: 5, 4: 8, 5: 13, 6: 16,
                          7: 18, 8: 20, 9: 21, 10: 33, 11: 39, 12: 42}


@dataclass
class RegimeParams:
    rsi_period: int = 21            # Regime!KG2
    ema_period: int = 16            # Regime!IF (adaptive in the Excel, fixed here)
    bifurcation_thr: float = 0.215  # Regime!$LK$3
    reverse_thr: float = 92.0       # Regime!$LL$3

    @property
    def rsi_alpha(self) -> float:
        return 2.0 / (self.rsi_period + 1.0)

    @property
    def ema_alpha(self) -> float:
        return 2.0 / (self.ema_period + 1.0)


@dataclass
class RegimeSeeds:
    """Boundary state for mid-stream replication (golden test, incremental
    runs). Lists are per quad in order disinflation..deflation."""
    ema: list[float] = field(default_factory=lambda: [0.0] * 4)
    avg_gain: list[float] = field(default_factory=lambda: [0.0] * 4)
    avg_loss: list[float] = field(default_factory=lambda: [0.0] * 4)
    final: list[float] | None = None   # LO..LR previous row (for the ROCs)


def _argmax_quad(values: np.ndarray) -> int:
    """Excel nested-IF argmax: first quad reaching the max wins (1..4)."""
    mx = values.max()
    for i in range(4):
        if values[i] == mx:
            return i + 1
    return 4


def compute_winner(
    agg: pd.DataFrame,
    params: RegimeParams | None = None,
    seeds: RegimeSeeds | None = None,
    ema_periods: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """`agg` is the QuadAggregator output (frac_<quad> columns required).
    Returns ema/rsi chains, selector diagnostics and the final `winner`.
    `ema_periods` optionally overrides params.ema_period with a per-row
    period (the Excel adapts it daily via the confirmation lookup)."""
    params = params or RegimeParams()
    seeds = seeds or RegimeSeeds()

    fracs = agg[[f"frac_{q}" for q in QUAD_NAMES]].to_numpy(dtype=float)
    n = len(agg)
    if ema_periods is not None:
        periods = np.asarray(ema_periods, dtype=float)
        if len(periods) != n:
            raise ValueError("ema_periods length must match agg")
        alphas = 2.0 / (periods + 1.0)
    else:
        alphas = np.full(n, params.ema_alpha)

    ema = np.zeros((n, 4))
    change = np.zeros((n, 4))
    avg_gain = np.zeros((n, 4))
    avg_loss = np.zeros((n, 4))
    rsi = np.zeros((n, 4))

    a_r = params.rsi_alpha
    e_prev = np.array(seeds.ema, dtype=float)
    g_prev = np.array(seeds.avg_gain, dtype=float)
    l_prev = np.array(seeds.avg_loss, dtype=float)
    # Without seeds (warm-up from the start) the chain starts at the FIRST
    # row with valid fractions: the initial days without votes have
    # frac=NaN and must not poison the EMA recursion (a single NaN makes it
    # NaN forever). With valid seeds (golden test, incremental steps) the
    # behaviour is identical to before.
    seeded = bool(np.any(e_prev != 0.0) or np.any(np.asarray(seeds.avg_gain, dtype=float) != 0.0))

    for i in range(n):
        if np.isnan(fracs[i]).any():
            ema[i] = np.nan
            change[i] = np.nan
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
            rsi[i] = np.nan
            continue
        if not seeded:                       # first valid row: initialise
            e_prev = fracs[i].copy()
            seeded = True
        e_now = (fracs[i] - e_prev) * alphas[i] + e_prev
        ema[i] = e_now
        change[i] = e_now - e_prev
        gain = np.where(change[i] > 0, change[i], 0.0)
        loss = np.where(change[i] < 0, np.abs(change[i]), 0.0)
        g_prev = (gain - g_prev) * a_r + g_prev
        l_prev = (loss - l_prev) * a_r + l_prev
        avg_gain[i] = g_prev
        avg_loss[i] = l_prev
        for q in range(4):
            if l_prev[q] == 0.0:
                rsi[i, q] = 100.0
            else:
                rsi[i, q] = 100.0 - 100.0 / (1.0 + g_prev[q] / l_prev[q])
        e_prev = e_now

    lk = ema.max(axis=1) - ema.min(axis=1)            # bifurcation
    ll = rsi.max(axis=1) - rsi.min(axis=1)            # reverse
    lj = np.array([_argmax_quad(rsi[i]) for i in range(n)])
    ie = np.array([_argmax_quad(ema[i]) for i in range(n)])

    # LO..LR "Final" and ROC chains
    use_rsi = lk < params.bifurcation_thr
    final = np.where(use_rsi[:, None], rsi, ema * 100.0)
    prev_final = np.vstack([
        np.array(seeds.final, dtype=float) if seeds.final else np.full(4, np.nan),
        final[:-1],
    ]) if n else final
    with np.errstate(divide="ignore", invalid="ignore"):
        roc = final / prev_final - 1.0

    roc_s = pd.DataFrame(roc, index=agg.index)
    roc3 = roc_s.rolling(3, min_periods=3).mean().to_numpy()
    roc5 = roc_s.rolling(5, min_periods=5).mean().to_numpy()

    def safe_argmax(rows: np.ndarray) -> np.ndarray:
        out = np.zeros(len(rows))
        for i, row in enumerate(rows):
            out[i] = np.nan if np.isnan(row).any() else _argmax_quad(row)
        return out

    lx, mc, mh = safe_argmax(roc), safe_argmax(roc3), safe_argmax(roc5)

    winner = np.zeros(n)
    invalid = np.isnan(fracs).any(axis=1)
    for i in range(n):
        if invalid[i]:
            winner[i] = np.nan
            continue
        if lk[i] < params.bifurcation_thr and ll[i] > params.reverse_thr:
            winner[i] = mh[i]
        elif lk[i] < params.bifurcation_thr:
            winner[i] = lj[i]
        else:
            winner[i] = ie[i]

    out = pd.DataFrame(index=agg.index)
    for j, q in enumerate(QUAD_NAMES):
        out[f"ema_{q}"] = ema[:, j]
        out[f"avg_gain_{q}"] = avg_gain[:, j]
        out[f"avg_loss_{q}"] = avg_loss[:, j]
        out[f"rsi_{q}"] = rsi[:, j]
        out[f"final_{q}"] = final[:, j]
        out[f"roc5_{q}"] = roc5[:, j]
    out["bifurcation"] = lk
    out["reverse"] = ll
    out["rsi_winner"] = lj
    out["ema_winner"] = ie
    out["roc5_winner"] = mh
    out["winner"] = winner
    return out
