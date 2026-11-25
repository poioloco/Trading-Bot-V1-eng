"""VIX indicator engine - exact replica of the VIX_Data sheet chains.

Column mapping (VIX_Data, formulas verified at rows 3000 and 5350):

  RSI chain (quad-dependent EMA RSI on SPX, on PRICE CHANGES not %):
    AC change    = spx_t - spx_{t-1}
    AD gain      = max(change, 0)
    AE loss      = abs(min(change, 0))
    AG alpha     = 2/(rs_ema+1)                       (rs_ema from quad)
    AH avg_gain  = (gain - avg_gain_prev)*alpha + avg_gain_prev
    AI avg_loss  = (loss - avg_loss_prev)*alpha + avg_loss_prev
    AK rsi_raw   = 100 if avg_loss==0 else 100 - 100/(1 + avg_gain/avg_loss)
    AL rsi_10d   = (rsi_raw - 50)/5
    AO rsi_smooth= EMA(rsi_10d, alpha=2/(rsi_smoother+1))   (quad-dependent)

  VRP chain:
    AR log_ret   = ln(spx_t / spx_{t-1})
    AS hv        = STDEV.S(last hv_lookback log_rets) * sqrt(252) * 100
                   (0.0 when the window is not fully available - Excel IFERROR)
    AT vrp_raw   = vix_t - hv
    AW vrp_smooth= EMA(vrp_raw, alpha=2/(vrp_ema+1))        (quad-dependent)

  TDEX chain (cols O..T):
    P mean21     = AVERAGE(tdex, 21 incl. today)
    Q std21      = STDEV (sample) over the same 21
    R z          = (tdex - mean21)/std21
    S z3         = -z / (1/3)  = -3*z
    T tdex_smooth= AVERAGE(z3, last 5 incl. today)

All recursive chains in Excel are seeded by EMPTY cells at row 3 (=0) and
run from Jan-2005. `seeds` lets the caller restore mid-stream state (used
by the golden regression test and by incremental production runs).
`history` provides extra leading raw rows used only to fill rolling
windows (log returns, TDEX 21d+5d) without emitting output for them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .quad_params import QuadParams

TRADING_DAYS = 252


@dataclass
class ChainSeeds:
    avg_gain: float = 0.0
    avg_loss: float = 0.0
    rsi_smooth: float = 0.0
    vrp_smooth: float = 0.0


def compute_indicators(
    df: pd.DataFrame,
    params_by_quad: dict[int, QuadParams],
    seeds: ChainSeeds | None = None,
    history: pd.DataFrame | None = None,
    seed_at_boundary: bool = False,
    trend_window: int = 200,
) -> pd.DataFrame:
    """Run the VIX_Data chains over `df` (columns: vix, spx, tdex, quad).

    `df` must be sorted by date ascending. Returns a frame aligned with
    `df` containing every intermediate and final indicator column.
    """
    required = {"vix", "spx", "tdex", "quad"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_indicators: missing columns {sorted(missing)}")

    seeds = seeds or ChainSeeds()
    n_hist = 0
    if history is not None and not history.empty:
        full = pd.concat([history[["vix", "spx", "tdex", "quad"]], df], axis=0)
        n_hist = len(history)
    else:
        full = df

    spx = full["spx"].astype(float).to_numpy()
    vix = full["vix"].astype(float).to_numpy()
    tdex = full["tdex"].astype(float).to_numpy()
    quad = full["quad"].to_numpy()
    n = len(full)

    # ---- price-change RSI chain (recursive, quad-dependent) ----------
    change = np.full(n, np.nan)
    change[1:] = spx[1:] - spx[:-1]
    gain = np.where(change > 0, change, 0.0)
    loss = np.where(change < 0, np.abs(change), 0.0)
    # first row of the run has no previous close -> Excel blank => 0 change
    gain[0] = 0.0
    loss[0] = 0.0

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi_raw = np.zeros(n)
    rsi_10d = np.zeros(n)
    rsi_smooth = np.zeros(n)
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(spx[1:] / spx[:-1])
    hv = np.zeros(n)
    vrp_raw = np.zeros(n)
    vrp_smooth = np.zeros(n)

    g_prev = seeds.avg_gain
    l_prev = seeds.avg_loss
    rs_prev = seeds.rsi_smooth
    vw_prev = seeds.vrp_smooth

    # With seed_at_boundary the seeds are the state of the LAST row of
    # history: history rows are not re-processed by the recursive chains
    # (they only serve as a run-up for changes and rolling windows), so the
    # new days continue EXACTLY from the saved state.
    start = n_hist if (seed_at_boundary and n_hist > 0) else 0

    for i in range(start, n):
        p = params_by_quad[int(quad[i])]
        a_rs = p.rs_alpha
        g_prev = (gain[i] - g_prev) * a_rs + g_prev
        l_prev = (loss[i] - l_prev) * a_rs + l_prev
        avg_gain[i] = g_prev
        avg_loss[i] = l_prev
        if l_prev == 0.0:
            rsi_raw[i] = 100.0
        else:
            rsi_raw[i] = 100.0 - 100.0 / (1.0 + g_prev / l_prev)
        rsi_10d[i] = (rsi_raw[i] - 50.0) / 5.0
        rs_prev = (rsi_10d[i] - rs_prev) * p.rsi_smooth_alpha + rs_prev
        rsi_smooth[i] = rs_prev

        # HV over the last `hv_lookback` log returns (incl. today's)
        k = p.hv_lookback
        window = log_ret[i - k + 1 : i + 1] if i - k + 1 >= 0 else np.array([])
        if len(window) == k and not np.isnan(window).any():
            hv[i] = float(np.std(window, ddof=1)) * math.sqrt(TRADING_DAYS) * 100.0
        else:
            hv[i] = 0.0  # Excel IFERROR(...)=0 during warm-up
        vrp_raw[i] = vix[i] - hv[i]
        vw_prev = (vrp_raw[i] - vw_prev) * p.vrp_alpha + vw_prev
        vrp_smooth[i] = vw_prev

    # ---- TDEX z-score chain (rolling, not recursive) ------------------
    tdex_s = pd.Series(tdex)
    mean21 = tdex_s.rolling(21, min_periods=21).mean()
    std21 = tdex_s.rolling(21, min_periods=21).std(ddof=1)
    z = (tdex_s - mean21) / std21
    z3 = -z / (1.0 / 3.0)
    tdex_smooth = z3.rolling(5, min_periods=5).mean()

    # ---- Trend SMA on SPX (for the signal generator's trend filter) -----
    # Computed on history+df like the other rolling windows: NaN until the
    # window is full (the trend veto treats NaN as "do not fire").
    spx_sma = pd.Series(spx).rolling(trend_window,
                                     min_periods=trend_window).mean()

    out = pd.DataFrame(
        {
            "change": change,
            "gain": gain,
            "loss": loss,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "rsi_raw": rsi_raw,
            "rsi_10d": rsi_10d,
            "rsi_smooth": rsi_smooth,
            "log_ret": log_ret,
            "hv": hv,
            "vrp_raw": vrp_raw,
            "vrp_smooth": vrp_smooth,
            "tdex_z": z.to_numpy(),
            "tdex_z3": z3.to_numpy(),
            "tdex_smooth": tdex_smooth.to_numpy(),
            "spx_close": spx,
            "spx_sma": spx_sma.to_numpy(),
            "quad": quad,
        },
        index=full.index,
    )
    return out.iloc[n_hist:]


# ---------------------------------------------------------------------- #
# Contango / term-structure metrics (Alerts dashboard)                    #
# ---------------------------------------------------------------------- #
def contango_metrics(
    vxf1: float, vxf2: float, vxf3: float, vix: float | None = None,
    vix3m: float | None = None,
) -> dict[str, float | None]:
    """Term-structure metrics. "Contango Next" = VXF3/VXF2 - 1 is the
    formula confirmed in the Alerts sheet (E14)."""
    out: dict[str, float | None] = {
        "contango_f1_f2": vxf2 / vxf1 - 1.0 if vxf1 else None,
        "contango_next": vxf3 / vxf2 - 1.0 if vxf2 else None,
        "roll_yield_spot": vxf1 / vix - 1.0 if vix else None,
    }
    if vix and vix3m:
        out["vix_vix3m"] = vix / vix3m - 1.0
    return out
