"""Quad-dependent VIX model parameters.

Decoded from the Excel "Inputs" sheet (B3:I6) of
"Tactical Trading - Toy Model 05-05-2026.xlsb". Every indicator chain in
VIX_Data selects its parameter per-day based on the regime quad in column Z:

    =IF(Z=1, Inputs!$B$3, IF(Z=2, Inputs!$B$4, IF(Z=3, Inputs!$B$5, Inputs!$B$6)))

Quad numbering (Alerts sheet): 1=Disinflation, 2=Reflation,
3=Stagflation, 4=Deflation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuadParams:
    rs_ema: int          # Inputs!B  "Gain/Loss RS EMA"      (RSI EMA period)
    rsi_smoother: int    # Inputs!C  "RSI Smoother"          (EMA period on rescaled RSI)
    hv_lookback: int     # Inputs!D  "HV's LookBack for VRP" (days of log returns)
    vrp_ema: int         # Inputs!E  "VRP's Dynamic EMA"     (EMA period on raw VRP)
    rsi_long_thr: float  # Inputs!F  "RSI longVOL"   (rsi_smooth <  thr -> long-vol vote)
    vrp_long_thr: float  # Inputs!G  "VRP longVOL"   (vrp_smooth <  thr -> long-vol vote)
    rsi_short_thr: float # Inputs!H  "RSI shortVOL"  (rsi_smooth >  thr -> buy-SPX vote)
    vrp_short_thr: float # Inputs!I  "VRP shortVOL"  (vrp_smooth >  thr -> buy-SPX vote)

    @property
    def rs_alpha(self) -> float:
        return 2.0 / (self.rs_ema + 1.0)

    @property
    def rsi_smooth_alpha(self) -> float:
        return 2.0 / (self.rsi_smoother + 1.0)

    @property
    def vrp_alpha(self) -> float:
        return 2.0 / (self.vrp_ema + 1.0)


# AUTHORITATIVE values from the midterm workbook (Inputs B3:I6, screenshot
# 03-07-2026): the set optimised by the author's backtests. It replaces the
# set of the "Toy Model 05-05-2026.xlsb". NB: transcribed from a screenshot,
# pending the .xlsb file for the cell-by-cell check.
DEFAULT_QUAD_PARAMS: dict[int, QuadParams] = {
    1: QuadParams(10, 3, 6, 13, -3.0, 9.0, 2.0, 1.0),
    2: QuadParams(2, 2, 18, 18, 9.0, 1.0, 0.0, -10.0),
    3: QuadParams(19, 2, 16, 10, 0.0, 3.0, -4.0, 2.0),
    4: QuadParams(3, 2, 12, 4, 9.0, -3.0, -2.0, -1.0),
}

QUAD_NAME_TO_NUMBER = {
    "disinflation": 1,
    "reflation": 2,
    "stagflation": 3,
    "deflation": 4,
}


def quad_params_from_config(cfg: dict | None) -> dict[int, QuadParams]:
    """Build the quad->params map from config.yaml (falls back to defaults)."""
    if not cfg:
        return dict(DEFAULT_QUAD_PARAMS)
    out: dict[int, QuadParams] = {}
    for quad_str, p in cfg.items():
        out[int(quad_str)] = QuadParams(
            rs_ema=int(p["rs_ema"]),
            rsi_smoother=int(p["rsi_smoother"]),
            hv_lookback=int(p["hv_lookback"]),
            vrp_ema=int(p["vrp_ema"]),
            rsi_long_thr=float(p["rsi_long_thr"]),
            vrp_long_thr=float(p["vrp_long_thr"]),
            rsi_short_thr=float(p["rsi_short_thr"]),
            vrp_short_thr=float(p["vrp_short_thr"]),
        )
    missing = set(DEFAULT_QUAD_PARAMS) - set(out)
    if missing:
        raise ValueError(f"quad_params in config missing quads: {sorted(missing)}")
    return out
