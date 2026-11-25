"""Per-ETF vote - exact replica of one Excel ETF sheet.

Decoded from the 40-ETF workbook ("Bottoms-Up Regime Segmentation - 40ETFs_F",
shared formulas verified on the TLT sheet and spot-checked across sheets):

    F  log_ret   = LN(close_t/close_{t-1})            (analytics only)
    G..L MA_k    = AVERAGE(close, last k bars)         rolling SMA incl. today
    M..R pct_k   = close/MA_k - 1                      distance from each MA
    S  summed    = SUM(pct_5..pct_63)
    T  treatment = summed * 100
    U  u3        = AVERAGE(treatment, last 3 bars)     "3-DMA"
    W  vote      = IF(u3 >  $W$3, +1,
                   IF(u3 < -$W$3, -1, 0))              $W$3 = 5 on every sheet
    AP = W                                             (the exported vote)

IMPORTANT - effective MA windows: the sheet headers label the six MAs
5/8/13/21/42/63, but the actual formulas use windows 5/8/13/20/41/63
(e.g. J23 = AVERAGE(E4:E23) = 20 cells, K44 = AVERAGE(E4:E44) = 41 cells).
The implementation below replicates the formulas, not the labels.

During warm-up (fewer bars than the longest window) the Excel cells hold
errors and the Macro sheet's IFERROR turns them into blanks, which the
aggregation then ignores; here those votes are NaN for the same effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# EFFECTIVE windows of the formulas (sheet labels: 5, 8, 13, 21, 42, 63)
MA_WINDOWS = [5, 8, 13, 20, 41, 63]
VOTE_THRESHOLD = 5.0  # $W$3, identical on every sheet checked


@dataclass
class EtfSignal:
    """Computes the per-ETF vote series (column AP) from daily bars."""

    symbol: str
    ma_windows: list[int] = field(default_factory=lambda: list(MA_WINDOWS))
    vote_threshold: float = VOTE_THRESHOLD

    # ------------------------------------------------------------------ #
    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """`bars` needs a 'close' column, ascending dates. Returns the
        full chain plus the final `vote` in {-1, 0, +1} (NaN in warm-up)."""
        close = bars["close"].astype(float)
        out = pd.DataFrame(index=bars.index)
        out["close"] = close

        pct_cols = []
        for w in self.ma_windows:
            ma = close.rolling(w, min_periods=w).mean()
            out[f"ma_{w}"] = ma
            col = f"pct_{w}"
            out[col] = close / ma - 1.0
            pct_cols.append(col)

        # SUM over the six distances; NaN until all six MAs exist
        out["summed"] = out[pct_cols].sum(axis=1, min_count=len(pct_cols))
        out["treatment"] = out["summed"] * 100.0
        out["u3"] = out["treatment"].rolling(3, min_periods=3).mean()

        thr = self.vote_threshold
        vote = pd.Series(np.nan, index=out.index)
        vote[out["u3"] > thr] = 1.0
        vote[out["u3"] < -thr] = -1.0
        vote[(out["u3"] <= thr) & (out["u3"] >= -thr)] = 0.0
        out["vote"] = vote  # NaN where u3 is NaN (warm-up), like the Excel
        return out

    # ------------------------------------------------------------------ #
    def latest_vote(self, bars: pd.DataFrame) -> float:
        return float(self.compute(bars)["vote"].iloc[-1])
