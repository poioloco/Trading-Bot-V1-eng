"""Quad aggregation - exact replica of the Regime sheet, columns HL..HT.

Decoded formulas (verified at rows 4996+ of the Regime sheet):

    HL..HO score_q = SUMIFS(votes, tags, quad, votes, ">0")
                     = sum of the POSITIVE votes among the quad's ETFs
                       (votes are +1/0/-1, so this counts the bullish ones)
    HP total       = SUM(score_1..score_4)
    HQ..HT frac_q  = IFERROR(score_q / total, previous frac_q)
                     (carry-forward when total = 0)

Quad numbering: 1=Disinflation, 2=Reflation, 3=Stagflation, 4=Deflation.
NaN votes (ETF warm-up; the Excel Macro sheet maps errors to blanks) are
excluded exactly like blanks in SUMIFS.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

QUAD_NAMES = ["disinflation", "reflation", "stagflation", "deflation"]
QUAD_NUMBERS = {name: i + 1 for i, name in enumerate(QUAD_NAMES)}


class QuadAggregator:
    def __init__(self, quads: dict[str, list[str]]) -> None:
        unknown = [q for q in quads if q not in QUAD_NAMES]
        if unknown:
            raise ValueError(f"Unknown quad names: {unknown}")
        missing = [q for q in QUAD_NAMES if q not in quads]
        if missing:
            raise ValueError(f"Missing quads in config: {missing}")
        self.quads = quads

    # ------------------------------------------------------------------ #
    def aggregate(
        self, votes: dict[str, pd.Series], seed_fracs: list[float] | None = None
    ) -> pd.DataFrame:
        """`votes`: symbol -> vote Series (+1/0/-1, NaN in warm-up).
        Returns a frame with score_<quad>, total and frac_<quad> columns.
        `seed_fracs` optionally seeds the carry-forward state (golden test
        / incremental runs)."""
        vote_frame = pd.DataFrame(votes).sort_index()
        out = pd.DataFrame(index=vote_frame.index)

        for quad, symbols in self.quads.items():
            members = [s for s in symbols if s in vote_frame.columns]
            absent = [s for s in symbols if s not in vote_frame.columns]
            if absent:
                logger.warning("Quad '%s': no data for %s", quad, absent)
            if not members:
                raise ValueError(f"Quad '{quad}' has no usable ETF votes")
            block = vote_frame[members]
            out[f"score_{quad}"] = block.where(block > 0).sum(axis=1).fillna(0.0)

        score_cols = [f"score_{q}" for q in QUAD_NAMES]
        out["total"] = out[score_cols].sum(axis=1)

        # fractions with carry-forward (IFERROR -> previous value)
        prev = list(seed_fracs) if seed_fracs else [np.nan] * 4
        fracs = np.zeros((len(out), 4))
        totals = out["total"].to_numpy()
        scores = out[score_cols].to_numpy()
        for i in range(len(out)):
            if totals[i] > 0:
                prev = list(scores[i] / totals[i])
            fracs[i] = prev
        for j, quad in enumerate(QUAD_NAMES):
            out[f"frac_{quad}"] = fracs[:, j]
        return out
