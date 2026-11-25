"""Data validation: "better no data than false data".

Every bar, from any source, passes through here before entering the
database. The checks are designed to block false data (which generates
false signals), not to silently "fix" it. When a value is suspicious it is
reported (and, if critical, discarded), never quietly corrected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    clean: pd.DataFrame              # surviving bars (sorted, de-duplicated)
    issues: list[str] = field(default_factory=list)  # human-readable sentences
    dropped: int = 0                 # discarded rows

    @property
    def ok(self) -> bool:
        return self.dropped == 0 and not self.issues


def validate_bars(
    df: pd.DataFrame,
    symbol: str,
    is_index: bool = False,
    max_daily_move: float = 0.60,
    known_split_dates: set | None = None,
) -> ValidationResult:
    """Checks OHLC consistency, invalid prices and outliers.

    - is_index: for indices/single-value series (^SPX, ^VIX, TDEX) volume
      may be 0 and differentiated OHLC is not required.
    - max_daily_move: daily change above which the bar is "suspicious";
      if the date matches a known split it is NOT an error.
    """
    issues: list[str] = []
    if df is None or df.empty:
        return ValidationResult(clean=pd.DataFrame(), issues=[f"{symbol}: no data"], dropped=0)

    d = df.copy()
    d.index = pd.to_datetime(d.index).normalize()
    before = len(d)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    if len(d) < before:
        issues.append(f"{symbol}: removed {before - len(d)} duplicate dates")

    cols = ["open", "high", "low", "close"]
    # 1) null/negative/zero prices -> discarded (they are false data)
    bad_price = (d[cols] <= 0).any(axis=1) | d[cols].isna().any(axis=1)
    if bad_price.any():
        issues.append(f"{symbol}: {int(bad_price.sum())} bars with null/<=0 prices discarded")

    # 2) OHLC consistency: low <= min(open,close) <= max(open,close) <= high
    lo_ok = d["low"] <= d[["open", "close"]].min(axis=1) + 1e-9
    hi_ok = d["high"] >= d[["open", "close"]].max(axis=1) - 1e-9
    lh_ok = d["low"] <= d["high"] + 1e-9
    incoherent = ~(lo_ok & hi_ok & lh_ok) & ~bad_price
    if incoherent.any():
        issues.append(f"{symbol}: {int(incoherent.sum())} inconsistent OHLC bars discarded")

    drop_mask = bad_price | incoherent
    clean = d[~drop_mask].copy()

    # 3) outliers: huge daily move not explained by a known split
    if len(clean) > 1:
        ret = clean["close"].pct_change().abs()
        splits = {pd.Timestamp(s).normalize() for s in (known_split_dates or set())}
        suspect = clean.index[(ret > max_daily_move)]
        unexplained = [ts for ts in suspect if ts not in splits]
        if unexplained:
            sample = ", ".join(pd.Timestamp(t).date().isoformat() for t in unexplained[:3])
            issues.append(
                f"{symbol}: {len(unexplained)} daily jumps > {max_daily_move:.0%} "
                f"not explained by splits (e.g. {sample}) - to be checked"
            )

    return ValidationResult(clean=clean, issues=issues, dropped=int(drop_mask.sum()))


def cross_check(
    a: pd.Series, b: pd.Series, tol: float = 0.02
) -> list[str]:
    """Cross-check between two sources for the same value (e.g. close).
    Reports the dates where they differ by more than `tol` (2% default)."""
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return ["cross-check impossible: no dates in common"]
    rel = (a.loc[common] - b.loc[common]).abs() / b.loc[common].replace(0, np.nan)
    mism = rel[rel > tol].dropna()
    if mism.empty:
        return []
    sample = ", ".join(pd.Timestamp(t).date().isoformat() for t in mism.index[:3])
    return [f"cross-check: {len(mism)} dates beyond {tol:.0%} (e.g. {sample})"]
