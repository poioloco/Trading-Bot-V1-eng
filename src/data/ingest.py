"""Data ingestion into the local database.

Two modes:
  - LIVE: downloads from the providers (Alpaca/Stooq, FRED, CBOE) when the
    API keys are present. Every bar goes through validation before entering.
  - DEMO/OFFLINE: generates a deterministic synthetic history (flagged as
    such) to run the WHOLE pipeline without network and without keys. It is
    for trying out and learning to use the bot; the numbers are NOT real
    market data.

In both cases the raw prices end up in `prices_raw` idempotently, and every
run is tracked in `runs_log` (heartbeat).
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from .calendar import missing_sessions, sessions_between
from .database import Database
from .store import PointInTimeStore
from .validation import validate_bars

logger = logging.getLogger(__name__)

# The 40 ETFs of the regime model + SPX/VIX/TDEX used by the VIX model.
DEMO_INDEX_SYMBOLS = ["^SPX", "^VIX", "TDEX"]


# ---------------------------------------------------------------------- #
# DEMO / OFFLINE                                                          #
# ---------------------------------------------------------------------- #
def _rng(symbol: str, base_seed: int) -> np.random.Generator:
    # deterministic seed per symbol: same data on every run
    h = (abs(hash(symbol)) % (2**31)) ^ base_seed
    return np.random.default_rng(h)


def _gbm_ohlcv(
    sessions: list[date], start_price: float, mu: float, sigma: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Geometric random walk with consistent OHLC around the close."""
    n = len(sessions)
    rets = rng.normal(mu / 252.0, sigma / np.sqrt(252.0), n)
    close = start_price * np.exp(np.cumsum(rets))
    intraday = np.abs(rng.normal(0, sigma / np.sqrt(252.0) / 2.0, n))
    high = close * (1.0 + intraday)
    low = close * (1.0 - intraday)
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    open_ = np.clip(open_, low, high)
    vol = rng.integers(1_000_000, 8_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=pd.DatetimeIndex(pd.to_datetime(sessions)),
    )


def _mean_reverting(
    sessions: list[date], level: float, theta: float, sigma: float, floor: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Positive mean-reverting series (for VIX / TDEX)."""
    n = len(sessions)
    x = np.empty(n)
    x[0] = level
    for i in range(1, n):
        x[i] = max(floor, x[i - 1] + theta * (level - x[i - 1]) + rng.normal(0, sigma))
    return pd.DataFrame(
        {"open": x, "high": x, "low": x, "close": x, "volume": 0.0},
        index=pd.DatetimeIndex(pd.to_datetime(sessions)),
    )


def seed_offline_demo(
    db: Database,
    etf_symbols: list[str],
    start: date,
    end: date,
    base_seed: int = 20260615,
) -> dict:
    """Fills the database with a synthetic history flagged `synthetic`."""
    sessions = sessions_between(start, end)
    if not sessions:
        raise ValueError("No trading sessions in the requested range")

    # protection: do not overwrite REAL data already present
    real = db.con.execute(
        "SELECT COUNT(*) FROM prices_raw WHERE synthetic = FALSE"
    ).fetchone()[0]
    if real > 0:
        raise RuntimeError(
            "There is already real data in the archive: the DEMO seed is blocked "
            "so as not to overwrite it. Use mode 'live' in params.yaml."
        )

    run_id = db.start_run("demo_seed", as_of=end)
    written = 0
    issues_all: list[str] = []
    try:
        # equity ETFs
        for sym in etf_symbols:
            rng = _rng(sym, base_seed)
            start_px = float(rng.uniform(20, 200))
            mu = float(rng.uniform(-0.02, 0.12))
            sigma = float(rng.uniform(0.10, 0.35))
            df = _gbm_ohlcv(sessions, start_px, mu, sigma, rng)
            res = validate_bars(df, sym)
            issues_all += res.issues
            written += db.upsert_prices(res.clean, sym, "demo", synthetic=True)

        # SPX (index, GBM)
        rng = _rng("^SPX", base_seed)
        spx = _gbm_ohlcv(sessions, 3800.0, 0.07, 0.16, rng)
        # index: we only use the close (open=high=low=close)
        for c in ("open", "high", "low"):
            spx[c] = spx["close"]
        spx["volume"] = 0.0
        db.upsert_prices(validate_bars(spx, "^SPX", is_index=True).clean, "^SPX", "demo", True)

        # VIX and TDEX (positive, mean-reverting)
        vix = _mean_reverting(sessions, level=18.0, theta=0.05, sigma=1.2, floor=9.0,
                              rng=_rng("^VIX", base_seed))
        db.upsert_prices(validate_bars(vix, "^VIX", is_index=True).clean, "^VIX", "demo", True)
        tdex = _mean_reverting(sessions, level=40.0, theta=0.04, sigma=2.5, floor=15.0,
                               rng=_rng("TDEX", base_seed))
        db.upsert_prices(validate_bars(tdex, "TDEX", is_index=True).clean, "TDEX", "demo", True)

        # one synthetic corporate action to exercise the adjustment
        if etf_symbols:
            mid = sessions[len(sessions) // 2]
            ca = pd.DataFrame(
                [{"symbol": etf_symbols[0], "date": mid, "action_type": "split", "value": 2.0}]
            )
            db.upsert_corporate_actions(ca, "demo")
            issues_all.append(f"DEMO: synthetic 2:1 split on {etf_symbols[0]} on {mid}")

        db.finish_run(run_id, "ok", f"demo seed: {written} ETF bars + indices")
    except Exception as exc:  # pragma: no cover
        db.finish_run(run_id, "error", str(exc))
        raise
    return {"sessions": len(sessions), "etf_bars": written, "issues": issues_all}


# ---------------------------------------------------------------------- #
# LIVE (network providers) - used when the keys are present               #
# ---------------------------------------------------------------------- #
def update_from_provider(
    db: Database, provider, symbols: list[str], start: date, end: date,
    is_index_symbols: set | None = None,
) -> dict:
    """Downloads and validates every symbol, then writes to the DB. Tolerates
    gaps in a single symbol (reports them) without failing the whole update."""
    is_index_symbols = is_index_symbols or {"^SPX", "^VIX"}
    run_id = db.start_run("update", as_of=end)
    written, issues, failed = 0, [], []
    try:
        for sym in symbols:
            try:
                bars = provider.get_daily_bars(sym, start, end)
            except Exception as exc:
                failed.append(sym)
                issues.append(f"{sym}: download failed ({exc})")
                continue
            res = validate_bars(bars, sym, is_index=sym in is_index_symbols)
            issues += res.issues
            written += db.upsert_prices(res.clean, sym, getattr(provider, "name", "net"))
        status = "ok" if not failed else "ok"  # partial gaps are not a fatal error
        db.finish_run(run_id, status, f"{written} bars, {len(failed)} symbols failed")
    except Exception as exc:
        db.finish_run(run_id, "error", str(exc))
        raise
    return {"written": written, "failed": failed, "issues": issues}


def make_store(db: Database, adjusted: bool = True) -> PointInTimeStore:
    return PointInTimeStore(db, adjusted=adjusted)


def incremental_update(
    db: Database, provider, symbols: list[str], end: date,
    is_index_symbols: set | None = None, fallback_start: date | None = None,
) -> dict:
    """Fills ONLY the missing days: for each symbol it starts from the day
    after the last date already in the archive and downloads up to `end`.
    This avoids re-downloading the whole history (cause of the too-much-data-
    per-IP errors) and does NOT touch the historical data already present
    (your imports)."""
    is_index_symbols = is_index_symbols or {"^SPX", "^VIX"}
    fallback_start = fallback_start or date(2005, 1, 1)
    run_id = db.start_run("incremental", as_of=end)
    written, skipped, failed, issues = 0, [], [], []
    pending = []   # the source answers but has not published the new days yet
    try:
        for sym in symbols:
            rng = db.con.execute(
                "SELECT MAX(date) FROM prices_raw WHERE symbol = ?", [sym]
            ).fetchone()
            last = rng[0] if rng else None
            if last is not None:
                gap_start = (pd.Timestamp(last) + pd.Timedelta(days=1)).date()
                miss = missing_sessions([], gap_start, end)
                if not miss:
                    skipped.append(sym)        # already up to date
                    continue
                start = miss[0]
            else:
                start = fallback_start         # new symbol: the whole history
            try:
                bars = provider.get_daily_bars(sym, start, end)
            except Exception as exc:
                failed.append(sym)
                issues.append(f"{sym}: download {start}->{end} failed ({exc})")
                continue
            # extra defence: write ONLY the requested days, never the history
            # (some sources ignore the date filter and send everything)
            if bars is not None and not bars.empty:
                idx = pd.to_datetime(bars.index)
                bars = bars[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
            if bars is None or bars.empty:
                pending.append(sym)
                continue
            res = validate_bars(bars, sym, is_index=sym in is_index_symbols)
            issues += res.issues
            written += db.upsert_prices(res.clean, sym, getattr(provider, "name", "net"))
        pending_txt = (f", {len(pending)} not yet published ({', '.join(pending)})"
                       if pending else "")
        db.finish_run(
            run_id, "ok",
            f"incremental: +{written} bars, {len(skipped)} already up to date"
            f"{pending_txt}, {len(failed)} failed",
        )
    except Exception as exc:
        db.finish_run(run_id, "error", str(exc))
        raise
    return {"written": written, "skipped": skipped, "failed": failed,
            "pending": pending, "issues": issues}
