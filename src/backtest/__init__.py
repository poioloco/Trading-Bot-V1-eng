"""Backtest = the warm-up of the state engines over the whole history.

No separate engine: StatefulRegime/StatefulVix are used, the same classes as
live. The point-in-time store guarantees that the calculation at a date only
sees data up to that date (anti-lookahead). Here we build the signal
timeline, the positions and an illustrative timing curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..data.store import PointInTimeStore
from ..regime.quad_aggregator import QUAD_NAMES
from ..regime.winner import RegimeParams
from ..vix.quad_params import QuadParams
from ..vix.signal_generator import SignalConfig
from ..engine.regime_engine import StatefulRegime
from ..engine.vix_engine import StatefulVix

EVENT_COLS = ["buy_vix", "sell_vix", "buy_spx", "sell_spx"]


@dataclass
class BacktestResult:
    timeline: pd.DataFrame             # one row per trading day
    stats: dict
    equity: pd.DataFrame               # illustrative curves (market leg)
    notes: list[str] = field(default_factory=list)


def _positions(buy: np.ndarray, sell: np.ndarray) -> np.ndarray:
    """Position machine: BUY opens (1), SELL closes (0), otherwise hold."""
    pos = np.zeros(len(buy))
    cur = 0.0
    for i in range(len(buy)):
        if buy[i]:
            cur = 1.0
        elif sell[i]:
            cur = 0.0
        pos[i] = cur
    return pos


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


class Backtester:
    def __init__(self, store: PointInTimeStore, quads: dict[str, list[str]],
                 regime_params: RegimeParams, ma_windows: list[int],
                 vote_threshold: float, params_by_quad: dict[int, QuadParams],
                 cfg: SignalConfig, long_asset: str, vol_asset: str,
                 spx_symbol: str = "^SPX", vix_symbol: str = "^VIX") -> None:
        self.long_asset = long_asset
        self.vol_asset = vol_asset
        self.regime = StatefulRegime(store, quads, regime_params, ma_windows, vote_threshold)
        self.vix = StatefulVix(store, params_by_quad, cfg, spx_symbol, vix_symbol)

    # ------------------------------------------------------------------ #
    def run(self, end: date) -> BacktestResult:
        winner, _ = self.regime.warm_up(end)
        ind, sig, _ = self.vix.warm_up(winner, end)

        df = pd.DataFrame(index=ind.index)
        df["quad"] = ind["quad"].astype(int)
        df["regime"] = df["quad"].map(lambda q: QUAD_NAMES[q - 1])
        df["spx"] = ind["spx"]
        df["vix"] = ind["vix"]
        for c in ("rsi_smooth", "vrp_smooth", "tdex_smooth"):
            df[c] = ind[c]
        for c in EVENT_COLS:
            df[c] = sig[c].astype(bool)
        # STATE (level, not event) of the market exit: column U/T of the
        # Excel (rsi<0 | vrp<0 | tdex_off). Needed by the ternary vote of the
        # market leg in the portfolio (in position=+1, active exit=-1,
        # otherwise abstain). NaN/missing -> False (no forced exit).
        if "sell_spx_raw" in sig.columns:
            df["spx_sell_state"] = sig["sell_spx_raw"].fillna(False).astype(bool)

        df["pos_market"] = _positions(df["buy_spx"].to_numpy(), df["sell_spx"].to_numpy())
        df["pos_volatility"] = _positions(df["buy_vix"].to_numpy(), df["sell_vix"].to_numpy())
        # per-asset signal of the day (discrete action)
        df[self.long_asset] = np.where(df["buy_spx"], "BUY",
                                np.where(df["sell_spx"], "SELL", "HOLD"))
        df[self.vol_asset] = np.where(df["buy_vix"], "BUY",
                              np.where(df["sell_vix"], "SELL", "HOLD"))

        equity, eq_stats = self._equity(df)
        stats = self._stats(df) | eq_stats
        notes = [
            "The backtest uses the same classes as live (StatefulRegime/StatefulVix).",
            "The curve is ILLUSTRATIVE: it uses SPX returns as a proxy for the timing "
            "of the market leg, NOT the real returns of "
            f"{self.long_asset} (leveraged). The volatility leg ({self.vol_asset}) "
            "is only reported as a signal count: a realistic P&L requires the "
            "instrument's real historical series.",
        ]
        return BacktestResult(df, stats, equity, notes)

    # ------------------------------------------------------------------ #
    def _equity(self, df: pd.DataFrame):
        ret = df["spx"].pct_change().fillna(0.0)
        # position active from the day AFTER the signal (EOD bot, T+1 entry)
        strat = df["pos_market"].shift(1).fillna(0.0) * ret
        eq = pd.DataFrame(index=df.index)
        eq["buy_and_hold_spx"] = (1.0 + ret).cumprod()
        eq["timing_market"] = (1.0 + strat).cumprod()
        stats = {
            "bh_multiple": float(eq["buy_and_hold_spx"].iloc[-1]),
            "timing_multiple": float(eq["timing_market"].iloc[-1]),
            "bh_max_drawdown": _max_drawdown(eq["buy_and_hold_spx"]),
            "timing_max_drawdown": _max_drawdown(eq["timing_market"]),
        }
        return eq, stats

    def _stats(self, df: pd.DataFrame) -> dict:
        n = len(df)
        reg = (df["regime"].value_counts(normalize=True) * 100).round(1).to_dict()
        return {
            "start": df.index[0].date().isoformat(),
            "end": df.index[-1].date().isoformat(),
            "days": n,
            "regime_distribution_pct": reg,
            "events": {c: int(df[c].sum()) for c in EVENT_COLS},
            "time_in_market_pct": round(float(df["pos_market"].mean()) * 100, 1),
            "time_in_volatility_pct": round(float(df["pos_volatility"].mean()) * 100, 1),
        }

    # ------------------------------------------------------------------ #
    def walk_forward_check(self, dates: list[date], full: pd.DataFrame) -> dict:
        """Anti-lookahead: recomputes the signal 'as of date D' (only data
        <= D) and compares it with the full backtest. It must match."""
        worst_quad = 0
        mism_events = 0
        checked = 0
        for d in dates:
            w, _ = self.regime.warm_up(d)
            ind, sig, _ = self.vix.warm_up(w, d)
            ts = ind.index[-1]
            if ts not in full.index:
                continue
            checked += 1
            worst_quad = max(worst_quad, abs(int(ind["quad"].iloc[-1]) - int(full.loc[ts, "quad"])))
            for c in EVENT_COLS:
                if bool(sig[c].iloc[-1]) != bool(full.loc[ts, c]):
                    mism_events += 1
        return {"dates_checked": checked, "max_diff_quad": worst_quad,
                "mismatched_events": mism_events}
