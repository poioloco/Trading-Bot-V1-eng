"""Regime engine as a state machine.

- warm_up: computes once from 2005 (reuses RegimeDetector) and saves the
  state (recursive EMA/RSI seeds + a window of recent fractions).
- step: for the new days only, recomputes the votes of the 40 ETFs (short
  window from the store) and continues the winner chain from the saved
  seeds, obtaining numbers IDENTICAL to a full recompute.

The maths stays in the already validated functions (EtfSignal,
QuadAggregator, compute_winner): here there is only state orchestration.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..data.store import PointInTimeStore
from ..regime.etf_signal import EtfSignal
from ..regime.quad_aggregator import QUAD_NAMES, QuadAggregator
from ..regime.regime_detector import RegimeDetector
from ..regime.winner import RegimeParams, RegimeSeeds, compute_winner
from . import frame_from_json, frame_to_json, load_state, save_state

WINDOW = 8           # recent fractions kept (>=6 for an exact ROC5 at the edge)
FRAC_COLS = [f"frac_{q}" for q in QUAD_NAMES]


class StatefulRegime:
    def __init__(self, store: PointInTimeStore, quads: dict[str, list[str]],
                 params: RegimeParams, ma_windows: list[int],
                 vote_threshold: float) -> None:
        self.store = store
        self.quads = quads
        self.params = params
        self.ma_windows = ma_windows
        self.vote_threshold = vote_threshold
        self.aggregator = QuadAggregator(quads)
        self._symbols = [s for q in QUAD_NAMES for s in quads[q]]

    # ------------------------------------------------------------------ #
    def warm_up(self, to_date: date) -> pd.Series:
        """Full recompute from 2005; returns the winner series and leaves
        the state ready for the incremental steps."""
        det = RegimeDetector(self.store, self.quads, self.params,
                             self.ma_windows, self.vote_threshold,
                             lookback_days=9000)
        hist = det.run(as_of=to_date).history
        state = self._state_from_history(hist)
        return hist["winner"], state

    def step(self, anchor: date, state: dict, to_date: date) -> tuple[pd.Series, dict]:
        """Computes the winner for the days in (anchor, to_date] starting
        from the saved state. Numbers identical to warm_up."""
        new_dates = [d for d in self.store.get_bars_asof(self._symbols[0], to_date)
                     .index if d.date() > anchor]
        if not new_dates:
            frac_window = frame_from_json(state["frac_window"])
            return pd.Series(dtype=float), state  # nothing new

        start = anchor - timedelta(days=120)     # window for the vote MAs
        votes: dict[str, pd.Series] = {}
        for sym in self._symbols:
            bars = self.store.get_daily_bars(sym, start, to_date)
            v = EtfSignal(symbol=sym, ma_windows=self.ma_windows,
                          vote_threshold=self.vote_threshold).compute(bars)["vote"]
            votes[sym] = v[v.index > pd.Timestamp(anchor)]

        frac_window = frame_from_json(state["frac_window"])
        seed_fracs = frac_window.iloc[-1].tolist()
        new_agg = self.aggregator.aggregate(votes, seed_fracs=seed_fracs)
        new_fracs = new_agg[FRAC_COLS].dropna()
        if new_fracs.empty:
            return pd.Series(dtype=float), state

        combined = pd.concat([frac_window, new_fracs])
        pre = RegimeSeeds(ema=state["ema"], avg_gain=state["avg_gain"],
                          avg_loss=state["avg_loss"])
        win = compute_winner(combined, self.params, seeds=pre)
        emitted = win.iloc[len(frac_window):]
        new_state = self._state_from_winner(combined, win)
        return emitted["winner"], new_state

    def ensure(self, db, to_date: date) -> pd.Series:
        """Loads the state, runs warm_up or step as needed, saves, and
        returns the NEW winner series (or the full one on the first run)."""
        loaded = load_state(db, "regime")
        if loaded is None:
            winner, state = self.warm_up(to_date)
            save_state(db, "regime", to_date, state)
            return winner
        anchor, state = loaded
        winner, new_state = self.step(anchor, state, to_date)
        if not winner.empty:
            save_state(db, "regime", to_date, new_state)
        return winner

    # ------------------------------------------------------------------ #
    def _state_from_history(self, hist: pd.DataFrame) -> dict:
        return self._state_from_winner(hist[FRAC_COLS], hist)

    def _state_from_winner(self, fracs: pd.DataFrame, win: pd.DataFrame) -> dict:
        frac_window = fracs.iloc[-WINDOW:]
        anchor_row = len(fracs) - WINDOW - 1     # row BEFORE the window
        anchor_row = max(anchor_row, 0)
        ema = [float(win[f"ema_{q}"].iloc[anchor_row]) for q in QUAD_NAMES]
        ag = [float(win[f"avg_gain_{q}"].iloc[anchor_row]) for q in QUAD_NAMES]
        al = [float(win[f"avg_loss_{q}"].iloc[anchor_row]) for q in QUAD_NAMES]
        return {"frac_window": frame_to_json(frac_window),
                "ema": ema, "avg_gain": ag, "avg_loss": al}
