"""VIX engine as a state machine (incremental and EXACT).

Uses the regime of EACH day. The saved state is:
  - tail: last raw rows (run-up for changes and rolling windows);
  - seeds_last: state of the EMA chains at the last row (exact continuity
    at the boundary thanks to compute_indicators(..., seed_at_boundary=True));
  - ind_tail: last indicator rows needed by the signal transitions
    (buy/sell use the previous day's state);
  - pos_tail_start: position (vix/spx) BEFORE the first row of ind_tail.
    In majority mode the exit events are anchored to the position: without
    this piece of state, a position opened before the tail would never get
    its exit when the signals are regenerated on the short tail
    (step/latest). Old persisted states (without the key) start flat: a
    `signals --recompute` realigns everything.
The maths stays in compute_indicators/generate_signals (validated).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..data.staleness import ffill_with_guard
from ..data.store import PointInTimeStore
from ..vix.indicators import ChainSeeds, compute_indicators
from ..vix.quad_params import QuadParams
from ..vix.signal_generator import SignalConfig, generate_signals
from . import frame_from_json, frame_to_json

TAIL = 31            # raw rows kept (>= max TDEX/HV window + 1);
                     # with the trend filter active the tail widens to
                     # trend_window+5 (the SMA needs its run-up)
SIG_TAIL = 3         # indicator rows for the signal transitions
RAW_COLS = ["vix", "spx", "tdex", "quad"]
IND_TAIL_COLS = ["rsi_smooth", "vrp_smooth", "tdex_smooth", "quad",
                 "spx_close", "spx_sma"]
SEED_KEYS = ("avg_gain", "avg_loss", "rsi_smooth", "vrp_smooth")
POS_DEFAULT = {"vix": False, "spx": False, "vix_state": False,
               "spx_state": False, "spx_base": False}


def _pos_before_tail(sig: pd.DataFrame, fallback: dict) -> dict:
    """Position AND state BEFORE the first row of ind_tail (= last SIG_TAIL
    rows of `sig`): it is row -(SIG_TAIL+1) of pos_vix/pos_spx and
    buy_vix_state/buy_spx_state. generate_signals needs the states to
    compute the BUY edges EXACTLY on the first row of the tail (a state
    already true at the boundary is not an edge). If the history is shorter
    than the tail, the boundary has not changed compared with `fallback`."""
    if len(sig) <= SIG_TAIL:
        return dict(fallback)
    row = sig.iloc[len(sig) - SIG_TAIL - 1]
    return {"vix": bool(row["pos_vix"]), "spx": bool(row["pos_spx"]),
            "vix_state": bool(row["buy_vix_state"]),
            "spx_state": bool(row["buy_spx_state"]),
            # BASE position (before the trend filter): without it, at the
            # boundary the step engine would lose an open but vetoed gate position
            "spx_base": bool(row.get("pos_spx_base", row["pos_spx"]))}


class StatefulVix:
    def __init__(self, store: PointInTimeStore, params_by_quad: dict[int, QuadParams],
                 cfg: SignalConfig, spx_symbol: str = "^SPX",
                 vix_symbol: str = "^VIX") -> None:
        self.store = store
        self.params_by_quad = params_by_quad
        self.cfg = cfg
        self.spx_symbol = spx_symbol
        self.vix_symbol = vix_symbol
        # Window of the trend SMA: the spx_close/spx_sma columns are ALWAYS
        # computed (zero cost); the raw tail only widens if the filter is
        # active, to have the full run-up of the SMA.
        self.trend_window = int(getattr(cfg, "trend_window", 200) or 200)
        self.tail = (max(TAIL, self.trend_window + 5)
                     if (getattr(cfg, "trend_enabled", False)
                         or getattr(cfg, "market_mode", "classic") == "trend")
                     else TAIL)

    # ------------------------------------------------------------------ #
    def _raw_frame(self, quad: pd.Series, start: date, to_date: date) -> pd.DataFrame:
        vix = self.store.get_daily_bars(self.vix_symbol, start, to_date)["close"]
        spx = self.store.get_daily_bars(self.spx_symbol, start, to_date)["close"]
        try:
            tdex = self.store.tdex_series(to_date)
            tdex = tdex[tdex.index >= pd.Timestamp(start)]
        except Exception:
            tdex = pd.Series(dtype=float)
        frame = pd.DataFrame({"vix": vix, "spx": spx}).dropna()
        # ffill with staleness guard: short gaps (FRED holidays) filled;
        # feed stopped for more than cfg.tdex_stale_days -> NaN, the TDEX
        # leg abstains instead of voting on a projected value.
        frame["tdex"] = ffill_with_guard(
            tdex, frame.index, getattr(self.cfg, "tdex_stale_days", 10))
        frame["quad"] = quad.reindex(frame.index)
        return frame.dropna(subset=["vix", "spx", "quad"])[RAW_COLS]

    def _seeds_at(self, ind: pd.DataFrame, row: int) -> dict:
        return {k: float(ind[k].iloc[row]) for k in SEED_KEYS}

    def _state(self, frame: pd.DataFrame, ind: pd.DataFrame,
               sig: pd.DataFrame) -> dict:
        return {
            "tail": frame_to_json(frame[RAW_COLS].iloc[-self.tail:]),
            "seeds_last": self._seeds_at(ind, -1),
            "ind_tail": frame_to_json(ind[IND_TAIL_COLS].iloc[-SIG_TAIL:]),
            # position (vix/spx) BEFORE the first row of ind_tail: needed by
            # generate_signals(pos0=...) so that exits anchored to the
            # position also fire when the entry happened before the tail
            # (majority). In the warm-up the history starts flat.
            "pos_tail_start": _pos_before_tail(sig, POS_DEFAULT),
        }

    # ------------------------------------------------------------------ #
    def warm_up(self, quad_full: pd.Series, to_date: date):
        start = pd.Timestamp(quad_full.index.min()).date()
        frame = self._raw_frame(quad_full, start, to_date)
        ind = compute_indicators(frame, self.params_by_quad,
                                 trend_window=self.trend_window)
        sig = generate_signals(ind, self.params_by_quad, self.cfg)
        state = self._state(frame, ind, sig)
        ind = ind.join(frame[["vix", "spx"]])
        return ind, sig, state

    def step(self, anchor: date, state: dict, quad_new: pd.Series, to_date: date):
        if quad_new.empty:
            return pd.DataFrame(), pd.DataFrame(), state
        tail = frame_from_json(state["tail"])
        new_raw = self._raw_frame(quad_new, anchor + timedelta(days=1), to_date)
        new_raw = new_raw[new_raw.index > pd.Timestamp(anchor)]
        if new_raw.empty:
            return pd.DataFrame(), pd.DataFrame(), state

        seeds_last = ChainSeeds(**state["seeds_last"])
        ind_new = compute_indicators(new_raw, self.params_by_quad,
                                     seeds=seeds_last, history=tail,
                                     seed_at_boundary=True,          # EXACT
                                     trend_window=self.trend_window)
        # signals: continuity of the transitions with the previous day is
        # needed AND the position at the boundary (backward compat: old
        # states without the key start flat; a `signals --recompute`
        # realigns everything)
        ind_tail = frame_from_json(state["ind_tail"])
        pos0 = state.get("pos_tail_start", POS_DEFAULT)
        ind_for_sig = pd.concat([ind_tail, ind_new])
        sig_all = generate_signals(ind_for_sig, self.params_by_quad, self.cfg,
                                   pos0=pos0)
        sig_new = sig_all.iloc[len(ind_tail):]

        ind_join = ind_new.join(new_raw[["vix", "spx"]])
        combined = pd.concat([tail, new_raw])
        new_state = {
            "tail": frame_to_json(combined[RAW_COLS].iloc[-self.tail:]),
            "seeds_last": self._seeds_at(ind_new, -1),
            "ind_tail": frame_to_json(ind_for_sig[IND_TAIL_COLS].iloc[-SIG_TAIL:]),
            "pos_tail_start": _pos_before_tail(sig_all, pos0),
        }
        return ind_join, sig_new, new_state

    # ------------------------------------------------------------------ #
    def latest(self, state: dict):
        """Rebuilds the last day from the state (no new data)."""
        tail = frame_from_json(state["tail"])
        ind_tail = frame_from_json(state["ind_tail"])
        pos0 = state.get("pos_tail_start", POS_DEFAULT)
        sig = generate_signals(ind_tail, self.params_by_quad, self.cfg,
                               pos0=pos0)
        last = ind_tail.iloc[[-1]].copy()
        last["vix"] = float(tail["vix"].iloc[-1])
        last["spx"] = float(tail["spx"].iloc[-1])
        return last, sig.iloc[[-1]]
