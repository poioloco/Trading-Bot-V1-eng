"""Daily pipeline: regime -> quad of the day -> VIX -> signals.

Orchestrates the two state engines in lockstep:
  - first run (no state): full warm-up from 2005, saves the state;
  - following runs: only the new days, starting from the saved state.
Writes the signal history (idempotent) and logs every run (heartbeat).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from ..data.database import Database
from ..data.store import PointInTimeStore
from ..regime.quad_aggregator import QUAD_NAMES
from ..regime.winner import RegimeParams, RegimeSeeds, compute_winner
from ..vix.indicators import ChainSeeds, compute_indicators
from ..vix.quad_params import QuadParams
from ..vix.signal_generator import SignalConfig, generate_signals, to_asset_signals
from . import frame_from_json, load_state, save_state
from .regime_engine import StatefulRegime
from .vix_engine import StatefulVix


@dataclass
class DailyReport:
    as_of: pd.Timestamp
    quad: int
    quad_name: str
    asset_signals: dict[str, str]
    events: dict[str, bool]
    metrics: dict[str, float | None] = field(default_factory=dict)
    mode: str = ""               # "warm-up" | "incremental" | "up to date"
    new_days: int = 0
    warning: str = ""            # e.g. "VIX of 20/11 not published yet…"


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]


def date_with_indices(db: Database, to_date: date,
                      spx_symbol: str = "^SPX", vix_symbol: str = "^VIX"
                      ) -> tuple[date, str]:
    """Last day <= `to_date` on which the archive has BOTH the VIX AND SPX.

    VIX and SPX come from FRED, which publishes a day's close only on the
    following business day (around 09:40 New York time). Before then the
    requested day is incomplete: instead of blocking, the bot computes on
    the last complete day and returns a readable warning. If an index is
    missing entirely, it returns `to_date` without a warning and lets the
    original error explain what is missing."""
    requested = pd.Timestamp(to_date).date()
    latest: dict[str, date] = {}
    for sym in (vix_symbol, spx_symbol):
        row = db.con.execute(
            "SELECT MAX(date) FROM prices_raw WHERE symbol = ? AND date <= ?",
            [sym, requested]).fetchone()
        if not row or row[0] is None:
            return requested, ""
        latest[sym] = pd.Timestamp(row[0]).date()
    effective = min(latest.values())
    if effective >= requested:
        return requested, ""

    from ..data.calendar import sessions_between
    missing = [s.lstrip("^") for s, d in latest.items() if d < requested]
    who = " and ".join(missing)
    missing_days = sessions_between(effective + timedelta(days=1), requested)
    # FRED publishes on the business day after the last missing day
    after = sessions_between(requested + timedelta(days=1),
                             requested + timedelta(days=10))
    when = (f"{_WEEKDAYS[after[0].weekday()]} {after[0]:%d/%m}" if after
            else "the next business day")
    plural = len(missing) > 1
    if len(missing_days) <= 1:
        what = (f"{who} for {requested:%d/%m} have not been published yet"
                if plural else
                f"The {who} for {requested:%d/%m} has not been published yet")
    else:
        what = (f"{who} data missing from {missing_days[0]:%d/%m} to "
                f"{requested:%d/%m}")
    warning = (f"{what}: signal computed on the last complete day "
               f"({_WEEKDAYS[effective.weekday()]} {effective:%d/%m}). "
               f"FRED publishes {'them' if plural else 'it'} {when} around 09:40 "
               f"New York time (15:40 CET): "
               f"after that time run the data update and the signals again.")
    if len(missing_days) > 2:
        warning += (" More days than usual are missing: check that "
                    "the data update succeeds.")
    return effective, warning


class DailyPipeline:
    def __init__(self, db: Database, store: PointInTimeStore,
                 quads: dict[str, list[str]], regime_params: RegimeParams,
                 ma_windows: list[int], vote_threshold: float,
                 params_by_quad: dict[int, QuadParams], cfg: SignalConfig,
                 long_asset: str, vol_asset: str,
                 spx_symbol: str = "^SPX", vix_symbol: str = "^VIX") -> None:
        self.db = db
        self.store = store
        self.regime_params = regime_params
        self.params_by_quad = params_by_quad
        self.long_asset = long_asset
        self.vol_asset = vol_asset
        self.spx_symbol = spx_symbol
        self.vix_symbol = vix_symbol
        self.regime = StatefulRegime(store, quads, regime_params, ma_windows, vote_threshold)
        self.vix = StatefulVix(store, params_by_quad, cfg, spx_symbol, vix_symbol)
        # configuration signature: if the parameters change (e.g. after an
        # applied calibration or a tweak from the control panel), the saved
        # incremental state is no longer consistent and must be rebuilt.
        import dataclasses
        import json as _json
        self._config_signature = _json.dumps({
            "cfg": dataclasses.asdict(cfg),
            "quad": {int(q): dataclasses.asdict(v)
                     for q, v in sorted(params_by_quad.items())},
            "regime": dataclasses.asdict(regime_params),
            "ma": list(ma_windows), "threshold": vote_threshold,
            "asset": [long_asset, vol_asset, spx_symbol, vix_symbol],
            "quads": {k: list(v) for k, v in sorted(quads.items())},
        }, sort_keys=True, default=str)

    # ------------------------------------------------------------------ #
    def run(self, to_date: date, force_rebuild: bool = False) -> DailyReport:
        # if VIX/SPX for the requested day are not out yet, compute on the
        # last complete day (same day for regime and VIX, so the two
        # engines stay aligned) and warn
        requested = to_date
        to_date, warning = date_with_indices(self.db, to_date,
                                             self.spx_symbol, self.vix_symbol)
        run_id = self.db.start_run("signals", as_of=to_date)
        try:
            rep = self._run(to_date, force_rebuild)
            rep.warning = warning
            note = (f" (indices for {requested} not published yet: "
                    f"computed as of {to_date})" if warning else "")
            self.db.finish_run(run_id, "ok",
                               f"{rep.mode}: {rep.new_days} new days, "
                               f"quad {rep.quad}{note}")
            return rep
        except Exception as exc:
            self.db.finish_run(run_id, "error", str(exc))
            raise

    def _run(self, to_date: date, force_rebuild: bool) -> DailyReport:
        rstate = None if force_rebuild else load_state(self.db, "regime")
        vstate = None if force_rebuild else load_state(self.db, "vix")

        # Misalignment guard: if the price history changed RETROACTIVELY
        # after the state was saved (backfill of past dates, corrections),
        # the incremental state no longer mirrors the database and must be
        # thrown away: everything is rebuilt (same numbers as the backtest).
        # The fingerprint is (no. of bars, sum of closes) up to the state's
        # anchor. Old states without a fingerprint are also rebuilt once.
        rebuilt = False
        if rstate is not None and vstate is not None:
            expected = rstate[1].pop("_data_fingerprint", None)
            vstate[1].pop("_data_fingerprint", None)
            expected_cfg = rstate[1].pop("_config_fingerprint", None)
            vstate[1].pop("_config_fingerprint", None)
            anchor = min(rstate[0], vstate[0])
            if (expected is None or list(expected) != self._data_fingerprint(anchor)
                    or expected_cfg != self._config_signature):
                rstate = vstate = None
                rebuilt = True

        if rstate is None or vstate is None:
            # full WARM-UP (lockstep: redo both)
            winner_full, r_new_state = self.regime.warm_up(to_date)
            ind, sig, v_new_state = self.vix.warm_up(winner_full, to_date)
            self._save(to_date, r_new_state, v_new_state)
            mode = "rebuild" if rebuilt else "warm-up"
            new_days = len(winner_full)
        else:
            r_anchor, rst = rstate
            v_anchor, vst = vstate
            winner_new, r_new_state = self.regime.step(r_anchor, rst, to_date)
            if winner_new.empty:
                # already up to date: only rebuild the last day from the state
                return self._latest_from_state(rst, vst)
            ind, sig, v_new_state = self.vix.step(v_anchor, vst, winner_new, to_date)
            self._save(to_date, r_new_state, v_new_state)
            mode, new_days = "incremental", len(winner_new)

        return self._report(ind, sig, mode, new_days)

    # ------------------------------------------------------------------ #
    def _data_fingerprint(self, up_to) -> list:
        """Compact fingerprint of the history up to a date: it changes if
        new bars arrive on past dates or if an existing bar is corrected."""
        row = self.db.con.execute(
            "SELECT COUNT(*), COALESCE(ROUND(SUM(close), 2), 0) "
            "FROM prices_raw WHERE date <= ?", [up_to]).fetchone()
        return [int(row[0]), float(row[1])]

    def _save(self, to_date: date, r_state: dict, v_state: dict) -> None:
        fingerprint = self._data_fingerprint(to_date)
        for st in (r_state, v_state):
            st["_data_fingerprint"] = fingerprint
            st["_config_fingerprint"] = self._config_signature
        save_state(self.db, "regime", to_date, r_state)
        save_state(self.db, "vix", to_date, v_state)

    # ------------------------------------------------------------------ #
    def _latest_from_state(self, rst: dict, vst: dict) -> DailyReport:
        li, sig = self.vix.latest(vst)
        return self._report(li, sig, "up to date", 0)

    def _report(self, ind: pd.DataFrame, sig: pd.DataFrame,
                mode: str, new_days: int) -> DailyReport:
        li, ls = ind.iloc[-1], sig.iloc[-1]
        quad = int(li["quad"])
        assets = to_asset_signals(ls, self.long_asset, self.vol_asset)
        events = {k: bool(ls[k]) for k in ("buy_vix", "sell_vix", "buy_spx", "sell_spx")}
        as_of = ind.index[-1]
        # signal history (idempotent)
        for sym, act in assets.items():
            self.db.write_signal(as_of.date(), sym, act, quad)
        metrics = {
            "spx": float(li["spx"]), "vix": float(li["vix"]),
            "vrp_smooth": float(li["vrp_smooth"]), "rsi_smooth": float(li["rsi_smooth"]),
            "tdex_smooth": float(li["tdex_smooth"]) if pd.notna(li.get("tdex_smooth")) else None,
        }
        return DailyReport(as_of=as_of, quad=quad, quad_name=QUAD_NAMES[quad - 1],
                           asset_signals=assets, events=events, metrics=metrics,
                           mode=mode, new_days=new_days)
