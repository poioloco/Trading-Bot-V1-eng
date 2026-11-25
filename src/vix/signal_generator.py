"""Signal generator - VIX_Output rule chain, with two voting methods.

There are two `method`s for combining the legs (VRP, RSI, TDEX), chosen
from config (`vix.voting_method` in params.yaml):

  method = "gate"  (EXACT REPLICA of the Excel, verified by the golden test)
  ------------------------------------------------------------------------
  TDEX enters as a GATE inside the ANDs, as in the workbook (rows
  3000/5350). It is the reference mode: do NOT change it, it is what keeps
  the replica honest against the Excel.

    Long-vol votes: G vote_vrp_longvol = vrp_smooth < vrp_long_thr
                    H vote_rsi_longvol = rsi_smooth < rsi_long_thr
    TDEX gate:      BJ tdex_on = tdex_smooth >  tdex_on_thr  (-0.5)
                    BK tdex_off= tdex_smooth >  tdex_off_thr (-7.5)
                    BL tdex_lv = tdex_smooth <= tdex_lv_thr  (4.0)
    Buy VIX  (I):   vrp_lv AND rsi_lv AND tdex_lv
    Buy SPX  (P):   spx_vrp AND spx_rsi AND tdex_on
    Sell SPX (T):   rsi<0 OR vrp<0 OR tdex_off
    (use_tdex=False -> legacy 2-vote logic, gates off)

  method = "majority"  (newer logic)
  ------------------------------------------------------------------------
  TDEX is an equal VOTE, no more veto. Every active leg casts one vote;
  entry fires if the votes in favour are a STRICT MAJORITY of the ACTIVE
  legs (floor(n/2)+1): with 3 legs 2 are needed, with 2 legs 2 are needed
  (unanimity), with 1 leg 1 is needed.

    - Legs are switched on/off from config (`vix.voting_legs`): a leg that
      is off neither votes nor triggers exits.
    - TDEX ABSTAINS when data is missing (tdex_smooth = NaN, e.g.
      pre-2013): it leaves the count (numerator and denominator), so the
      other legs still decide. With TDEX abstaining 2 legs remain and
      VRP+RSI unanimity is required: identical to the legacy behaviour.
    - On the Sell VIX side there are two semantics (`vix.longvol_exit`):
      "consensus" (default) = exit when the majority lapses, the dual of
      the 2-of-3 entry; "legs" = per-leg OR like the Excel (but with the
      majority entry it closes the next day any entry made with a
      dissenting leg, being the negation of unanimity). The TDEX term of
      the SPX exit fires on DEEP FEAR (tdex_smooth < tdex_sell_thr),
      consistent with rsi<0/vrp<0 which also fire on risk. NB: the Excel
      `gate` instead uses tdex_off_thr with direction ">" (BK = tdex>-7.5,
      almost always true): a faithful replica but an exit that is inert in
      practice; this is why the majority has its own exit threshold.
    - In majority mode the exit EVENTS are anchored to the POSITION: the
      Excel's transition filter assumes that entry implies sell false
      (true only in the gate, where entry=AND and exit=OR over the same
      legs); in 2-of-3 one can enter with sell already true and, without
      anchoring, the position would hang without an exit.

The TDEX thresholds stay in config: in `gate` the Excel values apply
(tdex_lv_thr=4.0, tdex_off_thr=-7.5); in `majority` the long-vol uses the
"fear" threshold (-3.0), the buy-SPX its own (-0.5) and the SPX exit its
own (tdex_sell_thr, default -4.0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .quad_params import QuadParams

GATE, MAJORITY = "gate", "majority"
CLASSIC, TREND = "classic", "trend"   # SignalConfig.market_mode


@dataclass
class SignalConfig:
    tdex_on_thr: float = -0.5    # VIX_Output!$P$3  (Buy SPX gate/vote)
    tdex_off_thr: float = -7.5   # VIX_Output!$T$3  (Sell SPX branch, gate/Excel ONLY)
    tdex_lv_thr: float = 4.0     # VIX_Output!$J$3  (long-vol gate/vote)
    tdex_sell_thr: float = -4.0  # SPX exit threshold in MAJORITY: exit on deep
                                 # fear, i.e. tdex_smooth < threshold. Used only
                                 # in majority mode (gate uses tdex_off_thr).
    use_tdex: bool = True        # backward compat: TDEX leg active (gate mode)
    gate_allow_missing_tdex: bool = False  # GATE: missing TDEX (NaN, e.g.
                                 # pre-2013) must NOT zero buy_spx_state. False
                                 # (legacy default) = missing TDEX blocks the
                                 # market entry -> bot blind until 2013 (dodges
                                 # 2008 for lack of data, not because of a
                                 # signal). True = faithful to the Excel: column
                                 # P is Buy also pre-2013 when VRP+RSI agree,
                                 # the gate is active from the start of history
                                 # (2008 = a real test, not skipped).
    method: str = GATE           # "gate" (Excel) | "majority"
    legs: dict | None = None     # {"vrp","rsi","tdex"} -> active in majority
    tdex_per_quad: dict | None = None  # {quad: {"sell_spx","longvol","buy_spx"}}
                                       # TDEX thresholds per quadrant; majority
                                       # ONLY. None => global thresholds.
    longvol_per_quad: dict | None = None  # {quad: {"rsi_long": x, "vrp_long": y}}
                                          # per-quadrant OVERRIDE of the long-vol
                                          # thresholds, majority ONLY (the gate
                                          # stays faithful to params_per_quad /
                                          # Excel). Optional keys: a quad may
                                          # override only rsi_long, only
                                          # vrp_long, or both. None/{} => use the
                                          # workbook thresholds.
    market_per_quad: dict | None = None   # {quad: {"rsi_exit": x, "vrp_exit": y,
                                          #         "rsi_buy": a, "vrp_buy": b}}
                                          # per-quadrant OVERRIDE of the MARKET
                                          # leg. rsi_exit/vrp_exit replace the
                                          # FIXED ZEROS of the Excel exit (sell
                                          # SPX) and apply in majority AND in
                                          # gate ("calibrated gate": entry
                                          # unchanged, exit anchored to the
                                          # position, TDEX branch = deep fear).
                                          # rsi_buy/vrp_buy (entry) apply in both
                                          # modes when present in the block.
                                          # Optional keys. None/{} => legacy
                                          # (zeros + workbook, gate = replica).
    market_reentry: str = "edge"  # RE-ENTRY of the MARKET leg after an exit:
                                 #   edge  = a NEW rising edge of the entry
                                 #           state is required (legacy: after
                                 #           an exit on a dip, if the AND/majority
                                 #           stays true you stay out until it
                                 #           goes false and then true again —
                                 #           even for months)
                                 #   state = if you are flat, the entry state is
                                 #           true and the exit is false, you
                                 #           RE-ENTER the same day (timely).
                                 # Applies in majority and in the CALIBRATED gate
                                 # (market_per_quad present). The pure gate is
                                 # ALWAYS the Excel replica, whatever the value.
    vol_exit: str = "consensus"  # exit of the volatility leg, majority ONLY:
                                 #   consensus = exit when the majority lapses
                                 #               (consistent with the 2-of-3 entry)
                                 #   legs      = per-leg OR like the Excel (an
                                 #               active leg that stops voting
                                 #               long-vol triggers the exit)
    trend_enabled: bool = False  # TREND FILTER (confirmed veto, 2026-07-19):
                                 # if SPX < SMA(trend_window) AND the
                                 # confirmation fires, the market position is
                                 # forced to 0. Overlay downstream of the
                                 # events: applies in gate AND majority. Off
                                 # (default) = exact replica.
    trend_window: int = 200      # window of the simple moving average on SPX
    trend_confirmation: str = "or"  # "or"    = veto if rsi<0 OR vrp<0 (tested:
                                 #           the best, DD -17% with top Calmar)
                                 # "and"   = veto if rsi<0 AND vrp<0
                                 # "total" = veto on the broken SMA alone
    tdex_stale_days: int | None = 10  # staleness guard of the TDEX ffill: if
                                      # the last FRED value is older than N
                                      # calendar days, TDEX goes back to NaN
                                      # and the leg ABSTAINS (like pre-2013)
                                      # instead of voting on a projected value.
                                      # None = guard off (plain ffill).
    market_mode: str = CLASSIC   # MARKET LEG, how it exits and re-enters
                                 # (2026-10-14):
                                 #   classic = entry/exit of the method (gate or
                                 #             majority) + trend filter on top;
                                 #             once the veto ends it re-enters
                                 #             automatically (legacy behaviour)
                                 #   trend   = exits ONLY via the trend filter
                                 #             (SPX below the SMA + RSI/VRP
                                 #             confirmation below threshold, per
                                 #             regime); after an exit it
                                 #             re-enters ONLY when the entry rule
                                 #             is true (RSI, VRP and TDEX above
                                 #             threshold, per regime) and there
                                 #             is no veto. Entry = AND as in the
                                 #             gate, with both methods.
    trend_entries_per_quad: dict | None = None  # {quad: {"rsi_buy", "vrp_buy",
                                 # "tdex_buy"}} entry thresholds of trend mode.
                                 # Missing keys: rsi_short/vrp_short of the
                                 # workbook and tdex_on_thr (tdex_gate_buy_spx).
    trend_exits_per_quad: dict | None = None     # {quad: {"rsi", "vrp"}} veto
                                 # CONFIRMATION thresholds in trend mode: veto
                                 # if SPX < SMA and (rsi < thr OR vrp < thr)
                                 # with confirmation "or" (AND with "and";
                                 # "total" = SMA only). Missing keys: 0 = the
                                 # current filter.

    def __post_init__(self) -> None:
        if self.legs is None:
            self.legs = {"vrp": True, "rsi": True, "tdex": bool(self.use_tdex)}


BUY, SELL, HOLD = "BUY", "SELL", "HOLD"


def generate_signals(
    ind: pd.DataFrame,
    params_by_quad: dict[int, QuadParams],
    cfg: SignalConfig | None = None,
    pos0: dict | None = None,
) -> pd.DataFrame:
    """`ind` is the output of compute_indicators (needs rsi_smooth,
    vrp_smooth, tdex_smooth, quad). Returns states, BUY/SELL events and the
    POSITION columns `pos_vix`/`pos_spx` (True = in position at end of day,
    including the entry day).

    `pos0` = position AND state BEFORE the first row of `ind`, e.g.
    {"vix": True, "spx": False, "vix_state": True, "spx_state": False}.
    It serves the incremental engine (StatefulVix), which regenerates the
    signals on a short tail of indicators: without the position at the
    boundary, in majority mode the exit event (anchored to the position)
    could not fire for positions opened before the tail; without the
    previous day's STATE (`*_state` = buy_*_state), the BUY edge on the
    first row would be spurious when the state was already true (possible
    on the SPX leg, where the state can stay true after an exit).
    None = flat (history from day one, like the Excel's empty cells)."""
    cfg = cfg or SignalConfig()
    p0 = {"vix": False, "spx": False, "vix_state": False, "spx_state": False}
    if pos0:
        p0.update({k: bool(v) for k, v in pos0.items()})
    # BASE position (before the trend filter) at the boundary: old states do
    # not have it -> it equals the final position (exact with the filter
    # off; a full recompute realigns everything anyway when the filter changes)
    p0.setdefault("spx_base", p0["spx"])

    vrp_long_thr = ind["quad"].map(lambda q: params_by_quad[int(q)].vrp_long_thr)
    rsi_long_thr = ind["quad"].map(lambda q: params_by_quad[int(q)].rsi_long_thr)
    vrp_short_thr = ind["quad"].map(lambda q: params_by_quad[int(q)].vrp_short_thr)
    rsi_short_thr = ind["quad"].map(lambda q: params_by_quad[int(q)].rsi_short_thr)

    # Per-quadrant long-vol override (live extension): applies ONLY in
    # majority, so the gate stays the faithful replica of the workbook.
    # It only touches the long-vol votes (entry/exit of the volatility
    # leg): the market leg (buy/sell SPX) uses the short thresholds and the
    # Excel's fixed zeros, so it does NOT change.
    if cfg.method == MAJORITY and cfg.longvol_per_quad:
        _ovr = {int(k): (v or {}) for k, v in cfg.longvol_per_quad.items()}
        _q = ind["quad"].astype(int)
        rsi_long_thr = _q.map(
            lambda q: float(_ovr.get(q, {}).get("rsi_long",
                                                params_by_quad[q].rsi_long_thr))
        )
        vrp_long_thr = _q.map(
            lambda q: float(_ovr.get(q, {}).get("vrp_long",
                                                params_by_quad[q].vrp_long_thr))
        )

    # Per-quadrant override of the MARKET leg (live extension): like
    # longvol_per_quad but on the Buy SPX ENTRY (rsi_buy/vrp_buy instead of
    # the workbook's short thresholds). Applies in majority AND in the
    # calibrated gate (2026-07-20, for entry timing): without rsi_buy/vrp_buy
    # keys in the block the defaults stay the workbook thresholds, so the
    # gate entry does not change. The EXIT thresholds (rsi_exit/vrp_exit)
    # are handled in _fill_majority and _fill_gate.
    if cfg.market_per_quad:
        _movr = {int(k): (v or {}) for k, v in cfg.market_per_quad.items()}
        _qm = ind["quad"].astype(int)
        rsi_short_thr = _qm.map(
            lambda q: float(_movr.get(q, {}).get("rsi_buy",
                                                 params_by_quad[q].rsi_short_thr))
        )
        vrp_short_thr = _qm.map(
            lambda q: float(_movr.get(q, {}).get("vrp_buy",
                                                 params_by_quad[q].vrp_short_thr))
        )

    out = pd.DataFrame(index=ind.index)
    # Elementary votes, common to both modes ---------------------------
    out["vote_vrp_longvol"] = ind["vrp_smooth"] < vrp_long_thr          # G
    out["vote_rsi_longvol"] = ind["rsi_smooth"] < rsi_long_thr          # H
    out["spx_vrp"] = ind["vrp_smooth"] > vrp_short_thr                  # N
    out["spx_rsi"] = ind["rsi_smooth"] > rsi_short_thr                  # O

    tdex = ind["tdex_smooth"]

    if cfg.method == MAJORITY:
        _fill_majority(out, ind, cfg, tdex)
    else:
        _fill_gate(out, ind, cfg, tdex, vrp_short_thr, rsi_short_thr)

    # ---- transition events -------------------------------------------
    bvs = out["buy_vix_state"].fillna(False).to_numpy(dtype=bool)
    svr = out["sell_vix_raw"].fillna(False).to_numpy(dtype=bool)
    bss = out["buy_spx_state"].fillna(False).to_numpy(dtype=bool)
    ssr = out["sell_spx_raw"].fillna(False).to_numpy(dtype=bool)

    if cfg.method == MAJORITY:
        # In majority mode the SELL event is ANCHORED TO THE POSITION. The
        # Excel's transition filter (below) is only safe in the gate: there
        # the entry is an AND over the same legs as the sell (OR), so buy
        # implies sell false and every position always gets its exit edge.
        # In 2-of-3 one can enter WITH a dissenting leg (sell already true):
        # without anchoring that exit would never fire and the position
        # would hang (defect found during calibration: time in vol that
        # ROSE when tightening the thresholds).
        out["buy_vix"], out["sell_vix"], pos_v = _events_pos(
            bvs, svr, p0["vix"], p0["vix_state"])
        if cfg.market_reentry == "state":
            out["buy_spx"], out["sell_spx"], pos_s = _events_pos_state(
                bss, ssr, p0["spx_base"])
        else:
            out["buy_spx"], out["sell_spx"], pos_s = _events_pos(
                bss, ssr, p0["spx_base"], p0["spx_state"])
        out["pos_vix"], out["pos_spx"] = pos_v, pos_s
    else:
        # Excel replica (golden-tested): events on the edges of the states only.
        out["buy_vix"] = bvs & ~_shift(bvs, 1)                          # J
        out["sell_vix"] = svr & (~_shift(svr, 1) | ~_shift(svr, 2))     # L
        out["pos_vix"] = _track_pos(
            out["buy_vix"].to_numpy(dtype=bool),
            out["sell_vix"].to_numpy(dtype=bool), p0["vix"])
        if cfg.market_per_quad:
            # CALIBRATED GATE: with per-quadrant exit thresholds the AND
            # entry no longer implies sell false (a threshold above the
            # short one can make the sell already true at entry): the Excel
            # transition filter would leave hanging positions. Same safety
            # net as the majority: SPX exit anchored to the POSITION.
            # With market_reentry: state the re-entry does not wait for a
            # new edge of the AND (timely after exits).
            if cfg.market_reentry == "state":
                out["buy_spx"], out["sell_spx"], pos_s = _events_pos_state(
                    bss, ssr, p0["spx_base"])
            else:
                out["buy_spx"], out["sell_spx"], pos_s = _events_pos(
                    bss, ssr, p0["spx_base"], p0["spx_state"])
            out["pos_spx"] = pos_s
        else:
            out["buy_spx"] = bss & ~_shift(bss, 1)                      # Q
            out["sell_spx"] = ssr & ~_shift(ssr, 1)                     # U
            # Position columns: ONLY tracking downstream of the Excel events
            # (no effect on the events, which stay the faithful replica).
            out["pos_spx"] = _track_pos(
                out["buy_spx"].to_numpy(dtype=bool),
                out["sell_spx"].to_numpy(dtype=bool), p0["spx_base"])

    # BASE position (before the filter): persisted in the incremental state,
    # so at the tail boundary the step engine restarts from the same base as
    # the full run even when the veto keeps the final position at zero.
    out["pos_spx_base"] = out["pos_spx"]

    # ---- TREND MODE: exit = trend filter, re-entry = entry rule --------
    # Fully replaces the market leg computed above (the volatility leg does
    # not change). No overlay: the veto already is the exit.
    if cfg.market_mode == TREND:
        _market_trend(out, ind, cfg, params_by_quad, p0)
        return out

    # ---- TREND FILTER (confirmed veto) on the market leg ---------------
    # Overlay DOWNSTREAM of the events (gate or majority): the final
    # position is pos_spx AND NOT veto; buy/sell SPX are regenerated from
    # the edges of the filtered position (so backtest, portfolio and live
    # stay consistent). The veto needs the spx_close/spx_sma columns in
    # `ind` (added by compute_indicators); if they are missing (old states)
    # the filter does not fire on those rows. Off (default): no effect,
    # exact golden-tested replica.
    if cfg.trend_enabled and "spx_close" in ind.columns and "spx_sma" in ind.columns:
        veto = _trend_veto(ind, cfg)
        base = out["pos_spx"].to_numpy(dtype=bool)
        fin = base & ~veto
        prev = np.concatenate(([bool(p0["spx"])], fin[:-1])) if len(fin) else fin
        out["trend_veto"] = veto
        out["buy_spx"] = fin & ~prev
        out["sell_spx"] = ~fin & prev
        out["pos_spx"] = fin
    return out


def _threshold_per_quad(quad: pd.Series, per_quad: dict | None, key: str,
                        default) -> np.ndarray:
    """Day-by-day threshold: value of the per-quad block for that day's
    regime, otherwise `default` (a number or a function of the quadrant)."""
    pq = {int(k): (v or {}) for k, v in (per_quad or {}).items()}
    q = quad.astype(int).to_numpy()
    out = np.empty(len(q), dtype=float)
    for qq in np.unique(q):
        d = default(int(qq)) if callable(default) else default
        out[q == qq] = float(pq.get(int(qq), {}).get(key, d))
    return out


def market_trend_arrays(ind: pd.DataFrame, cfg: SignalConfig,
                        params_by_quad: dict[int, QuadParams]):
    """Trend mode: (entry state E, veto V) as boolean arrays.

    E = vrp > vrp_buy  AND  rsi > rsi_buy  AND  tdex > tdex_buy   (per regime)
        Missing TDEX (NaN): passes only with gate_allow_missing_tdex, as in
        the gate; TDEX leg off (use_tdex False): condition ignored.
    V = SPX < SMA  AND  confirmation, confirmation "or" = rsi < r OR
        vrp < v (per regime), "and" = both, "total" = always true.
    NaN indicators count as False (no entry, no veto).
    This is the ONLY definition: the engine and the grid calibration use it."""
    quad = ind["quad"]
    rsi = pd.to_numeric(ind["rsi_smooth"], errors="coerce").to_numpy(dtype=float)
    vrp = pd.to_numeric(ind["vrp_smooth"], errors="coerce").to_numpy(dtype=float)
    tdex = pd.to_numeric(ind["tdex_smooth"], errors="coerce").to_numpy(dtype=float)
    rb = _threshold_per_quad(quad, cfg.trend_entries_per_quad, "rsi_buy",
                             lambda q: params_by_quad[q].rsi_short_thr)
    vb = _threshold_per_quad(quad, cfg.trend_entries_per_quad, "vrp_buy",
                             lambda q: params_by_quad[q].vrp_short_thr)
    tb = _threshold_per_quad(quad, cfg.trend_entries_per_quad, "tdex_buy",
                             float(cfg.tdex_on_thr))
    with np.errstate(invalid="ignore"):
        e = (vrp > vb) & (rsi > rb)
        if cfg.use_tdex:
            e = e & np.where(np.isnan(tdex), bool(cfg.gate_allow_missing_tdex),
                             tdex > tb)
        if "spx_close" in ind.columns and "spx_sma" in ind.columns:
            close = pd.to_numeric(ind["spx_close"], errors="coerce").to_numpy(dtype=float)
            sma = pd.to_numeric(ind["spx_sma"], errors="coerce").to_numpy(dtype=float)
            broken = close < sma
        else:
            broken = np.zeros(len(ind), dtype=bool)
        mode = str(cfg.trend_confirmation or "or").lower()
        if mode == "total":
            conf = np.ones(len(ind), dtype=bool)
        else:
            rc = _threshold_per_quad(quad, cfg.trend_exits_per_quad, "rsi", 0.0)
            vc = _threshold_per_quad(quad, cfg.trend_exits_per_quad, "vrp", 0.0)
            a, b = rsi < rc, vrp < vc
            conf = (a & b) if mode == "and" else (a | b)
    return e.astype(bool), (broken & conf).astype(bool)


def trend_position(e: np.ndarray, v: np.ndarray, pos0: bool = False) -> np.ndarray:
    """Position machine of trend mode: from outside you enter if E and no
    veto; from inside you exit if there is a veto. Markovian: pos0 suffices."""
    n = len(e)
    pos = np.zeros(n, dtype=bool)
    cur = bool(pos0)
    for i in range(n):
        if cur:
            if v[i]:
                cur = False
        elif e[i] and not v[i]:
            cur = True
        pos[i] = cur
    return pos


def _market_trend(out, ind, cfg, params_by_quad, p0) -> None:
    e, v = market_trend_arrays(ind, cfg, params_by_quad)
    pos = trend_position(e, v, bool(p0["spx"]))
    prev = np.concatenate(([bool(p0["spx"])], pos[:-1])) if len(pos) else pos
    out["buy_spx_state"] = e
    out["sell_spx_raw"] = v
    out["trend_veto"] = v
    out["buy_spx"] = pos & ~prev
    out["sell_spx"] = ~pos & prev
    out["pos_spx"] = pos
    out["pos_spx_base"] = pos


def _trend_veto(ind: pd.DataFrame, cfg: SignalConfig) -> np.ndarray:
    """Trend veto: broken SMA (spx < sma) plus the configured confirmation.
    NaNs (SMA warm-up, columns missing at the tail boundary) count as
    False: the veto never fires without data, never the other way round."""
    broken = (ind["spx_close"] < ind["spx_sma"]).fillna(False).to_numpy(dtype=bool)
    mode = str(cfg.trend_confirmation or "or").lower()
    if mode == "total":
        return broken
    rsi_neg = (ind["rsi_smooth"] < 0).fillna(False).to_numpy(dtype=bool)
    vrp_neg = (ind["vrp_smooth"] < 0).fillna(False).to_numpy(dtype=bool)
    conf = (rsi_neg & vrp_neg) if mode == "and" else (rsi_neg | vrp_neg)
    return broken & conf


# --------------------------------------------------------------------- #
# GATE mode - exact replica of the Excel (golden-tested). With the        #
# market_per_quad block it becomes the "CALIBRATED GATE": entry and       #
# volatility leg UNCHANGED (replica), but the market exit drops the fixed #
# zeros and the inert TDEX branch (tdex > -7.5, almost always true) for   #
# the per-quadrant thresholds — so calibrate-market/Monte Carlo also work #
# in gate. Without the block: exact replica, byte for byte.              #
# --------------------------------------------------------------------- #
def _fill_gate(out, ind, cfg, tdex, vrp_short_thr, rsi_short_thr) -> None:
    calibrated = bool(cfg.market_per_quad)
    if calibrated:
        _mqe = {int(k): (v or {}) for k, v in cfg.market_per_quad.items()}
        _qe = ind["quad"].astype(int)
        rsi_exit_thr = _qe.map(lambda q: float(_mqe.get(q, {}).get("rsi_exit", 0.0)))
        vrp_exit_thr = _qe.map(lambda q: float(_mqe.get(q, {}).get("vrp_exit", 0.0)))
        # TDEX branch of the exit: DEEP fear (tdex < threshold), same
        # semantics as the majority; per-quad if configured. NaN -> False
        # (TDEX abstains, like pre-2013).
        if cfg.tdex_per_quad:
            _pq = {int(k): v for k, v in cfg.tdex_per_quad.items()}
            tdex_sell_thr = _qe.map(lambda q: float(_pq[q]["sell_spx"]))
        else:
            tdex_sell_thr = cfg.tdex_sell_thr
    else:
        rsi_exit_thr, vrp_exit_thr = 0.0, 0.0

    if cfg.use_tdex:
        out["tdex_on"] = tdex > cfg.tdex_on_thr                          # BJ
        if cfg.gate_allow_missing_tdex:
            # Missing TDEX (NaN) = gate open, like the Excel pre-2013: the
            # missing value must not block the entry (VRP+RSI decide).
            out["tdex_on"] = out["tdex_on"].mask(tdex.isna(), True)
        out["tdex_off"] = ((tdex < tdex_sell_thr) if calibrated          # BK
                           else (tdex > cfg.tdex_off_thr))
        out["tdex_lv"] = tdex <= cfg.tdex_lv_thr                         # BL
        out["buy_vix_state"] = (                                         # I
            out["vote_vrp_longvol"] & out["vote_rsi_longvol"] & out["tdex_lv"]
        )
        out["buy_spx_state"] = (                                         # P
            (ind["vrp_smooth"] > vrp_short_thr)
            & (ind["rsi_smooth"] > rsi_short_thr)
            & out["tdex_on"]
        )
        out["sell_spx_raw"] = (                                          # T
            (ind["rsi_smooth"] < rsi_exit_thr)
            | (ind["vrp_smooth"] < vrp_exit_thr)
            | out["tdex_off"]
        )
    else:  # legacy 2-vote logic (gates off)
        out["tdex_on"] = out["tdex_off"] = out["tdex_lv"] = pd.NA
        out["buy_vix_state"] = out["vote_vrp_longvol"] & out["vote_rsi_longvol"]
        out["buy_spx_state"] = (ind["vrp_smooth"] > vrp_short_thr) & (
            ind["rsi_smooth"] > rsi_short_thr
        )
        out["sell_spx_raw"] = (ind["rsi_smooth"] < rsi_exit_thr) | (
            ind["vrp_smooth"] < vrp_exit_thr
        )

    out["sell_vix_raw"] = ~out["vote_vrp_longvol"] | ~out["vote_rsi_longvol"]  # K


# --------------------------------------------------------------------- #
# MAJORITY mode - TDEX votes, no veto, abstention, toggles               #
# --------------------------------------------------------------------- #
def _fill_majority(out, ind, cfg, tdex) -> None:
    L = cfg.legs or {}
    on_vrp = bool(L.get("vrp", True))
    on_rsi = bool(L.get("rsi", True))
    on_tdex = bool(L.get("tdex", True))

    idx = out.index
    true_s = pd.Series(True, index=idx)
    false_s = pd.Series(False, index=idx)
    # mask "leg active AND casting a vote today": TDEX abstains on NaN
    # (excluded from the denominator), VRP/RSI are always present if on
    m_vrp = true_s if on_vrp else false_s
    m_rsi = true_s if on_rsi else false_s
    m_tdex = (tdex.notna()) if on_tdex else false_s

    # TDEX thresholds: per-quad if configured (live extension beyond the
    # Excel, where they are global cells), otherwise the global ones. The
    # quad changes day by day (from the 40-ETF model), so the threshold
    # follows the regime.
    if cfg.tdex_per_quad:
        _pq = {int(k): v for k, v in cfg.tdex_per_quad.items()}
        _quad = ind["quad"].astype(int)
        lv_thr = _quad.map(lambda q: float(_pq[q]["longvol"]))
        on_thr = _quad.map(lambda q: float(_pq[q]["buy_spx"]))
        sell_thr = _quad.map(lambda q: float(_pq[q]["sell_spx"]))
    else:
        lv_thr, on_thr, sell_thr = cfg.tdex_lv_thr, cfg.tdex_on_thr, cfg.tdex_sell_thr

    # TDEX leg vote (exposed for diagnostics, NA when abstaining)
    tdex_lv_vote = (tdex <= lv_thr)                 # fear -> buy vol
    tdex_on_vote = (tdex > on_thr)                  # no fear -> market ok
    out["tdex_lv"] = tdex_lv_vote.where(m_tdex, other=pd.NA)
    out["tdex_on"] = tdex_on_vote.where(m_tdex, other=pd.NA)
    # SPX exit: DEEP fear triggers the exit -> tdex_smooth < threshold
    # (correct direction, consistent with rsi<0 / vrp<0, which also fire on risk)
    out["tdex_off"] = (tdex < sell_thr).where(m_tdex, other=pd.NA)

    # ---- BUY VIX (long-vol): majority of the active legs ----------------
    out["buy_vix_state"] = _majority(
        votes={"vrp": out["vote_vrp_longvol"], "rsi": out["vote_rsi_longvol"],
               "tdex": tdex_lv_vote},
        masks={"vrp": m_vrp, "rsi": m_rsi, "tdex": m_tdex},
        idx=idx,
    )
    # ---- BUY SPX (market): majority of the active legs ------------------
    out["buy_spx_state"] = _majority(
        votes={"vrp": out["spx_vrp"], "rsi": out["spx_rsi"], "tdex": tdex_on_vote},
        masks={"vrp": m_vrp, "rsi": m_rsi, "tdex": m_tdex},
        idx=idx,
    )

    # ---- SELL: exits ----------------------------------------------------
    # Sell VIX, two semantics (cfg.vol_exit):
    #   consensus (default) = exit when the MAJORITY lapses: it is the dual
    #     of the 2-of-3 entry (same philosophy as the portfolio: "exit when
    #     the consensus lapses") and makes threshold calibration monotonic
    #     and readable.
    #   legs = per-leg OR like the Excel: an active leg that stops voting
    #     long-vol triggers the exit. NB: with the majority entry, an entry
    #     with a dissenting leg is closed again the next day (the OR is the
    #     negation of unanimity, not of the majority).
    if cfg.vol_exit == "legs":
        svix = false_s.copy()
        if on_vrp:
            svix = svix | ~out["vote_vrp_longvol"]
        if on_rsi:
            svix = svix | ~out["vote_rsi_longvol"]
        out["sell_vix_raw"] = svix
    else:  # consensus
        out["sell_vix_raw"] = ~out["buy_vix_state"]

    # Sell SPX: rsi<thr OR vrp<thr OR tdex_off (legs that are off excluded).
    # The exit thresholds are the Excel's FIXED ZEROS, or the per-quadrant
    # overrides of market_per_quad (rsi_exit/vrp_exit): with the exit
    # anchored to the position the zeros are hyperactive (churn), the regime
    # quantile makes it calibratable — same philosophy as the long-vol.
    if cfg.market_per_quad:
        _mqe = {int(k): (v or {}) for k, v in cfg.market_per_quad.items()}
        _qe = ind["quad"].astype(int)
        rsi_exit_thr = _qe.map(lambda q: float(_mqe.get(q, {}).get("rsi_exit", 0.0)))
        vrp_exit_thr = _qe.map(lambda q: float(_mqe.get(q, {}).get("vrp_exit", 0.0)))
    else:
        rsi_exit_thr, vrp_exit_thr = 0.0, 0.0
    sspx = false_s.copy()
    if on_vrp:
        sspx = sspx | (ind["vrp_smooth"] < vrp_exit_thr)
    if on_rsi:
        sspx = sspx | (ind["rsi_smooth"] < rsi_exit_thr)
    if on_tdex:
        sspx = sspx | ((tdex < sell_thr) & m_tdex)
    out["sell_spx_raw"] = sspx


def _majority(votes: dict, masks: dict, idx) -> pd.Series:
    """Strict-majority vote over the active legs voting today only.
    threshold = floor(n_active/2)+1. n_active=0 -> no signal (HOLD)."""
    favor = pd.Series(0, index=idx)
    n_active = pd.Series(0, index=idx)
    for leg, vote in votes.items():
        m = masks[leg]
        favor = favor + (vote.fillna(False) & m).astype(int)
        n_active = n_active + m.astype(int)
    threshold = (n_active // 2) + 1
    return (n_active > 0) & (favor >= threshold)


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """Boolean array shifted forward by k days (the Excel's 'day before');
    missing history counts as False ('Do Nothing'), like empty cells."""
    out = np.zeros_like(a, dtype=bool)
    if k < len(a):
        out[k:] = a[: len(a) - k]
    return out


def _events_pos(buy_state: np.ndarray, sell_raw: np.ndarray,
                pos0: bool = False, state0: bool = False):
    """Events anchored to the position (majority ONLY): BUY on the rising
    edge of the entry state; SELL when IN position and the exit condition
    is true (regardless of the edges of the raw sell). Guarantees that
    every open position always receives its exit.

    `pos0` = position before the first row; `state0` = buy state of the
    day before the first row (for the incremental engine): without it, a
    state already true at the boundary would produce a false BUY edge on
    row 0 and corrupt the propagation of the position.
    Returns (buy_ev, sell_ev, pos): `pos` is the end-of-day position,
    including the entry day."""
    n = len(buy_state)
    buy_ev = buy_state & ~_shift(buy_state, 1)
    if n:
        buy_ev[0] = bool(buy_state[0]) and not bool(state0)
    sell_ev = np.zeros(n, dtype=bool)
    pos_arr = np.zeros(n, dtype=bool)
    pos = bool(pos0)
    for i in range(n):
        if buy_ev[i]:
            pos = True
        elif pos and sell_raw[i]:
            sell_ev[i] = True
            pos = False
        pos_arr[i] = pos
    return buy_ev, sell_ev, pos_arr


def _events_pos_state(buy_state: np.ndarray, sell_raw: np.ndarray,
                      pos0: bool = False):
    """Events with STATE RE-ENTRY (market_reentry: state): if you are flat,
    the entry state is true and the exit condition is false, you ENTER —
    without waiting for a new edge of the state. The exit stays anchored
    to the position. Compared with _events_pos it removes the months out
    of the market after an exit when the AND/majority stays true: the
    re-entry comes as soon as the exit clears (timely). If entry and exit
    are both true while flat, the exit blocks the entry (no churn).
    Markovian in the position: at the incremental tail boundary `pos0` is
    enough, no previous-day state is needed."""
    n = len(buy_state)
    buy_ev = np.zeros(n, dtype=bool)
    sell_ev = np.zeros(n, dtype=bool)
    pos_arr = np.zeros(n, dtype=bool)
    pos = bool(pos0)
    for i in range(n):
        if not pos and buy_state[i] and not sell_raw[i]:
            buy_ev[i] = True
            pos = True
        elif pos and sell_raw[i]:
            sell_ev[i] = True
            pos = False
        pos_arr[i] = pos
    return buy_ev, sell_ev, pos_arr


def _track_pos(buy_ev: np.ndarray, sell_ev: np.ndarray,
               pos0: bool = False) -> np.ndarray:
    """Position machine downstream of the EVENTS (for the gate): BUY opens,
    SELL closes, otherwise hold. Same convention as _positions in the
    backtest (position True already on the entry day). Diagnostics only:
    it does not touch the Excel events."""
    n = len(buy_ev)
    pos_arr = np.zeros(n, dtype=bool)
    pos = bool(pos0)
    for i in range(n):
        if buy_ev[i]:
            pos = True
        elif sell_ev[i]:
            pos = False
        pos_arr[i] = pos
    return pos_arr


def to_asset_signals(
    signals_row: pd.Series, long_asset: str = "TQQQ", vol_asset: str = "UVIX"
) -> dict[str, str]:
    """Maps the day's events onto BUY/SELL/HOLD per asset."""
    out = {long_asset: HOLD, vol_asset: HOLD}
    if bool(signals_row["buy_spx"]):
        out[long_asset] = BUY
    elif bool(signals_row["sell_spx"]):
        out[long_asset] = SELL
    if bool(signals_row["buy_vix"]):
        out[vol_asset] = BUY
    elif bool(signals_row["sell_vix"]):
        out[vol_asset] = SELL
    return out
