"""GRID calibration of the market leg in trend mode (2026-10-14).

What it does
------------
In trend mode (vix.market_mode: trend) the bot exits ONLY via the trend
filter and re-enters ONLY when the entry rule is true. The two are
calibrated SEPARATELY:

  * ENTRIES: for each regime the rsi_buy, vrp_buy and tdex_buy thresholds;
    each one tries EVERY value from -10 to +10 in steps of 0.1 (201 values),
    so 201^3 = 8,120,601 combinations per regime. Exits stay fixed.
  * EXITS: for each regime the veto confirmation thresholds (rsi, vrp),
    every value from -10 to +10 in steps of 0.1: 201^2 = 40,401
    combinations per regime. Entries stay fixed.

All four regimes TOGETHER would be 201^12 combinations (entries):
impossible. So one regime at a time is done (all the combinations of that
regime, the other three fixed) and the round is repeated until no regime
changes any more. The outcome changes only if the training result REALLY
improves.

Tie-break between equivalent thresholds (2026-10-20)
----------------------------------------------------
Many combinations give the same result: they form a "plateau" on the grid.
The tie-break (`tiebreak`) decides which point to choose:

  * robust (default): among outcomes tied in training (the current one
    included) the DEEPEST plateau wins, and inside the plateau the most
    INTERNAL point (ties: the one closest to the centroid, then to the
    current threshold). If the current outcome wins the threshold is only
    RECENTRED: all the historical numbers stay identical, the margin grows.
    If another tied outcome wins, training stays identical and only the
    unseen years can change (for better or worse: they are not looked at).
  * nearest: the previous rule (keep; if it changes, the point closest to
    the starting threshold).
  * unseen: chooses, among the outcomes tied in training, the one with the
    best Calmar in the unseen years. Declared and off by default: if used,
    the unseen years are NO longer a clean test.

The unseen years never enter the choice, except with `unseen`.
"Margin" = by how many 0.1 steps one can move from the chosen point while
staying in the same outcome (the edge of the grid, -10 and +10, counts as
an edge).

Why it is fast (and exact)
--------------------------
Very many combinations produce EXACTLY the same positions: only the first
day on which the entry rule becomes true in each period without a veto
matters. Each combination is "counted" (the day on which it would re-enter
is computed) and combinations with the same outcome are simulated only
once. Every exploration verifies itself: 100 random combinations are also
simulated directly and must give the same numbers; the final result and a
few random combinations are moreover compared with the REAL ENGINE
(generate_signals). If even a single position differs, the calibration
stops with an error.

Windows
-------
Evaluation starts only from `start` (default 2013-07-26, first day with
TDEX): the indicators warm up on the earlier history, but the numbers start
from there, from a flat position. Training runs on `train_start` -> today and
2013 -> today and 2013 -> train_start are also reported (years NEVER seen
by the training: the real check). Costs: `cost_bps` per side.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..vix.signal_generator import (TREND, SignalConfig, generate_signals,
                                    market_trend_arrays, trend_position)

GRID = np.arange(-100, 101) / 10.0         # -10.0 .. +10.0, step 0.1
NG = len(GRID)                             # 201
QUADS = (1, 2, 3, 4)
QUAD_LABELS = {1: "Disinflation", 2: "Reflation", 3: "Stagflation",
               4: "Deflation"}
OBJECTIVES = {"calmar": "calmar", "return": "cagr_pct", "sharpe": "sharpe"}
BATCH = 400            # candidates simulated together (memory ~ tens of MB)
TOL = 1e-12
TIEBREAKS = ("robust", "nearest", "unseen")
MAX_TIES = 300         # tied outcomes described in the report (the largest)


class VerificationFailed(RuntimeError):
    """The fast engine does not match the real one: no results."""


def _g(x: float) -> float:
    """Value rounded to one decimal (what ends up in the YAML)."""
    return float(round(float(x), 1))


def _idx(thresholds: dict, keys) -> tuple:
    """Grid indices of the thresholds (rounded to the step, within -10..+10)."""
    return tuple(int(np.clip(round((float(thresholds[k]) + 10.0) * 10), 0, NG - 1))
                 for k in keys)


def _on_grid(thresholds: dict, keys) -> bool:
    """True if all thresholds are exact grid values."""
    for k in keys:
        v = float(thresholds[k])
        if not (-10.0 - 1e-9 <= v <= 10.0 + 1e-9) or abs(v * 10 - round(v * 10)) > 1e-9:
            return False
    return True


# ---------------------------------------------------------------------- #
# Plateaus: depth and innermost point (numpy only)                        #
# ---------------------------------------------------------------------- #
def _depth_map(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth of every point of a region of the grid.

    `pts` = grid indices of the region's points, shape (N, d). The region
    is "eroded" one step at a time from the edges: at each step only the
    points with both neighbours (-1 and +1) on EVERY axis inside the region
    survive. Outside the bounding box and outside the grid counts as an
    edge. Returns (lo, depth): `depth` covers the box starting at `lo`;
    -1 = outside the region, k = the point survives k steps (0 = it lies on
    the edge). A point with depth k can be moved k steps in total, in any
    combination of directions, staying inside the region.
    """
    pts = np.asarray(pts, dtype=np.int64).reshape(len(pts), -1)
    if len(pts) == 0:
        raise ValueError("empty region")
    lo = pts.min(axis=0)
    hi = pts.max(axis=0) + 1
    shape = tuple(int(x) for x in (hi - lo + 2))       # +1 of border per side
    cur = np.zeros(shape, dtype=bool)
    cur[tuple((pts - lo + 1).T)] = True
    depth = np.where(cur, 0, -1).astype(np.int32)
    core = (slice(1, -1),) * len(shape)
    while True:
        remain = cur[core].copy()
        for ax in range(len(shape)):
            minus, plus = list(core), list(core)
            minus[ax], plus[ax] = slice(0, -2), slice(2, None)
            remain &= cur[tuple(minus)] & cur[tuple(plus)]
        if not remain.any():
            break
        cur = np.zeros_like(cur)
        cur[core] = remain
        depth[cur] += 1
    return lo, depth[core]


def _innermost_point(pts: np.ndarray, current: tuple | None = None) -> tuple[tuple, int]:
    """The innermost point of a region and its depth (margin).
    Ties on depth: the closest to the region's centroid, then the closest
    to `current` (indices), then the first in order."""
    pts = np.asarray(pts, dtype=np.int64).reshape(len(pts), -1)
    lo, depth = _depth_map(pts)
    margin = int(depth.max())
    cand = np.argwhere(depth == margin) + lo
    centroid = pts.mean(axis=0)
    d_centroid = np.round(((cand - centroid) ** 2).sum(axis=1), 6)
    if current is not None:
        d_cur = ((cand - np.asarray(current)) ** 2).sum(axis=1)
    else:
        d_cur = np.zeros(len(cand))
    j = int(np.lexsort((d_cur, d_centroid))[0])
    return tuple(int(x) for x in cand[j]), margin


def _depth_of(pts: np.ndarray, point: tuple) -> int:
    """Depth (margin) of a specific point inside its region."""
    lo, depth = _depth_map(pts)
    rel = tuple(int(p - l) for p, l in zip(point, lo))
    if any(r < 0 or r >= s for r, s in zip(rel, depth.shape)):
        return -1
    return int(depth[rel])


def _axis_margins(inside: np.ndarray, point: tuple) -> list[tuple[int, int]]:
    """For each axis: how many steps one can go down and up (moving ONLY
    that threshold) staying inside the region `inside` (bool, grid shape)."""
    out = []
    for ax in range(inside.ndim):
        sel = list(point)
        sel[ax] = slice(None)
        line = inside[tuple(sel)]
        i = point[ax]
        down = 0
        while i - down - 1 >= 0 and line[i - down - 1]:
            down += 1
        up = 0
        while i + up + 1 < len(line) and line[i + up + 1]:
            up += 1
        out.append((down, up))
    return out


def _groups(inv: np.ndarray, outcomes: np.ndarray, shape: tuple) -> dict:
    """{outcome: coordinates (N, d) of its combinations} for the given outcomes."""
    members = np.flatnonzero(np.isin(inv, outcomes))
    if len(members) == 0:
        return {}
    order = np.argsort(inv[members], kind="stable")
    members = members[order]
    e = inv[members]
    cuts = np.r_[0, np.flatnonzero(np.diff(e)) + 1, len(e)]
    coords = np.stack(np.unravel_index(members, shape), axis=1)
    return {int(e[a]): coords[a:b] for a, b in zip(cuts[:-1], cuts[1:])}


class GridCalibrator:
    def __init__(self, ind_full: pd.DataFrame, params_by_quad: dict,
                 cfg: SignalConfig, start: str = "2013-07-26",
                 train_start: str = "2018-01-01", cost_bps: float = 5.0,
                 objective: str = "calmar", seed: int = 7,
                 tiebreak: str = "robust") -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {list(OBJECTIVES)}")
        tiebreak = str(tiebreak or "robust").strip().lower()
        if tiebreak not in TIEBREAKS:
            raise ValueError(f"tiebreak must be one of {list(TIEBREAKS)}")
        self.tiebreak = tiebreak
        self.pq = params_by_quad
        self.cfg_today = cfg                        # as configured NOW
        self.cfg = replace(cfg, market_mode=TREND, trend_enabled=True)
        ind = ind_full[ind_full.index >= pd.Timestamp(start)]
        if len(ind) < 250:
            raise ValueError(f"too few days since {start} ({len(ind)})")
        self.ind = ind
        self.n = n = len(ind)
        self.ord = np.arange(n)
        self.qi = ind["quad"].astype(int).to_numpy()
        num = lambda c: pd.to_numeric(ind[c], errors="coerce").to_numpy(dtype=float)
        self.rsi, self.vrp, self.tdex = num("rsi_smooth"), num("vrp_smooth"), num("tdex_smooth")
        if "spx_close" in ind.columns and "spx_sma" in ind.columns:
            with np.errstate(invalid="ignore"):
                self.broken = num("spx_close") < num("spx_sma")
        else:
            raise ValueError("the trend filter columns (spx_close/spx_sma) are missing")
        # explicit copy: with pandas 3 the to_numpy arrays are read-only
        self.ret = np.array(ind["spx"].pct_change().fillna(0.0).to_numpy(dtype=float),
                            dtype=float, copy=True)
        self.ret[0] = 0.0
        self.cost = float(cost_bps) / 1e4
        self.cost_bps = float(cost_bps)
        i_tr = int(np.searchsorted(ind.index.values,
                                   np.datetime64(pd.Timestamp(train_start))))
        if i_tr < 60 or i_tr > n - 60:
            raise ValueError("the training start date must leave at least "
                             "60 sessions before and after")
        self.windows = {"train": (i_tr, n), "unseen": (0, i_tr),
                        "full": (0, n)}
        d = ind.index
        self.window_dates = {
            "train": (d[i_tr].date(), d[-1].date()),
            "unseen": (d[0].date(), d[i_tr - 1].date()),
            "full": (d[0].date(), d[-1].date()),
        }
        self.objective = objective
        self.key = OBJECTIVES[objective]
        self.rng = np.random.default_rng(seed)
        self.n_dedup_checks = 0
        # grid indices (entries: value > threshold <=> threshold < value)
        self._ia = self._idx_greater(self.rsi)
        self._ib = self._idx_greater(self.vrp)
        if not self.cfg.use_tdex:
            self._ic = np.full(n, NG)
        else:
            tn = np.isnan(self.tdex)
            self._ic = np.where(tn, NG if self.cfg.gate_allow_missing_tdex else 0,
                                np.searchsorted(GRID, np.nan_to_num(self.tdex), "left"))
        # exits: value < threshold <=> threshold index >= #(grid <= value)
        self._ja = np.where(np.isnan(self.rsi), NG + 1,
                            np.searchsorted(GRID, np.nan_to_num(self.rsi), "right"))
        self._jb = np.where(np.isnan(self.vrp), NG + 1,
                            np.searchsorted(GRID, np.nan_to_num(self.vrp), "right"))

    @staticmethod
    def _idx_greater(x: np.ndarray) -> np.ndarray:
        """How many grid values are < x (0 if x is NaN: never true)."""
        return np.where(np.isnan(x), 0,
                        np.searchsorted(GRID, np.nan_to_num(x), "left"))

    # ------------------------------------------------------------------ #
    # Thresholds and engine                                               #
    # ------------------------------------------------------------------ #
    def current_thresholds(self) -> tuple[dict, dict]:
        """(entries, exits) as configured now (defaults included)."""
        ent_c = {int(k): (v or {}) for k, v in (self.cfg.trend_entries_per_quad or {}).items()}
        ext_c = {int(k): (v or {}) for k, v in (self.cfg.trend_exits_per_quad or {}).items()}
        ent, ext = {}, {}
        for q in QUADS:
            d = ent_c.get(q, {})
            ent[q] = {"rsi_buy": float(d.get("rsi_buy", self.pq[q].rsi_short_thr)),
                      "vrp_buy": float(d.get("vrp_buy", self.pq[q].vrp_short_thr)),
                      "tdex_buy": float(d.get("tdex_buy", self.cfg.tdex_on_thr))}
            u = ext_c.get(q, {})
            ext[q] = {"rsi": float(u.get("rsi", 0.0)), "vrp": float(u.get("vrp", 0.0))}
        return ent, ext

    def cfg_with(self, ent: dict, ext: dict) -> SignalConfig:
        return replace(self.cfg, trend_entries_per_quad={q: dict(v) for q, v in ent.items()},
                       trend_exits_per_quad={q: dict(v) for q, v in ext.items()})

    def _ev(self, ent: dict, ext: dict):
        return market_trend_arrays(self.ind, self.cfg_with(ent, ext), self.pq)

    def _positions(self, E: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Exact latch of trend_position (flat start), vectorised: in if
        the last 'enter' (E and no veto) comes after the last veto."""
        E, V = np.atleast_2d(E), np.atleast_2d(V)
        S = E & ~V
        last_s = np.maximum.accumulate(np.where(S, self.ord, -1), axis=1)
        last_r = np.maximum.accumulate(np.where(V, self.ord, -1), axis=1)
        return last_s > last_r

    def _metrics(self, pos: np.ndarray) -> dict:
        """Real metrics (SPX closes, T+1 execution, costs per side) for each
        window; vectorised over the candidates (rows of `pos`)."""
        pos = np.atleast_2d(pos)
        act = np.zeros_like(pos)
        act[:, 1:] = pos[:, :-1]
        change = np.zeros(pos.shape, dtype=bool)
        change[:, 1:] = act[:, 1:] != act[:, :-1]
        r = np.where(act, self.ret, 0.0) - change * self.cost
        entry = np.zeros_like(pos)
        entry[:, 1:] = pos[:, 1:] & ~pos[:, :-1]
        entry[:, 0] = pos[:, 0]
        out = {}
        for name, (a, b) in self.windows.items():
            rr = r[:, a:b]
            eq = np.cumprod(1.0 + rr, axis=1)
            mult = eq[:, -1].copy()     # copy: a view would keep all of eq in memory
            years = (b - a) / 252.0
            cagr = np.where(mult > 0, np.abs(mult) ** (1.0 / years) - 1.0, -1.0)
            peak = np.maximum.accumulate(np.maximum(eq, 1.0), axis=1)
            dd = (eq / peak - 1.0).min(axis=1)
            calmar = np.where(dd < 0, cagr / np.where(dd < 0, -dd, 1.0), cagr)
            sd = rr.std(axis=1, ddof=1)
            sharpe = np.where(sd > 0, rr.mean(axis=1) / np.where(sd > 0, sd, 1.0)
                              * np.sqrt(252.0), 0.0)
            out[name] = {"multiple": mult, "cagr_pct": cagr * 100.0,
                         "max_dd_pct": dd * 100.0, "calmar": calmar,
                         "sharpe": sharpe,
                         "time_in_market_pct": pos[:, a:b].mean(axis=1) * 100.0,
                         "entries": entry[:, a:b].sum(axis=1).astype(float)}
        return out

    @staticmethod
    def _row(m: dict, i: int = 0) -> dict:
        return {f: {k: float(v[i]) for k, v in d.items()} for f, d in m.items()}

    def config_metrics(self, ent: dict, ext: dict) -> dict:
        E, V = self._ev(ent, ext)
        return self._row(self._metrics(self._positions(E, V)))

    def today_metrics(self) -> dict:
        """The params.yaml configuration as it is (real engine), from the
        same start date, flat."""
        sig = generate_signals(self.ind, self.pq, self.cfg_today)
        return self._row(self._metrics(sig["pos_spx"].to_numpy(dtype=bool)[None, :]))

    def _obj(self, m: dict) -> float:
        return float(m["train"][self.key])

    # ------------------------------------------------------------------ #
    # ENTRY exploration of a regime: 201^3 combinations                   #
    # ------------------------------------------------------------------ #
    def _explore_entries(self, q: int, ent: dict, ext: dict) -> dict:
        E_fix, V = self._ev(ent, ext)
        mq = self.qi == q
        ia, ib, ic = self._ia, self._ib, self._ic
        h = np.zeros((NG, NG, NG), dtype=np.uint64)
        f = np.empty((NG, NG, NG), dtype=np.int32)
        K = np.uint64(1_000_003)
        free = ~V
        starts = np.flatnonzero(free & ~np.r_[False, free[:-1]])
        ends = np.flatnonzero(free & ~np.r_[free[1:], False])
        for s, e in zip(starts, ends):
            days = np.arange(s, e + 1)
            fixed = days[(~mq[days]) & E_fix[days]]
            F = int(fixed[0]) if len(fixed) else e + 1      # first "fixed" entry
            gq = days[mq[days] & (days < F)]
            if len(gq) == 0:
                continue                                    # same outcome for all
            f.fill(F - s)
            for t in gq[::-1]:                              # the earliest wins
                a, b, c = ia[t], ib[t], ic[t]
                if a and b and c:
                    f[:a, :b, :c] = t - s
            np.multiply(h, K, out=h)
            np.add(h, f.astype(np.uint64), out=h)
        flat = h.ravel()
        _, first, inv = np.unique(flat, return_index=True, return_inverse=True)
        inv = inv.reshape(-1).astype(np.int32)
        A, B, C = np.unravel_index(first, (NG, NG, NG))
        mrsi, mvrp = self.rsi[mq], self.vrp[mq]
        mtdn = np.isnan(self.tdex[mq])
        mtd = np.nan_to_num(self.tdex[mq])

        def simulate(a, b, c):
            k = len(a)
            Eb = np.broadcast_to(E_fix, (k, self.n)).copy()
            with np.errstate(invalid="ignore"):
                cond = ((mrsi[None, :] > GRID[a][:, None])
                        & (mvrp[None, :] > GRID[b][:, None]))
                if self.cfg.use_tdex:
                    cond &= np.where(mtdn[None, :], bool(self.cfg.gate_allow_missing_tdex),
                                     mtd[None, :] > GRID[c][:, None])
            Eb[:, mq] = cond
            return self._metrics(self._positions(Eb, V))

        metr = self._in_batches(simulate, (A, B, C))
        # dedup check: 100 random combinations simulated directly
        pick = self.rng.integers(0, NG ** 3, 100)
        pa, pb, pc = np.unravel_index(pick, (NG, NG, NG))
        direct = simulate(pa, pb, pc)
        self._compare(direct, metr, inv[pick], f"entries Q{q}")
        return {"shape": (NG, NG, NG), "inv": inv, "metrics": metr,
                "keys": ("rsi_buy", "vrp_buy", "tdex_buy")}

    # ------------------------------------------------------------------ #
    # EXIT exploration of a regime: 201^2 combinations                    #
    # ------------------------------------------------------------------ #
    def _explore_exits(self, q: int, ent: dict, ext: dict) -> dict:
        mode = str(self.cfg.trend_confirmation or "or").lower()
        if mode == "total":
            raise ValueError("with trend_filter.confirmation: total the veto has "
                             "no RSI/VRP thresholds to calibrate")
        E, V_fix = self._ev(ent, ext)
        D = np.flatnonzero((self.qi == q) & self.broken)   # where the veto is possible
        A, B = np.meshgrid(np.arange(NG), np.arange(NG), indexing="ij")
        A, B = A.ravel(), B.ravel()
        if len(D):
            ca = A[:, None] >= self._ja[D][None, :]
            cb = B[:, None] >= self._jb[D][None, :]
            P = (ca & cb) if mode == "and" else (ca | cb)
            key_p = np.packbits(P, axis=1)
            _, first, inv = np.unique(key_p, axis=0, return_index=True,
                                      return_inverse=True)
        else:
            first, inv = np.array([0]), np.zeros(len(A), dtype=np.int64)
        inv = inv.reshape(-1).astype(np.int32)

        def simulate(a, b):
            k = len(a)
            Vb = np.broadcast_to(V_fix, (k, self.n)).copy()
            if len(D):
                ca = a[:, None] >= self._ja[D][None, :]
                cb = b[:, None] >= self._jb[D][None, :]
                Vb[:, D] = (ca & cb) if mode == "and" else (ca | cb)
            return self._metrics(self._positions(E, Vb))

        metr = self._in_batches(simulate, (A[first], B[first]))
        pick = self.rng.integers(0, NG ** 2, 100)
        direct = simulate(A[pick], B[pick])
        self._compare(direct, metr, inv[pick], f"exits Q{q}")
        return {"shape": (NG, NG), "inv": inv, "metrics": metr, "keys": ("rsi", "vrp")}

    # ------------------------------------------------------------------ #
    def _in_batches(self, simulate, coords) -> dict:
        k = len(coords[0])
        out: dict = {}
        for i0 in range(0, k, BATCH):
            part = simulate(*[c[i0:i0 + BATCH] for c in coords])
            for f, d in part.items():
                for m, v in d.items():
                    out.setdefault(f, {}).setdefault(m, []).append(v)
        return {f: {m: np.concatenate(v) for m, v in d.items()} for f, d in out.items()}

    def _compare(self, direct: dict, metr: dict, which: np.ndarray, what: str) -> None:
        for f, d in direct.items():
            for m, v in d.items():
                if not np.allclose(v, metr[f][m][which], rtol=1e-9, atol=1e-9):
                    raise VerificationFailed(
                        f"internal check failed ({what}, {f}/{m}): the "
                        "count of the combinations does not match the "
                        "direct simulation")
        self.n_dedup_checks += len(which)

    # ------------------------------------------------------------------ #
    # Choosing the best point of a regime                                 #
    # ------------------------------------------------------------------ #
    def _choose(self, exp: dict, current: dict, obj_current: float):
        """(chosen thresholds, action) with action:
          'keep'        = nothing changes;
          'recentre'    = same outcome, moved to the innermost point of its
                          plateau: positions and historical numbers IDENTICAL;
          'change_tie'  = different outcome but TIED in training (identical
                          objective since 2018): with a deeper plateau; only
                          the unseen years can change;
          'change'      = different outcome with a BETTER training objective.

        Robust tie-break (default):
          1. never go down: switch to an outcome with a different objective
             only if the training objective really improves;
          2. among the outcomes tied in training (the current one included)
             the deepest plateau wins (ties: the largest, then the current
             one, then the closest): the choice no longer depends on the
             starting point;
          3. inside the plateau the innermost point wins (ties: the closest
             to the centroid, then to the current threshold);
          4. if the current outcome wins the threshold is only recentred.
        'nearest' = previous rule (keep; if it changes, the closest point).
        'unseen'  = among the outcomes tied in training the best Calmar in
                    the unseen years wins (it consumes the clean test)."""
        shape, keys = exp["shape"], exp["keys"]
        obj_u = exp["metrics"]["train"][self.key]
        best = float(np.nanmax(obj_u))
        cur_best = obj_current >= best - TOL
        ties = np.flatnonzero(obj_u >= best - TOL)
        if self.tiebreak == "nearest":
            if cur_best:
                return dict(current), "keep"
            members = np.flatnonzero(np.isin(exp["inv"], ties))
            coords = np.unravel_index(members, shape)
            dist = np.zeros(len(members))
            for c, k in zip(coords, keys):
                dist += (GRID[c] - current[k]) ** 2
            j = int(np.argmin(dist))
            return {k: _g(GRID[c[j]]) for c, k in zip(coords, keys)}, "change"

        # robust / unseen: outcome of the current threshold on the grid
        idx_cur = _idx(current, keys)
        u_cur = -1
        if _on_grid(current, keys):
            u_cur = int(exp["inv"].reshape(shape)[idx_cur])
            # caution: the outcome on the grid must give the current objective
            if abs(float(obj_u[u_cur]) - obj_current) > 1e-9 * max(1.0, abs(obj_current)):
                u_cur = -1
        if cur_best and u_cur < 0:
            return dict(current), "keep"        # threshold off the grid: untouched
        if self.tiebreak == "unseen":
            mv = exp["metrics"]["unseen"]["calmar"]
            top = float(np.nanmax(mv[ties]))
            cand = ties[mv[ties] >= top - TOL]
        else:
            cand = ties
        # among the outcomes tied in training (the current one included, if
        # it is among the best) the deepest plateau wins: so the choice does
        # not depend on the starting point
        u = self._deepest(exp, cand, idx_cur, u_cur)
        pts = _groups(exp["inv"], np.array([u]), shape)[u]
        point, _ = _innermost_point(pts, idx_cur)
        chosen = {k: _g(GRID[i]) for k, i in zip(keys, point)}
        if u != u_cur:
            return chosen, ("change_tie" if cur_best else "change")
        return chosen, ("keep" if point == idx_cur else "recentre")

    @staticmethod
    def _deepest(exp: dict, outcomes: np.ndarray, idx_cur: tuple, u_cur: int = -1) -> int:
        """Among the given outcomes, the one with the deepest plateau. Ties:
        more combinations, then the current one, then the inner point
        closest to the current threshold."""
        best, best_key = None, None
        for u, pts in _groups(exp["inv"], np.asarray(outcomes), exp["shape"]).items():
            point, depth = _innermost_point(pts, idx_cur)
            d = sum((a - b) ** 2 for a, b in zip(point, idx_cur))
            k = (-depth, -len(pts), u != u_cur, d, u)
            if best_key is None or k < best_key:
                best, best_key = u, k
        return int(best)

    def _ties(self, exp: dict, u: int, idx: tuple) -> dict:
        """Outcomes tied in training with the chosen outcome (choice
        included): size, box, depth and - INFORMATIONAL ONLY - the numbers
        of the unseen years."""
        shape, keys = exp["shape"], exp["keys"]
        oa = exp["metrics"]["train"][self.key]
        mv = exp["metrics"]["unseen"]
        ties = np.flatnonzero(np.abs(oa - oa[u]) <= TOL)
        cnt = np.bincount(exp["inv"], minlength=len(oa))
        ties = ties[np.argsort(-cnt[ties], kind="stable")]
        sel = ties[:MAX_TIES]
        if u not in sel:
            sel = np.r_[u, sel[:-1]]
        rows = []
        for e, pts in _groups(exp["inv"], sel, shape).items():
            point, depth = _innermost_point(pts, idx)
            rows.append({
                "outcome": e, "combinations": int(len(pts)), "depth": int(depth),
                "box": {k: (_g(GRID[pts[:, i].min()]), _g(GRID[pts[:, i].max()]))
                        for i, k in enumerate(keys)},
                "inner_point": {k: _g(GRID[p]) for k, p in zip(keys, point)},
                "train_objective": float(oa[e]),
                "unseen_calmar": float(mv["calmar"][e]),
                "unseen_cagr_pct": float(mv["cagr_pct"][e]),
                "unseen_max_dd_pct": float(mv["max_dd_pct"][e]),
                "chosen": e == u})
        rows.sort(key=lambda r: (not r["chosen"], -r["depth"], -r["combinations"], r["outcome"]))
        return {"rows": rows, "outcomes": int(len(ties)),
                "combinations": int(cnt[ties].sum()),
                "truncated": bool(len(ties) > len(sel)),
                "all": ties}

    def _describe(self, exp: dict, chosen: dict) -> dict:
        """Plateau (combinations with THE SAME outcome as the choice),
        margin, ties in training, neighbours at +-0.5 and maps for the report."""
        shape, keys = exp["shape"], exp["keys"]
        idx = _idx(chosen, keys)
        inv = exp["inv"].reshape(shape)
        u = int(inv[idx])
        members = np.flatnonzero(exp["inv"] == u)
        coords = np.unravel_index(members, shape)
        plateau = {k: (_g(GRID[c.min()]), _g(GRID[c.max()]))
                   for c, k in zip(coords, keys)}
        # margin: depth of the chosen point and free steps for each threshold
        pts_u = np.stack(coords, axis=1)
        lo, depth = _depth_map(pts_u)
        margin = int(depth[tuple(int(i - l) for i, l in zip(idx, lo))])
        max_margin = int(depth.max())
        per_axis = _axis_margins(inv == u, idx)
        margins = {k: {"down": g, "up": s, "min": min(g, s), "width": g + s}
                   for k, (g, s) in zip(keys, per_axis)}
        notes = []
        if len(members) == 1:
            notes.append("isolated point")
        else:
            names = {"rsi_buy": "RSI", "vrp_buy": "VRP", "tdex_buy": "TDEX", "rsi": "RSI", "vrp": "VRP"}
            zero = [names[k] for k, m in margins.items() if m["width"] == 0]
            one = [names[k] for k, m in margins.items() if m["width"] == 1]
            edge = [names[k] for k, m in margins.items() if m["width"] >= 2 and m["min"] == 0]
            if zero:
                notes.append("zero width on " + " and ".join(zero))
            if one:
                notes.append("only one step wide on " + " and ".join(one))
            if edge:
                notes.append("on the edge on " + " and ".join(edge))
        ties = self._ties(exp, u, idx)
        sl = tuple(slice(max(0, i - 5), min(NG, i + 6)) for i in idx)
        near = np.unique(inv[sl])
        oa = exp["metrics"]["train"][self.key]
        om = exp["metrics"]["unseen"]["calmar"]
        cnt = np.bincount(inv[sl].ravel(), minlength=len(oa))[near]
        rep = lambda arr: np.repeat(arr[near], cnt)
        neighbours = {"median": float(np.median(rep(oa))), "worst": float(rep(oa).min()),
                      "unseen_median": float(np.median(rep(om))),
                      "unseen_worst": float(rep(om).min())}
        if len(shape) == 3:
            slc = inv[:, :, idx[2]]
            slice_title = f"{keys[2]} = {chosen[keys[2]]:g}"
        else:
            slc = inv
            slice_title = ""
        # table of outcomes: for each distinct outcome, how many combinations
        # produce it and the box of thresholds that contains them
        order = np.argsort(exp["inv"], kind="stable")
        inv_ord = exp["inv"][order]
        cuts = np.r_[0, np.flatnonzero(np.diff(inv_ord)) + 1]
        counts = np.diff(np.r_[cuts, len(inv_ord)])
        tab = {"outcome": inv_ord[cuts], "combinations": counts}
        for c, k in zip(np.unravel_index(order, shape), keys):
            tab[f"{k}_from"] = GRID[np.minimum.reduceat(c, cuts)]
            tab[f"{k}_to"] = GRID[np.maximum.reduceat(c, cuts)]
        for win in ("train", "unseen", "full"):
            for m in ("calmar", "cagr_pct", "max_dd_pct", "time_in_market_pct", "entries"):
                tab[f"{win}_{m}"] = exp["metrics"][win][m][tab["outcome"]]
        tab["chosen"] = tab["outcome"] == u
        tab["tie_in_train"] = np.isin(tab["outcome"], ties["all"])
        depth_e = np.full(len(oa), np.nan)
        for r in ties["rows"]:
            depth_e[r["outcome"]] = r["depth"]
        tab["depth"] = depth_e[tab["outcome"]]
        ties.pop("all")
        return {"combinations": int(np.prod(shape)),
                "table": tab,
                "distinct_outcomes": int(len(oa)),
                "same_outcome": int(len(members)), "plateau": plateau,
                "margin": margin, "max_margin": max_margin,
                "margins": margins, "notes": notes, "ties": ties,
                "neighbours": neighbours,
                "train_map": exp["metrics"]["train"][self.key][slc],
                "unseen_map": exp["metrics"]["unseen"]["calmar"][slc],
                "slice": slice_title, "point": idx[:2]}

    # ------------------------------------------------------------------ #
    # Rounds                                                              #
    # ------------------------------------------------------------------ #
    def calibrate(self, part: str, max_rounds: int = 10, progress=None) -> dict:
        if part not in ("entries", "exits"):
            raise ValueError("part must be 'entries' or 'exits'")
        say = progress or (lambda s: None)
        ent0, ext0 = self.current_thresholds()
        ent = {q: dict(v) for q, v in ent0.items()}
        ext = {q: dict(v) for q, v in ext0.items()}
        history, explorations = [], {}
        rounds = 0
        for rnd in range(1, max_rounds + 1):
            rounds = rnd
            changed = False
            for q in QUADS:
                obj_cur = self._obj(self.config_metrics(ent, ext))
                if part == "entries":
                    exp = self._explore_entries(q, ent, ext)
                    new, action = self._choose(exp, ent[q], obj_cur)
                else:
                    exp = self._explore_exits(q, ent, ext)
                    new, action = self._choose(exp, ext[q], obj_cur)
                explorations[q] = exp
                before = dict(ent[q] if part == "entries" else ext[q])
                ch = action != "keep"
                if ch:
                    (ent if part == "entries" else ext)[q] = new
                    changed = True
                obj_after = self._obj(self.config_metrics(ent, ext))
                if (action in ("recentre", "change_tie")
                        and abs(obj_after - obj_cur) > 1e-9 * max(1.0, abs(obj_cur))):
                    raise VerificationFailed(
                        f"{action} Q{q}: the training objective changed "
                        f"({obj_cur} → {obj_after}) but it had to stay identical")
                history.append({"round": rnd, "quad": q, "before": before,
                                "after": dict(new), "changed": ch, "action": action,
                                "objective_before": obj_cur, "objective_after": obj_after,
                                "distinct_outcomes": int(len(exp["metrics"]["train"][self.key]))})
                say(f"round {rnd} · Q{q} {QUAD_LABELS[q]}: "
                    f"{int(np.prod(exp['shape'])):,} combinations tried"
                    + f" ({len(exp['metrics']['train'][self.key])} distinct outcomes) → "
                    + {"change": "CHANGE",
                       "change_tie": "changes outcome among ties (deeper plateau)",
                       "recentre": "recentred (same results)",
                       "keep": "kept"}[action] + f" · {self.key} "
                    f"{obj_cur:.3f} → {obj_after:.3f}")
            if not changed:
                break
        converged = not changed
        # what happened to each regime over the whole calibration
        actions = {}
        for q in QUADS:
            a = [s["action"] for s in history if s["quad"] == q]
            fin, ini = (ent, ent0) if part == "entries" else (ext, ext0)
            actions[q] = ("kept" if fin[q] == ini[q] else
                          "changed" if "change" in a else
                          "changed_among_ties" if "change_tie" in a else "recentred")
        # the explorations of the last round describe the final point only if
        # no regime changed after its own exploration: redo the "old" ones
        # to get maps consistent with the final thresholds
        if not converged:
            for q in QUADS:
                explorations[q] = (self._explore_entries(q, ent, ext) if part == "entries"
                                   else self._explore_exits(q, ent, ext))
        final = ent if part == "entries" else ext
        per_quad = {q: self._describe(explorations[q], final[q]) for q in QUADS}
        verification = self.verify_engine(ent, ext, part)
        return {
            "part": part, "rounds": rounds, "converged": converged,
            "history": history, "objective": self.objective, "key": self.key,
            "mode_today": self.cfg_today.market_mode,
            "start": {"entries": ent0, "exits": ext0},
            "final": {"entries": ent, "exits": ext},
            "references": {"today": self.today_metrics(),
                           "start": self.config_metrics(ent0, ext0),
                           "calibrated": self.config_metrics(ent, ext)},
            "per_quad": per_quad, "windows": self.window_dates,
            "cost_bps": self.cost_bps, "verification": verification,
            "tiebreak": self.tiebreak, "actions": actions,
        }

    # ------------------------------------------------------------------ #
    def verify_engine(self, ent: dict, ext: dict, part: str, n: int = 12) -> dict:
        """The fast engine must give the SAME positions as the real engine
        (generate_signals) on the final result and on random combinations."""
        cases = [(ent, ext)]
        for _ in range(n):
            q = int(self.rng.integers(1, 5))
            i2 = {k: dict(v) for k, v in ent.items()}
            u2 = {k: dict(v) for k, v in ext.items()}
            if part == "entries":
                i2[q] = {k: _g(self.rng.choice(GRID)) for k in ("rsi_buy", "vrp_buy", "tdex_buy")}
            else:
                u2[q] = {k: _g(self.rng.choice(GRID)) for k in ("rsi", "vrp")}
            cases.append((i2, u2))
        for i2, u2 in cases:
            E, V = self._ev(i2, u2)
            fast = self._positions(E, V)[0]
            real = generate_signals(self.ind, self.pq, self.cfg_with(i2, u2))["pos_spx"].to_numpy(dtype=bool)
            ref = trend_position(E, V, False)
            if not (np.array_equal(fast, real) and np.array_equal(ref, real)):
                diff = int((fast != real).sum())
                raise VerificationFailed(f"the fast engine differs from the real engine "
                                         f"on {diff} days: results discarded")
        return {"engine_cases": len(cases), "dedup_cases": self.n_dedup_checks}


# ---------------------------------------------------------------------- #
# Writing to params.yaml (surgical, comments preserved, .bak backup)      #
# ---------------------------------------------------------------------- #
_MARKERS = {"entries": ("ENTRY THRESHOLDS PER REGIME", "trend_entries_per_quad"),
            "exits": ("TREND FILTER CONFIRMATION THRESHOLDS PER REGIME",
                      "trend_exits_per_quad")}


def grid_yaml_block(part: str, thresholds: dict, label: str, today: str) -> str:
    marker, key = _MARKERS[part]
    if part == "entries":
        head = [
            f"  # >>> {marker} (market_mode: trend) <<<",
            "  # After a trend-filter exit the bot RE-ENTERS only when RSI, VRP and",
            "  # TDEX are ABOVE these thresholds (regime of the day) and there is no veto.",
            f"  # Calibrated with the grid on {today} ({label}).",
            "  # Delete the block to go back to the workbook's rsi_short/vrp_short",
            "  # and to tdex_gate_buy_spx.",
        ]
        fmt = lambda d: f"rsi_buy: {d['rsi_buy']:g}, vrp_buy: {d['vrp_buy']:g}, tdex_buy: {d['tdex_buy']:g}"
    else:
        head = [
            f"  # >>> {marker} (market_mode: trend) <<<",
            "  # The bot EXITS when SPX closes below the trend-filter average AND (RSI",
            "  # below 'rsi' OR VRP below 'vrp') in the regime of the day.",
            f"  # Calibrated with the grid on {today} ({label}).",
            "  # Delete the block to go back to the zero confirmation (rsi<0 or vrp<0).",
        ]
        fmt = lambda d: f"rsi: {d['rsi']:g}, vrp: {d['vrp']:g}"
    lines = head + [f"  {key}:"]
    for q in QUADS:
        lines.append(f"    {q}: {{{fmt(thresholds[q])}}}   # {QUAD_LABELS[q]}")
    return "\n".join(lines)


MODE_BLOCK = """  # >>> MODE OF THE MARKET LEG (added 2026-10-14) <<<
  #   classic = as always: entry/exit of the voting method and, if
  #             enabled, the trend filter on top; once the veto ends the
  #             bot re-enters automatically.
  #   trend   = EXITS only via the trend filter (SPX below the average AND
  #             RSI/VRP confirmation, per regime) and RE-ENTERS only when
  #             the entry rule is true (RSI, VRP and TDEX above threshold,
  #             per regime). The thresholds live in the
  #             trend_entries_per_quad / trend_exits_per_quad blocks and
  #             are calibrated from the menu, item 14. The trend filter is
  #             forced on (window and confirmation from trend_filter).
  market_mode: {value}
"""


def apply_yaml(params_path: str, part: str, thresholds: dict, label: str,
               today: str | None = None) -> None:
    """Writes the block of the calibrated part and sets market_mode: trend.
    Backup in params.yaml.bak; the file must stay valid YAML."""
    import yaml as _yaml
    today = today or date.today().isoformat()
    marker, key = _MARKERS[part]
    src = Path(params_path).read_text(encoding="utf-8")
    shutil.copyfile(params_path, params_path + ".bak")
    new = src
    # 1) market_mode: trend
    pm = re.compile(r"^([ \t]*market_mode:[ \t]*)[^\n#]*", re.MULTILINE)
    if pm.search(new):
        new = pm.sub(lambda m: m.group(1) + "trend", new, count=1)
    else:
        mode_block = MODE_BLOCK.format(value="trend")
        ar = re.compile(r"^[ \t]*market_reentry:[^\n]*\n", re.MULTILINE)
        m = ar.search(new)
        if m:
            new = new[: m.end()] + "\n" + mode_block + new[m.end():]
        else:
            new = _insert_before_params(new, mode_block)
    # 2) thresholds block
    block = grid_yaml_block(part, thresholds, label, today)
    pat = re.compile(
        r"(?:^[ \t]*#[^\n]*" + re.escape(marker) + r"[^\n]*\n(?:^[ \t]*#[^\n]*\n)*)?"
        r"^[ \t]*" + key + r":[ \t]*\n(?:^[ \t]+\d+:[^\n]*\n?)*",
        re.MULTILINE,
    )
    if pat.search(new):
        new = pat.sub(lambda m: block + "\n", new, count=1)
    else:
        new = _insert_before_params(new, block + "\n")
    parsed = _yaml.safe_load(new)
    vix = parsed.get("vix") or {}
    if vix.get("market_mode") != "trend" or len(vix.get(key) or {}) != 4:
        raise RuntimeError("after the edit params.yaml does not contain the "
                           "expected block inside vix: file NOT modified")
    Path(params_path).write_text(new, encoding="utf-8")


def _insert_before_params(src: str, block: str) -> str:
    anchor = re.compile(r"^([ \t]*# Parameters per quadrant)", re.MULTILINE)
    m = anchor.search(src)
    if not m:
        raise RuntimeError("anchor '# Parameters per quadrant' not found in the YAML")
    return src[: m.start()] + block + "\n" + src[m.start():]
