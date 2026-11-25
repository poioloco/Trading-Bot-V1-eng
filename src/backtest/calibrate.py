"""Recalibration of the long-vol leg for MAJORITY mode.

Problem: the rsi_long/vrp_long thresholds of the midterm workbook are
optimised for the GATE logic (3-condition AND). In majority mode (2-of-3)
the same thresholds become too permissive: individually "loose" because
the AND intersected them, in 2-of-3 they keep the bot in volatility almost
half of the time.

Method (same criterion as the per-quad TDEX seeds): FREQUENCY EQUALISATION.
For a target frequency f, a quadrant's threshold is the empirical
f-quantile of the smoothed indicator computed on the days of that regime
ONLY: P(rsi_smooth < threshold | quad) ~= f. So the leg votes with the same
selectivity in every regime, and f becomes the only knob.

The heavy computation (40-ETF regime + VIX chains) is done ONCE
(`prepare`): the long-vol thresholds do not affect the indicators, only the
votes, so each candidate costs a single vectorised pass of
`generate_signals`.

Candidates are evaluated through `SignalConfig.longvol_per_quad`, i.e.
EXACTLY the mechanism that then ends up in params.yaml: what you measure is
what you configure. `params_per_quad` (midterm workbook) is never modified;
the gate stays the faithful replica.

NB (2026-07-20): the calibration of the MARKET leg (sweep_market and
montecarlo_market, via `market_per_quad`) also works with
voting_method: gate — "calibrated gate": Excel entry unchanged, per-quadrant
exit anchored to the position, TDEX branch = deep fear. Without the
market_per_quad block the gate stays the exact replica. Only the LONG-VOL
calibration (sweep) remains exclusive to the majority.

NB on quality: without the real historical series of the long-vol
instrument the leg's P&L cannot be computed. The quality metrics are
declared PROXIES: VIX change captured while in position, average SPX
return on the days in position (it should be low/negative: the leg should
be on during stress) and coverage of SPX's worst days.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from ..regime.winner import RegimeParams
from ..vix.quad_params import QuadParams
from ..vix.signal_generator import MAJORITY, SignalConfig, generate_signals
from ..engine.regime_engine import StatefulRegime
from ..engine.vix_engine import StatefulVix

LEG_COLUMN = {"rsi": "rsi_smooth", "vrp": "vrp_smooth"}
LEG_KEY = {"rsi": "rsi_long", "vrp": "vrp_long"}


@dataclass
class Candidate:
    """A sweep candidate: label + overrides + statistics."""
    label: str
    overrides: dict[int, dict[str, float]]   # {} = no override (current)
    stats: dict


class LongVolCalibrator:
    def __init__(self, store, quads: dict[str, list[str]],
                 regime_params: RegimeParams, ma_windows: list[int],
                 vote_threshold: float, params_by_quad: dict[int, QuadParams],
                 cfg: SignalConfig, spx_symbol: str = "^SPX",
                 vix_symbol: str = "^VIX") -> None:
        # NB: no guard on the method here. The LONG-VOL calibration stays
        # majority-only (guard in sweep: longvol_per_quad does not exist in
        # the gate), but the MARKET leg (sweep_market/montecarlo_market) also
        # applies in gate: market_per_quad activates the "calibrated gate".
        self.params_by_quad = params_by_quad
        # base WITHOUT overrides: the starting point is the midterm workbook
        self.base_cfg = replace(cfg, longvol_per_quad=None)
        self.current_cfg = cfg   # as configured NOW (any overrides already in the yaml)
        self.regime = StatefulRegime(store, quads, regime_params, ma_windows,
                                     vote_threshold)
        self.vix = StatefulVix(store, params_by_quad, cfg, spx_symbol, vix_symbol)
        self.ind: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    def prepare(self, end: date) -> None:
        """The heavy run, done once: day-by-day regime + VIX chains. The
        indicators do NOT depend on the long-vol thresholds."""
        winner, _ = self.regime.warm_up(end)
        ind, _, _ = self.vix.warm_up(winner, end)
        self.ind = ind

    # ------------------------------------------------------------------ #
    def thresholds_for_frequency(
        self, leg: str, freq_pct: float, quads: list[int]
    ) -> dict[int, float]:
        """Per-quadrant threshold = empirical quantile of the target
        frequency, computed on the days of that regime only. Rounded to 1
        decimal (what ends up in the YAML is what gets evaluated)."""
        self._require_prepared()
        col = LEG_COLUMN[leg]
        out: dict[int, float] = {}
        q_series = self.ind["quad"].astype(int)
        for q in quads:
            vals = self.ind.loc[q_series == q, col].dropna()
            if len(vals) < 30:
                raise ValueError(f"quad {q}: too few days ({len(vals)}) "
                                 "to estimate the quantile")
            out[q] = round(float(np.quantile(vals, freq_pct / 100.0)), 1)
        return out

    # ------------------------------------------------------------------ #
    def evaluate(self, overrides: dict[int, dict[str, float]] | None,
                 label: str = "") -> Candidate:
        """Evaluates a set of long-vol overrides (None/{} = pure workbook)."""
        self._require_prepared()
        cfg = replace(self.base_cfg, longvol_per_quad=overrides or None)
        sig = generate_signals(self.ind, self.params_by_quad, cfg)
        stats = self._stats(sig, overrides or {})
        return Candidate(label=label, overrides=overrides or {}, stats=stats)

    # ------------------------------------------------------------------ #
    def sweep(self, legs: list[str], freqs_pct: list[float],
              quads: list[int]) -> list[Candidate]:
        """'current' row (today's config) + one row per target frequency."""
        if self.current_cfg.method != MAJORITY:
            raise ValueError(
                "calibrate-longvol only makes sense with voting_method: majority "
                "(longvol_per_quad does not exist in the gate, which replicates the Excel)"
            )
        out = [self.evaluate(self.current_cfg.longvol_per_quad, label="current")]
        for f in freqs_pct:
            ovr: dict[int, dict[str, float]] = {}
            for leg in legs:
                thr = self.thresholds_for_frequency(leg, f, quads)
                for q, v in thr.items():
                    ovr.setdefault(q, {})[LEG_KEY[leg]] = v
            out.append(self.evaluate(ovr, label=f"f={f:g}%"))
        return out

    # ------------------------------------------------------------------ #
    def _stats(self, sig: pd.DataFrame, overrides: dict) -> dict:
        ind = self.ind
        # position straight from the generator (pos_vix): the same machine
        # that produces the events, no parallel reconstruction
        pos_s = sig["pos_vix"].astype(bool)
        # T+1 convention like the backtest's market leg: today's (EOD)
        # signal is exposed from tomorrow
        active = pos_s.shift(1).fillna(False).astype(bool)

        n = len(ind)
        entries = int((pos_s & ~pos_s.shift(1).fillna(False).astype(bool)).sum())
        days_in = int(pos_s.sum())
        avg_duration = days_in / entries if entries else 0.0

        # ---- quality proxies (declared, not real P&L) -------------------
        dlog_vix = np.log(ind["vix"]).diff().fillna(0.0)
        vix_captured = float(np.exp((dlog_vix[active]).sum()))  # VIX multiple
        ret_spx = ind["spx"].pct_change().fillna(0.0)
        spx_in_vol_bp = float(ret_spx[active].mean() * 1e4) if active.any() else float("nan")
        # coverage of the worst days: share of SPX's worst 5% spent in a
        # long-vol position
        crash_threshold = ret_spx.quantile(0.05)
        crash_days = ret_spx <= crash_threshold
        coverage = float(active[crash_days].mean() * 100) if crash_days.any() else float("nan")

        # ---- vote frequencies per quadrant (diagnostics) ----------------
        q_series = ind["quad"].astype(int)
        freq_rsi, freq_vrp, vol_share = {}, {}, {}
        for q in (1, 2, 3, 4):
            m = q_series == q
            if not m.any():
                continue
            freq_rsi[q] = round(float(sig.loc[m, "vote_rsi_longvol"].mean() * 100), 1)
            freq_vrp[q] = round(float(sig.loc[m, "vote_vrp_longvol"].mean() * 100), 1)
            vol_share[q] = round(float(pos_s[m].mean() * 100), 1)

        return {
            "time_in_vol_pct": round(days_in / n * 100, 1),
            "entries": entries,
            "avg_duration_days": round(avg_duration, 1),
            "vix_captured_x": round(vix_captured, 2),
            "spx_in_vol_bp_day": round(spx_in_vol_bp, 1),
            "crash_coverage_pct": round(coverage, 1),
            "rsi_vote_freq_pct": freq_rsi,
            "vrp_vote_freq_pct": freq_vrp,
            "time_in_vol_per_quad_pct": vol_share,
            "days": n,
        }

    # ------------------------------------------------------------------ #
    # MARKET leg (same philosophy: per-quadrant quantiles)                #
    # ------------------------------------------------------------------ #
    def sweep_market(self, legs: list[str], quantiles_pct: list[float],
                     quads: list[int]) -> list[Candidate]:
        """Candidates for the MARKET leg. The knob is the EXIT threshold:
        in majority mode the sell SPX (rsi<0 OR vrp<0, anchored to the
        position) is hyperactive and produces churn (exits after a few
        days). Here a quadrant's exit threshold becomes the empirical
        quantile g of the indicator on the days of that regime: you exit
        only when the indicator is in the weakest g% of ITS regime. The
        entry stays the workbook's. Unlike the long-vol, here the P&L can
        really be computed (SPX closes): timing× and drawdown are real."""
        self._require_prepared()
        out = [self.evaluate_market(self.current_cfg.market_per_quad,
                                    label="current")]
        for g in quantiles_pct:
            ovr: dict[int, dict[str, float]] = {}
            for leg in legs:
                thr = self.thresholds_for_frequency(leg, g, quads)
                for q, v in thr.items():
                    ovr.setdefault(q, {})[f"{LEG_KEY[leg].split('_')[0]}_exit"] = v
            out.append(self.evaluate_market(ovr, label=f"q={g:g}%"))
        return out

    def evaluate_market(self, overrides: dict | None,
                        label: str = "") -> Candidate:
        """Evaluates a set of market overrides (None/{} = Excel zeros). The
        base is the CURRENT config (already calibrated long-vol and TDEX
        stay active)."""
        self._require_prepared()
        cfg = replace(self.current_cfg, market_per_quad=overrides or None)
        sig = generate_signals(self.ind, self.params_by_quad, cfg)
        return Candidate(label=label, overrides=overrides or {},
                         stats=self._stats_market(sig))

    def _stats_market(self, sig: pd.DataFrame) -> dict:
        ind = self.ind
        pos = sig["pos_spx"].astype(bool)
        prev = pos.shift(1, fill_value=False).astype(bool)
        active = prev                       # T+1 like the backtest (EOD -> tomorrow)
        n = len(ind)
        entries = int((pos & ~prev).sum())
        ret = ind["spx"].pct_change().fillna(0.0)
        eq = (1.0 + ret.where(active, 0.0)).cumprod()
        bh = (1.0 + ret).cumprod()
        dd = float((eq / eq.cummax() - 1.0).min())
        q_series = ind["quad"].astype(int)
        per_quad = {q: round(float(pos[q_series == q].mean() * 100), 1)
                    for q in (1, 2, 3, 4) if (q_series == q).any()}
        return {
            "time_in_market_pct": round(float(pos.mean() * 100), 1),
            "entries": entries,
            "avg_duration_days": round(float(pos.sum()) / entries, 1) if entries else 0.0,
            "timing_multiple": round(float(eq.iloc[-1]), 2),
            "bh_multiple": round(float(bh.iloc[-1]), 2),
            "timing_max_drawdown_pct": round(dd * 100, 1),
            "time_per_quad_pct": per_quad,
            "days": n,
        }

    # ------------------------------------------------------------------ #
    # MONTE CARLO for the market leg: looks for the exit thresholds that  #
    # maximise a statistic, with walk-forward validation                  #
    # ------------------------------------------------------------------ #
    def montecarlo_market(self, n: int = 500, quads: list[int] | None = None,
                          objective: str = "calmar", train_fraction: float = 0.7,
                          seed: int = 42, q_min: float = 1.0,
                          q_max: float = 30.0, include_entries: bool = False,
                          buy_min: float = 10.0, buy_max: float = 70.0) -> dict:
        """Samples `n` combinations of exit quantiles (per quadrant: rsi,
        vrp and — if the leg is active in the vote and the train data is
        enough — TDEX's sell_spx; all independent, uniform in
        [q_min, q_max]%), evaluates them on the real engine and sorts them by
        the objective computed on the training window ONLY (first
        `train_fraction` of the days). The thresholds are quantiles
        estimated on the training window (no lookahead). For TDEX ONLY the
        sell_spx is sampled (the only one that touches the market exit):
        longvol and buy_spx stay as configured.

        With `include_entries=True` the ENTRY thresholds are ALSO sampled
        (rsi_buy/vrp_buy per quadrant, uniform quantiles in
        [buy_min, buy_max]%): threshold = quantile of the indicator in the
        regime, the market vote is cast when the indicator is ABOVE it. LOW
        entry quantile = you enter easily (state often true), HIGH = you
        enter only on real strength. Together with `market_reentry: state`
        it is the lever for timing: without it, after an exit you stay out
        until the workbook's AND becomes true again.

        Returns a dict with:
          candidates  : sorted list (best first) of dicts with quantiles,
                        overrides (rsi/vrp), tdex_sell (sell_spx thresholds
                        per quadrant), train and test metrics
          stability   : metrics of the winner's "neighbours" (quantiles
                        ±2pp): if the worst neighbour collapses, the
                        maximum is fragile
          sensitivity : for EACH threshold (quadrant × leg): how much it
                        matters, in which direction to move it and the
                        recommended value
          recommended : the config built from the recommended values,
                        evaluated on train and test (comparable with current)
          legs        : the sampled knobs ("rsi","vrp" and possibly "tdex")
          split       : dates of the train/test window
        """
        self._require_prepared()
        quads = quads or [1, 2, 3, 4]
        rng = np.random.default_rng(seed)
        n_train = int(len(self.ind) * train_fraction)
        idx_train = self.ind.index[:n_train]
        idx_test = self.ind.index[n_train:]
        m_train = self.ind.index.isin(idx_train)
        m_test = ~m_train

        q_series = self.ind["quad"].astype(int)

        # TDEX leg: sampled only if active in the vote and with enough train
        # data in every quadrant (otherwise it stays as configured)
        columns = {**LEG_COLUMN, "tdex": "tdex_smooth",
                   "rsi_buy": "rsi_smooth", "vrp_buy": "vrp_smooth"}
        L = self.current_cfg.legs or {}
        tdex_ok = bool(L.get("tdex", True)) and "tdex_smooth" in self.ind.columns
        if tdex_ok:
            for q in quads:
                m = (q_series == q) & m_train
                if self.ind.loc[m, "tdex_smooth"].dropna().shape[0] < 30:
                    tdex_ok = False
                    break
        legs = ["rsi", "vrp"] + (["tdex"] if tdex_ok else [])
        if include_entries:
            legs += ["rsi_buy", "vrp_buy"]
        # sampling range per knob: tight exits (weak tails of the regime),
        # entries over the whole useful band
        ranges = {leg: ((buy_min, buy_max) if leg.endswith("_buy")
                        else (q_min, q_max)) for leg in legs}

        # per-quadrant distributions ON THE TRAIN window only
        dist: dict[tuple[int, str], np.ndarray] = {}
        for q in quads:
            m = (q_series == q) & m_train
            for leg in legs:
                vals = self.ind.loc[m, columns[leg]].dropna().to_numpy()
                if len(vals) < 30:
                    raise ValueError(f"quad {q}: too few train days "
                                     f"({len(vals)}) to estimate the quantiles")
                dist[(q, leg)] = vals

        # per-quad TDEX base: current values (or global thresholds) onto
        # which only the sampled sell_spx is grafted; longvol/buy_spx untouched
        cfg0_pq = {int(k): dict(v)
                   for k, v in (self.current_cfg.tdex_per_quad or {}).items()}
        base_tdex_pq: dict[int, dict[str, float]] = {}
        for q in (1, 2, 3, 4):
            d = cfg0_pq.get(q, {})
            base_tdex_pq[q] = {
                "sell_spx": float(d.get("sell_spx", self.current_cfg.tdex_sell_thr)),
                "longvol": float(d.get("longvol", self.current_cfg.tdex_lv_thr)),
                "buy_spx": float(d.get("buy_spx", self.current_cfg.tdex_on_thr)),
            }

        def _overrides(qtl: dict) -> tuple[dict, dict]:
            mkt: dict[int, dict[str, float]] = {}
            tdex_sell: dict[int, float] = {}
            for q in quads:
                for leg in legs:
                    thr = float(np.quantile(dist[(q, leg)], qtl[(q, leg)] / 100.0))
                    if leg == "tdex":
                        tdex_sell[q] = round(thr, 1)
                    elif leg.endswith("_buy"):
                        mkt.setdefault(q, {})[leg] = round(thr, 1)
                    else:
                        mkt.setdefault(q, {})[f"{leg}_exit"] = round(thr, 1)
            return mkt, tdex_sell

        def _evaluate(qtl: dict) -> dict:
            mkt, tdex_sell = _overrides(qtl)
            cfg = replace(self.current_cfg, market_per_quad=mkt)
            if tdex_sell:
                tdex_pq = {q: dict(v) for q, v in base_tdex_pq.items()}
                for q, thr in tdex_sell.items():
                    tdex_pq[q]["sell_spx"] = thr
                cfg = replace(cfg, tdex_per_quad=tdex_pq)
            sig = generate_signals(self.ind, self.params_by_quad, cfg)
            pos = sig["pos_spx"].astype(bool)
            return {
                "quantiles": {q: {leg: round(qtl[(q, leg)], 1)
                                  for leg in legs} for q in quads},
                "overrides": mkt,
                "tdex_sell": tdex_sell,
                "train": self._window_metrics(pos, m_train),
                "test": self._window_metrics(pos, m_test),
            }

        candidates = []
        for _ in range(int(n)):
            qtl = {(q, leg): float(rng.uniform(*ranges[leg]))
                   for q in quads for leg in legs}
            candidates.append(_evaluate(qtl))
        # reference: current configuration (Excel zeros or block already applied)
        cfg0 = self.current_cfg
        sig0 = generate_signals(self.ind, self.params_by_quad, cfg0)
        pos0 = sig0["pos_spx"].astype(bool)
        current = {"quantiles": None, "overrides": cfg0.market_per_quad or {},
                   "tdex_sell": {},
                   "train": self._window_metrics(pos0, m_train),
                   "test": self._window_metrics(pos0, m_test)}

        key = {"calmar": "calmar", "return": "multiple",
               "sharpe": "sharpe"}.get(objective, "calmar")
        candidates.sort(key=lambda c: c["train"][key], reverse=True)

        # stability of the winner: neighbours at ±2 percentage points per knob
        neighbours = []
        if candidates:
            base = {(q, leg): candidates[0]["quantiles"][q][leg]
                    for q in quads for leg in legs}
            for delta in (-2.0, 2.0):
                for knob in base:
                    qtl = dict(base)
                    lo, hi = ranges[knob[1]]
                    qtl[knob] = float(np.clip(qtl[knob] + delta, lo, hi))
                    neighbours.append(_evaluate(qtl)["train"][key])
        stability = {
            "winner_objective": candidates[0]["train"][key] if candidates else None,
            "worst_neighbour": round(min(neighbours), 2) if neighbours else None,
            "median_neighbour": round(float(np.median(neighbours)), 2) if neighbours else None,
        }

        # sensitivity per threshold + recommended configuration: clear
        # numbers on WHICH knob matters, which way to move it and where to put it
        sensitivity = self._market_sensitivity(candidates, quads, legs, key,
                                               ranges)
        recommended = None
        if sensitivity:
            qtl_rec = {(s["quad"], s["leg"]): s["recommended"]["median"]
                       for s in sensitivity}
            recommended = _evaluate(qtl_rec)
        return {
            "candidates": candidates, "current": current, "stability": stability,
            "sensitivity": sensitivity, "recommended": recommended,
            "legs": legs, "objective": key,
            "split": {"train_from": str(idx_train[0].date()),
                      "train_to": str(idx_train[-1].date()),
                      "test_from": str(idx_test[0].date()) if len(idx_test) else None,
                      "test_to": str(idx_test[-1].date()) if len(idx_test) else None},
        }

    @staticmethod
    def _market_sensitivity(candidates: list[dict], quads: list[int],
                            legs: list[str], key: str,
                            ranges: dict[str, tuple[float, float]]) -> list[dict]:
        """Marginal effect of each threshold, read from the Monte Carlo samples.

        For each knob (quadrant × leg) the samples are split into three
        quantile bands (low / mid / high). The median of the objective in
        each band tells in which DIRECTION it pays to move the threshold;
        the difference between the best and worst band ("effect") tells HOW
        MUCH that threshold matters. The recommended value is the knob's
        median in the best 20% of the candidates (p25-p75 = typical range).
        List sorted by decreasing effect."""
        if not candidates:
            return []
        ys_all = np.array([c["train"][key] for c in candidates], dtype=float)
        spread = float(np.percentile(ys_all, 90) - np.percentile(ys_all, 10))
        n_top = max(1, int(round(len(candidates) * 0.2)))
        top = candidates[:n_top]                 # already sorted, best first
        names = ("low", "mid", "high")
        out: list[dict] = []
        for q in quads:
            for leg in legs:
                edges = np.linspace(*ranges[leg], 4)
                xs = np.array([c["quantiles"][q][leg] for c in candidates])
                bands = []
                for i, name in enumerate(names):
                    lo, hi = float(edges[i]), float(edges[i + 1])
                    m = (xs >= lo) & ((xs <= hi) if i == 2 else (xs < hi))
                    sel = [c for c, inside in zip(candidates, m) if inside]
                    if sel:
                        bands.append({
                            "name": name, "from": round(lo, 1), "to": round(hi, 1),
                            "n": len(sel),
                            key: round(float(np.median(
                                [c["train"][key] for c in sel])), 2),
                            "cagr_pct": round(float(np.median(
                                [c["train"]["cagr_pct"] for c in sel])), 2),
                            "max_dd_pct": round(float(np.median(
                                [c["train"]["max_dd_pct"] for c in sel])), 1),
                        })
                    else:
                        bands.append({"name": name, "from": round(lo, 1),
                                      "to": round(hi, 1), "n": 0, key: None,
                                      "cagr_pct": None, "max_dd_pct": None})
                med = [f[key] for f in bands if f[key] is not None]
                effect = round(max(med) - min(med), 2) if len(med) >= 2 else 0.0
                if spread > 0 and effect >= 0.5 * spread:
                    strength = "STRONG"
                elif spread > 0 and effect >= 0.2 * spread:
                    strength = "medium"
                else:
                    strength = "weak"
                best = max((f for f in bands if f[key] is not None),
                           key=lambda f: f[key], default=None)
                x_top = np.array([c["quantiles"][q][leg] for c in top])
                out.append({
                    "quad": q, "leg": leg, "effect": effect,
                    "strength": strength, "bands": bands,
                    "best_band": best["name"] if best else None,
                    "recommended": {
                        "median": round(float(np.median(x_top)), 1),
                        "p25": round(float(np.percentile(x_top, 25)), 1),
                        "p75": round(float(np.percentile(x_top, 75)), 1),
                    },
                })
        out.sort(key=lambda s: s["effect"], reverse=True)
        return out

    def _window_metrics(self, pos: pd.Series, mask) -> dict:
        """Real metrics (SPX closes, T+1 entry) over a window."""
        ret = self.ind["spx"].pct_change().fillna(0.0)[mask]
        act = pos.shift(1, fill_value=False).astype(bool)[mask]
        strat = ret.where(act, 0.0)
        eq = (1.0 + strat).cumprod()
        years = max(len(eq) / 252.0, 1e-9)
        multiple = float(eq.iloc[-1]) if len(eq) else 1.0
        cagr = multiple ** (1.0 / years) - 1.0
        dd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
        calmar = cagr / abs(dd) if dd < 0 else cagr
        sd = float(strat.std())
        sharpe = float(strat.mean()) / sd * np.sqrt(252.0) if sd > 0 else 0.0
        pos_w = pos[mask]
        return {
            "multiple": round(multiple, 2),
            "cagr_pct": round(cagr * 100, 2),
            "max_dd_pct": round(dd * 100, 1),
            "calmar": round(calmar, 2),
            "sharpe": round(sharpe, 2),
            "time_in_market_pct": round(float(pos_w.mean() * 100), 1),
            "years": round(years, 1),
        }

    def _require_prepared(self) -> None:
        if self.ind is None:
            raise RuntimeError("call prepare(end) first")


# ---------------------------------------------------------------------- #
# YAML block ready to paste / apply                                       #
# ---------------------------------------------------------------------- #
def yaml_block(overrides: dict[int, dict[str, float]], freq_label: str,
               today: str) -> str:
    """The `longvol_per_quad` block as text, with the explanations."""
    lines = [
        "  # >>> LONG-VOL THRESHOLDS PER QUADRANT (majority mode) <<<",
        "  # Live extension, NOT an Excel replica: override of rsi_long/vrp_long",
        "  # ONLY for the volatility leg in majority mode. params_per_quad",
        "  # stays faithful to the midterm workbook (gate and .xlsb check intact).",
        f"  # Recalibrated with 'calibrate-longvol' on {today} (frequency {freq_label}).",
        "  # Delete or comment out the block to go back to the workbook thresholds.",
        "  longvol_per_quad:",
    ]
    for q in sorted(overrides):
        parts = ", ".join(f"{k}: {v:g}" for k, v in overrides[q].items())
        lines.append(f"    {q}: {{{parts}}}")
    return "\n".join(lines)


def yaml_block_market(overrides: dict[int, dict[str, float]],
                      freq_label: str, today: str) -> str:
    """The `market_per_quad` block as text, with the explanations."""
    lines = [
        "  # >>> MARKET-LEG EXIT THRESHOLDS PER QUADRANT <<<",
        "  # Live extension, NOT an Excel replica: the sell SPX drops the fixed",
        "  # zeros (rsi<0 / vrp<0) and uses the regime quantile. Applies in",
        "  # majority AND in gate ('calibrated gate': Excel entry unchanged,",
        "  # exit anchored to the position, TDEX branch = deep fear).",
        "  # Entry and params_per_quad do NOT change (.xlsb check intact).",
        f"  # Recalibrated with 'calibrate-market' on {today} (quantile {freq_label}).",
        "  # Delete or comment out the block to go back to the Excel zeros.",
        "  market_per_quad:",
    ]
    for q in sorted(overrides):
        parts = ", ".join(f"{k}: {v:g}" for k, v in overrides[q].items())
        lines.append(f"    {q}: {{{parts}}}")
    return "\n".join(lines)
