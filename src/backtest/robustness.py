"""ROBUSTNESS battery for the strategy — "does this number hold up?"

The backtest says HOW MUCH it returned. This module says WHETHER it can be
believed. They are two different questions: the first is answered with a
curve, the second with a distribution. Everything here serves to turn a
point number ("4.46x, Calmar 0.22") into a statement with error bars
("4.46x, but 90% of the plausible histories lie between X and Y, and pure
chance gets there Z% of the time").

The eight families of tests, and WHY each one is here
-----------------------------------------------------
1. COSTS AND LATENCY (`costs_latency`)
   The existing backtest is zero-cost with T+1 execution. With 200-400
   entries in 21 years costs are not a detail, and latency is the best
   single fragility test: if performance dies going from T+1 to T+2, the
   signal lives inside one day of noise and is not an edge. It also
   includes LEVERAGE: the real asset is TQQQ (3x), not SPX, and a -40%
   drawdown at 3x leverage is not -40%.

2. ANCHORED MULTI-FOLD WALK-FORWARD (`walk_forward`)
   The existing 70/30 split gives ONE out-of-sample number, which is itself
   a lottery ticket. Here the training window widens and the calibration
   is redone at every fold, with thresholds estimated ONLY on the past: the
   result is the curve you would really have obtained by recalibrating
   periodically, plus a distribution of out-of-sample results instead of
   a single one.

3. PBO — Probability of Backtest Overfitting (`pbo`)
   CSCV method (Bailey, Borwein, López de Prado, Zhu 2014). If you try 500
   combinations and take the best, its result is the MAXIMUM of 500 draws:
   inflated by construction. PBO directly measures the probability that
   the selection procedure produces a configuration that, out of sample,
   sits below the median of the others. Above 50% the procedure does not
   select: it picks at random, with extra steps.

4. DEFLATED SHARPE RATIO (`deflated_sharpe`)
   Same disease, complementary measure: it discounts the winner's Sharpe
   for the number of trials, for the dispersion of the Sharpes tried and
   for the skewness/kurtosis of the returns. It returns the probability
   that the true Sharpe is > 0, and above all the "Sharpe expected by pure
   chance" with that number of trials: if the winner is below it, it is
   noise.

5. BLOCK BOOTSTRAP (`bootstrap`)
   History is ONE realisation. Resampling in blocks (which preserve
   autocorrelation and volatility clustering) yields thousands of
   plausible histories and therefore confidence intervals on multiple,
   CAGR, Sharpe, Calmar — and on the DIFFERENCE versus buy & hold, which is
   the real question: "does it beat the market" or "we don't know"?

6. PERMUTATION TEST (`permutation`)
   The right null hypothesis for a timing strategy is not "zero return":
   it is "being in 60% of the time in a rising market". The position
   series is rotated (preserving EXACTLY the number of trades, durations
   and time in the market) and only the alignment with the returns is
   destroyed. The resulting p-value answers: does the timing add anything
   to "being there"?

7. STABILITY ACROSS SUB-PERIODS (`subperiods`)
   A number over 21 years can come from a single episode. Here: table per
   year, rolling 3 years, return per quadrant, the big stress episodes one
   by one, and two concentration measures — how much of the advantage
   comes from the 5 best months, and what remains removing one year at a
   time (jackknife). If removing 2008 the edge disappears, the strategy is
   a single number, not a method.

8. SENSITIVITY TO THE ENGINE PARAMETERS (`parameter_sensitivity`)
   The calibrated thresholds are tested by the walk-forward; here the
   STRUCTURAL parameters are touched (RSI period, EMA, vote threshold,
   moving-average windows, TDEX thresholds). It serves two purposes:
   understanding whether we are on a plateau or on a razor's edge, and
   closing an open risk — the 32 midterm parameters were transcribed from
   a screenshot and not yet verified against the .xlsb: if the result is
   flat around the current values, a transcription error is harmless; if
   it jumps, it is fatal.

Common conventions
------------------
* All metrics use the real SPX returns (closes) as a proxy for the timing
  of the market leg, as the backtest already does. The proxy is declared,
  not hidden: `leverage` exists precisely to estimate the effect of the
  real asset.
* `pos` is the end-of-day position (True = in). Execution happens
  `latency` days later (default 1 = the bot decides EOD and enters the
  following day).
* Every function returns serialisable dicts/DataFrames: the report lays
  them out without recomputing anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import pandas as pd

from ..vix.quad_params import QuadParams
from ..vix.signal_generator import SignalConfig, generate_signals

DAYS_PER_YEAR = 252.0
EULER_GAMMA = 0.5772156649015329

LEG_COL = {"rsi": "rsi_smooth", "vrp": "vrp_smooth", "tdex": "tdex_smooth",
           "rsi_buy": "rsi_smooth", "vrp_buy": "vrp_smooth"}


# ====================================================================== #
#  Evaluation engine: costs, latency, leverage                           #
# ====================================================================== #
@dataclass(frozen=True)
class Costs:
    """Frictions of real execution.

    bps_side    : cost per SIDE in basis points (commissions + spread +
                  slippage). 5 bps = 0.05% for every entry and every exit.
    latency     : days between the signal (EOD) and execution. 1 = enter
                  at the next close (the bot's behaviour).
    leverage    : multiplier of the returns, rebalanced daily.
                  1 = SPX proxy; 3 = TQQQ/UVIX profile.
    fin_annual  : annual financing cost of the leveraged part, in %
                  (applied to (leverage-1) when in position).
    """
    bps_side: float = 0.0
    latency: int = 1
    leverage: float = 1.0
    fin_annual: float = 0.0

    @property
    def label(self) -> str:
        s = f"{self.bps_side:g}bps·T+{self.latency}"
        if self.leverage != 1.0:
            s += f"·lev{self.leverage:g}"
        return s


def strategy_returns(pos: pd.Series, ret: pd.Series,
                     costs: Costs = Costs()) -> pd.Series:
    """NET daily returns of the strategy.

    The exposure on day t is the position of t-latency (the signal cannot be
    executed at the instant it is born). Costs are paid on the day the
    exposure CHANGES, proportionally to the turnover: entering and exiting
    at 3x leverage costs 3 times as much.
    """
    exp = pos.astype(float).shift(costs.latency).fillna(0.0) * costs.leverage
    gross = exp * ret
    turnover = exp.diff().abs().fillna(exp.abs())
    friction = turnover * (costs.bps_side / 10_000.0)
    fin = np.where(exp > 1.0, (exp - 1.0) * costs.fin_annual / 100.0 / DAYS_PER_YEAR, 0.0)
    return (gross - friction - fin).rename("strategy")


def metrics(r: pd.Series, pos: pd.Series | None = None,
            label: str = "") -> dict:
    """The statistics that matter, on a series of daily returns."""
    r = r.dropna()
    if len(r) == 0:
        return {"label": label, "days": 0}
    eq = (1.0 + r).cumprod()
    years = len(r) / DAYS_PER_YEAR
    mult = float(eq.iloc[-1])
    cagr = mult ** (1.0 / max(years, 1e-9)) - 1.0 if mult > 0 else -1.0
    dd_series = eq / eq.cummax() - 1.0
    dd = float(dd_series.min())
    sd = float(r.std())
    sharpe = float(r.mean()) / sd * math.sqrt(DAYS_PER_YEAR) if sd > 0 else 0.0
    neg = r[r < 0]
    sd_down = float(neg.std()) if len(neg) > 1 else 0.0
    sortino = float(r.mean()) / sd_down * math.sqrt(DAYS_PER_YEAR) if sd_down > 0 else 0.0
    out = {
        "label": label,
        "days": int(len(r)),
        "years": round(years, 1),
        "multiple": round(mult, 2),
        "cagr_pct": round(cagr * 100, 2),
        "vol_pct": round(sd * math.sqrt(DAYS_PER_YEAR) * 100, 1),
        "max_dd_pct": round(dd * 100, 1),
        "calmar": round(cagr / abs(dd), 2) if dd < 0 else None,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "avg_dd_pct": round(float(dd_series.mean()) * 100, 1),
        "worst_day_pct": round(float(r.min()) * 100, 1),
        "max_recovery_days": _days_underwater(eq),
    }
    if pos is not None:
        p = pos.reindex(r.index).astype(bool)
        entries = int((p.astype(int).diff() == 1).sum())
        out |= {
            "time_in_market_pct": round(float(p.mean()) * 100, 1),
            "entries": entries,
            "avg_duration_sessions": round(float(p.sum()) / entries, 1) if entries else None,
        }
    return out


def _days_underwater(eq: pd.Series) -> int:
    """The longest period spent below the previous high: the metric a saver
    really feels (how long 'at a loss')."""
    peak = eq.cummax()
    under = eq < peak * (1 - 1e-12)
    if not under.any():
        return 0
    grp = (~under).cumsum()
    return int(under.groupby(grp).sum().max())


# ====================================================================== #
#  Main class                                                            #
# ====================================================================== #
class Robustness:
    """Test battery on ALREADY computed indicators.

    `ind` is the output of `LongVolCalibrator.prepare()` (or of the
    StatefulVix warm-up): it contains quad, spx, vix and the smoothed
    indicators. The indicators do NOT depend on the voting thresholds, so
    they are computed once and then each configuration costs one
    vectorised pass.
    """

    def __init__(self, ind: pd.DataFrame, params_by_quad: dict[int, QuadParams],
                 cfg: SignalConfig, seed: int = 42) -> None:
        self.ind = ind
        self.pbq = params_by_quad
        self.cfg = cfg
        self.seed = int(seed)
        self.ret = ind["spx"].pct_change().fillna(0.0)
        self.quad = ind["quad"].astype(int)

    # ------------------------------------------------------------------ #
    def position(self, cfg: SignalConfig | None = None, **override) -> pd.Series:
        """Position of the market leg for a configuration."""
        c = replace(cfg or self.cfg, **override) if override else (cfg or self.cfg)
        sig = generate_signals(self.ind, self.pbq, c)
        return sig["pos_spx"].astype(bool)

    def benchmark(self, costs: Costs = Costs()) -> pd.Series:
        """Buy & hold, with the same leverage and the same financing: the
        honest comparison for a leveraged strategy is the leveraged market."""
        inside = pd.Series(True, index=self.ret.index)
        return strategy_returns(inside, self.ret, replace(costs, bps_side=0.0))

    # ================================================================== #
    #  1. COSTS AND LATENCY                                              #
    # ================================================================== #
    def costs_latency(self, configs: dict[str, SignalConfig],
                      bps: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0),
                      latencies: tuple[int, ...] = (1, 2, 3),
                      leverages: tuple[float, ...] = (1.0, 3.0),
                      fin_annual: float = 5.0) -> pd.DataFrame:
        """Costs × latency × leverage grid for every configuration.

        How to read it: along the row of increasing costs what matters is
        the SLOPE, not the level (how much CAGR each bps eats: proportional
        to the number of trades). In the latency column what matters is the
        COLLAPSE: a healthy strategy loses little going to T+2, one that
        lives on noise is halved.
        """
        rows = []
        for name, c in configs.items():
            pos = self.position(c)
            for lev in leverages:
                for lat in latencies:
                    for b in bps:
                        co = Costs(bps_side=b, latency=lat, leverage=lev,
                                   fin_annual=fin_annual if lev > 1 else 0.0)
                        m = metrics(strategy_returns(pos, self.ret, co), pos)
                        rows.append({"config": name, "leverage": lev, "latency": lat,
                                     "bps": b, **{k: v for k, v in m.items()
                                                  if k != "label"}})
        df = pd.DataFrame(rows)
        # reference: the market, under the same conditions
        for lev in leverages:
            co = Costs(bps_side=0.0, latency=1, leverage=lev,
                       fin_annual=fin_annual if lev > 1 else 0.0)
            m = metrics(self.benchmark(co))
            df = pd.concat([df, pd.DataFrame([{"config": "buy&hold", "leverage": lev,
                                               "latency": 1, "bps": 0.0,
                                               **{k: v for k, v in m.items()
                                                  if k != "label"}}])],
                           ignore_index=True)
        return df

    # ================================================================== #
    #  2. ANCHORED MULTI-FOLD WALK-FORWARD                               #
    # ================================================================== #
    def walk_forward(self, n_fold: int = 6, n_samples: int = 300,
                     objective: str = "calmar", quads: list[int] | None = None,
                     q_min: float = 1.0, q_max: float = 30.0,
                     include_entries: bool = False, buy_min: float = 10.0,
                     buy_max: float = 70.0, min_train_years: float = 5.0,
                     costs: Costs = Costs(bps_side=5.0)) -> dict:
        """ANCHORED walk-forward: the training window always starts from the
        beginning and widens; the test is the next, unseen block.

        For each fold: `n_samples` combinations of exit quantiles (per
        quadrant) are sampled, with the quantiles estimated ONLY on the
        training data; the best one by `objective` in training is taken; it
        is evaluated out of sample. Then the out-of-sample blocks are
        stitched into a single curve: that is the performance the
        calibration procedure would really have produced.

        The two references are the BASE configuration (without overrides,
        i.e. the one in params.yaml today) and buy & hold, evaluated on the
        same out-of-sample blocks.
        """
        quads = quads or [1, 2, 3, 4]
        rng = np.random.default_rng(self.seed)
        idx = self.ind.index
        n_min = int(min_train_years * DAYS_PER_YEAR)
        if len(idx) <= n_min + n_fold * 60:
            raise ValueError("history too short for this number of folds")
        # evenly spaced cuts: n_fold consecutive test blocks
        cuts = np.linspace(n_min, len(idx), n_fold + 1).astype(int)
        key = {"calmar": "calmar", "return": "cagr_pct",
               "sharpe": "sharpe"}.get(objective, "calmar")

        legs = ["rsi", "vrp"] + (["rsi_buy", "vrp_buy"] if include_entries else [])
        ranges = {g: ((buy_min, buy_max) if g.endswith("_buy") else (q_min, q_max))
                  for g in legs}
        pos_base = self.position()
        r_base_tot = strategy_returns(pos_base, self.ret, costs)

        fold_out, oos_cal, oos_base, oos_bh = [], [], [], []
        for k in range(n_fold):
            i_tr_end, i_te_end = cuts[k], cuts[k + 1]
            m_tr = np.zeros(len(idx), bool); m_tr[:i_tr_end] = True
            m_te = np.zeros(len(idx), bool); m_te[i_tr_end:i_te_end] = True
            # per-quadrant distributions in training ONLY
            dist, ok = {}, True
            for q in quads:
                mq = (self.quad.to_numpy() == q) & m_tr
                for g in legs:
                    v = self.ind.loc[mq, LEG_COL[g]].dropna().to_numpy()
                    if len(v) < 30:
                        ok = False
                    dist[(q, g)] = v
            if not ok:
                fold_out.append({"fold": k + 1, "skip": "insufficient train data",
                                 "train_to": str(idx[i_tr_end - 1].date())})
                continue

            def evaluate(qtl: dict, mask) -> tuple[dict, pd.Series]:
                ov: dict[int, dict[str, float]] = {}
                for q in quads:
                    for g in legs:
                        thr = float(np.quantile(dist[(q, g)], qtl[(q, g)] / 100.0))
                        k_ = g if g.endswith("_buy") else f"{g}_exit"
                        ov.setdefault(q, {})[k_] = round(thr, 1)
                p = self.position(replace(self.cfg, market_per_quad=ov))
                r = strategy_returns(p, self.ret, costs)
                return ov, r[mask]

            best, best_val, best_ov = None, -1e18, None
            for _ in range(int(n_samples)):
                qtl = {(q, g): float(rng.uniform(*ranges[g]))
                       for q in quads for g in legs}
                ov, r_tr = evaluate(qtl, m_tr)
                m = metrics(r_tr)
                v = m.get(key)
                if v is not None and v > best_val:
                    best_val, best, best_ov = v, qtl, ov
            # apply the winner out of sample
            p_win = self.position(replace(self.cfg, market_per_quad=best_ov))
            r_win = strategy_returns(p_win, self.ret, costs)
            m_tr_win = metrics(r_win[m_tr], p_win[m_tr])
            m_te_win = metrics(r_win[m_te], p_win[m_te])
            m_te_base = metrics(r_base_tot[m_te], pos_base[m_te])
            m_te_bh = metrics(self.benchmark(costs)[m_te])
            fold_out.append({
                "fold": k + 1,
                "train_from": str(idx[0].date()), "train_to": str(idx[i_tr_end - 1].date()),
                "test_from": str(idx[i_tr_end].date()), "test_to": str(idx[i_te_end - 1].date()),
                "thresholds": best_ov,
                "train_calibrated": m_tr_win, "test_calibrated": m_te_win,
                "test_base": m_te_base, "test_buyhold": m_te_bh,
                "degradation": (round(m_te_win[key] - m_tr_win[key], 2)
                                if m_tr_win.get(key) is not None
                                and m_te_win.get(key) is not None else None),
                "beats_base": (m_te_win.get(key) or -9e9) > (m_te_base.get(key) or -9e9),
            })
            oos_cal.append(r_win[m_te]); oos_base.append(r_base_tot[m_te])
            oos_bh.append(self.benchmark(costs)[m_te])

        stitched = {}
        if oos_cal:
            stitched = {
                "calibrated": metrics(pd.concat(oos_cal), label="WF calibrated"),
                "base": metrics(pd.concat(oos_base), label="base (params.yaml)"),
                "buy_hold": metrics(pd.concat(oos_bh), label="buy & hold"),
            }
        valid = [f for f in fold_out if "skip" not in f]
        return {
            "objective": key, "n_fold": n_fold, "n_samples": n_samples,
            "costs": costs.label, "fold": fold_out,
            "oos_stitched": stitched,
            "folds_beating_base": sum(1 for f in valid if f["beats_base"]),
            "valid_folds": len(valid),
            "median_degradation": (round(float(np.median([f["degradation"] for f in valid
                                                          if f["degradation"] is not None])), 2)
                                   if valid else None),
        }

    # ================================================================== #
    #  3. PBO — Probability of Backtest Overfitting (CSCV)               #
    # ================================================================== #
    def pbo(self, n_samples: int = 300, s_blocks: int = 10,
            objective: str = "sharpe", quads: list[int] | None = None,
            q_min: float = 1.0, q_max: float = 30.0, include_entries: bool = False,
            buy_min: float = 10.0, buy_max: float = 70.0,
            costs: Costs = Costs(bps_side=5.0),
            include_base: bool = True) -> dict:
        """CSCV: how likely is it that SELECTING the best produces a loser
        out of sample?

        N configurations are generated, the matrix of daily returns (T×N)
        is recorded, time is split into S blocks and for each of the
        C(S, S/2) combinations half of the blocks are used as "in sample"
        and half as "out". For each combination: take the in-sample winner
        and look at its RANK out of sample. PBO = fraction of combinations
        in which the winner ends up below the median. 0% = selection works;
        >50% = selection is a disguised lottery.

        Methodological note: the candidates' thresholds are quantiles
        estimated over the whole history (this is the CSCV design: it tests
        the SELECTION procedure, not the absence of lookahead — that is
        tested by the walk-forward in point 2).
        """
        if s_blocks % 2:
            s_blocks += 1
        quads = quads or [1, 2, 3, 4]
        rng = np.random.default_rng(self.seed + 1)
        legs = ["rsi", "vrp"] + (["rsi_buy", "vrp_buy"] if include_entries else [])
        ranges = {g: ((buy_min, buy_max) if g.endswith("_buy") else (q_min, q_max))
                  for g in legs}
        dist = {(q, g): self.ind.loc[self.quad == q, LEG_COL[g]].dropna().to_numpy()
                for q in quads for g in legs}

        R, labels = [], []
        for _ in range(int(n_samples)):
            ov: dict[int, dict[str, float]] = {}
            for q in quads:
                for g in legs:
                    u = float(rng.uniform(*ranges[g]))
                    thr = float(np.quantile(dist[(q, g)], u / 100.0))
                    k_ = g if g.endswith("_buy") else f"{g}_exit"
                    ov.setdefault(q, {})[k_] = round(thr, 1)
            p = self.position(replace(self.cfg, market_per_quad=ov))
            R.append(strategy_returns(p, self.ret, costs).to_numpy())
            labels.append(ov)
        # the BASE configuration can join as a candidate: it shows whether
        # the selection at least manages NOT to make it worse. With
        # include_base=False the PBO of the calibration procedure alone is
        # measured (stricter: it is forced to pick a calibrated configuration).
        if include_base:
            p0 = self.position()
            R.append(strategy_returns(p0, self.ret, costs).to_numpy())
            labels.append({})
        M = np.column_stack(R)                       # T × (N+1)
        T, N = M.shape
        edges = np.linspace(0, T, s_blocks + 1).astype(int)
        blocks = [np.arange(edges[i], edges[i + 1]) for i in range(s_blocks)]

        def perf(sub: np.ndarray) -> np.ndarray:
            if objective == "return":
                return np.log1p(sub).sum(axis=0)
            mu, sd = sub.mean(axis=0), sub.std(axis=0)
            return np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), 0.0)

        half = s_blocks // 2
        lam, below, ranks, base_chosen = [], 0, [], 0
        combos = list(combinations(range(s_blocks), half))
        for c in combos:
            i_is = np.concatenate([blocks[i] for i in c])
            i_oos = np.concatenate([blocks[i] for i in range(s_blocks) if i not in c])
            p_is, p_oos = perf(M[i_is]), perf(M[i_oos])
            n_star = int(np.argmax(p_is))
            if include_base and n_star == N - 1:
                base_chosen += 1
            # relative out-of-sample rank (1 = best, 0 = worst)
            w = float((p_oos < p_oos[n_star]).sum() + 0.5) / N
            w = min(max(w, 1.0 / (2 * N)), 1.0 - 1.0 / (2 * N))
            ranks.append(w)
            lam.append(math.log(w / (1.0 - w)))
            below += int(w < 0.5)
        return {
            "pbo_pct": round(100.0 * below / len(combos), 1),
            "n_candidates": N, "n_blocks": s_blocks, "n_combinations": len(combos),
            "objective": objective, "costs": costs.label,
            "include_base": bool(include_base),
            "median_oos_rank": round(float(np.median(ranks)), 3),
            "mean_logit": round(float(np.mean(lam)), 3),
            "base_chosen_pct": round(100.0 * base_chosen / len(combos), 1),
            "_matrix": M, "_labels": labels,
        }

    # ================================================================== #
    #  4. DEFLATED SHARPE RATIO                                          #
    # ================================================================== #
    @staticmethod
    def deflated_sharpe(r_winner: pd.Series, sharpes_tried: np.ndarray) -> dict:
        """The winner's Sharpe, discounted for the number of trials.

        Returns:
          expected_chance_sharpe : the Sharpe that the BEST of N strategies
                                   with no edge at all would get anyway. It
                                   is the bar to beat, not zero.
          dsr_pct                : probability that the true Sharpe is > 0,
                                   after the discount. Below ~95% nothing can
                                   be claimed; below 50% the winner is noise.
        """
        r = pd.Series(r_winner).dropna()
        T = len(r)
        if T < 30:
            return {"error": "series too short"}
        mu, sd = float(r.mean()), float(r.std(ddof=1))
        sr = mu / sd if sd > 0 else 0.0                     # per observation
        g1 = float(pd.Series(r).skew())
        g2 = float(pd.Series(r).kurt()) + 3.0               # non-excess kurtosis
        s = np.asarray([x for x in np.ravel(sharpes_tried) if np.isfinite(x)], float)
        N = max(len(s), 2)
        var_s = float(np.var(s, ddof=1)) if len(s) > 1 else 0.0
        # expected Sharpe of the maximum of N draws under the null hypothesis
        z1 = _norm_ppf(1.0 - 1.0 / N)
        z2 = _norm_ppf(1.0 - 1.0 / (N * math.e))
        sr0 = math.sqrt(max(var_s, 0.0)) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
        den = math.sqrt(max(1.0 - g1 * sr + (g2 - 1.0) / 4.0 * sr * sr, 1e-12))
        z = (sr - sr0) * math.sqrt(T - 1.0) / den
        return {
            "n_trials": N, "observations": T,
            "annual_sharpe": round(sr * math.sqrt(DAYS_PER_YEAR), 2),
            "expected_chance_sharpe_annual": round(sr0 * math.sqrt(DAYS_PER_YEAR), 2),
            "trials_dispersion": round(math.sqrt(max(var_s, 0.0))
                                       * math.sqrt(DAYS_PER_YEAR), 2),
            "skewness": round(g1, 2), "kurtosis": round(g2, 1),
            "dsr_pct": round(100.0 * _norm_cdf(z), 1),
            "beats_chance": bool(sr > sr0),
        }

    # ================================================================== #
    #  5. BLOCK BOOTSTRAP                                                #
    # ================================================================== #
    def bootstrap(self, pos: pd.Series, n: int = 2000, block: int = 21,
                  costs: Costs = Costs(bps_side=5.0),
                  levels: tuple[float, ...] = (5.0, 50.0, 95.0)) -> dict:
        """Circular block bootstrap, PAIRED with the benchmark.

        Blocks of `block` consecutive days are resampled (default one
        trading month: it preserves autocorrelation and volatility
        clustering) and the history is rebuilt. Strategy and benchmark are
        resampled on the SAME blocks: so the difference between the two
        stays comparable and a confidence interval can be given on the
        ADVANTAGE, not only on the levels.

        Honest caveat: max drawdown depends on the order of the days, which
        the bootstrap reshuffles; the DD interval should be read as
        indicative ("what DD could I expect from a similar process"), not
        as an estimate of the historical DD.
        """
        r_s = strategy_returns(pos, self.ret, costs).to_numpy()
        r_b = self.benchmark(costs).to_numpy()
        T = len(r_s)
        rng = np.random.default_rng(self.seed + 2)
        n_blocks = int(np.ceil(T / block))
        samples = {k: [] for k in ("multiple", "cagr_pct", "sharpe", "max_dd_pct",
                                   "calmar", "extra_cagr_pct", "extra_sharpe")}
        for _ in range(int(n)):
            starts = rng.integers(0, T, n_blocks)
            idx = np.concatenate([(np.arange(a, a + block) % T) for a in starts])[:T]
            s, b = pd.Series(r_s[idx]), pd.Series(r_b[idx])
            ms, mb = metrics(s), metrics(b)
            for k in ("multiple", "cagr_pct", "sharpe", "max_dd_pct", "calmar"):
                samples[k].append(ms.get(k) if ms.get(k) is not None else np.nan)
            samples["extra_cagr_pct"].append((ms.get("cagr_pct") or 0) - (mb.get("cagr_pct") or 0))
            samples["extra_sharpe"].append((ms.get("sharpe") or 0) - (mb.get("sharpe") or 0))
        obs = metrics(pd.Series(r_s), pos)
        obs_b = metrics(pd.Series(r_b))
        out = {"n": int(n), "block_days": block, "costs": costs.label,
               "observed": obs, "benchmark": obs_b, "intervals": {}}
        for k, v in samples.items():
            a = np.asarray(v, float); a = a[np.isfinite(a)]
            if not len(a):
                continue
            out["intervals"][k] = {f"p{int(l)}": round(float(np.percentile(a, l)), 2)
                                   for l in levels}
        # probability of beating the market, in the simulated histories
        ec = np.asarray(samples["extra_cagr_pct"], float)
        es = np.asarray(samples["extra_sharpe"], float)
        out["prob_extra_cagr_positive_pct"] = round(100.0 * float((ec > 0).mean()), 1)
        out["prob_extra_sharpe_positive_pct"] = round(100.0 * float((es > 0).mean()), 1)
        return out

    # ================================================================== #
    #  6. PERMUTATION TEST                                               #
    # ================================================================== #
    def permutation(self, pos: pd.Series, n: int = 1000,
                    costs: Costs = Costs(bps_side=5.0),
                    mode: str = "rotation") -> dict:
        """The right null hypothesis: "same presence, random timing".

        mode = "rotation" : the position series is rotated cyclically by a
            random lag. It preserves EXACTLY the time in the market, the
            number of trades and the distribution of durations; it only
            destroys the alignment with the returns. It is the cleanest
            null for a timing strategy.
        mode = "blocks"   : the "in" blocks are relocated at random in the
            calendar (same durations, new positions). More aggressive on the
            calendar, less conservative on autocorrelation.

        p-value = share of random histories that do AT LEAST AS WELL as the
        strategy. p = 0.30 means: a die with the same market presence beats
        me 3 times out of 10 — there is no demonstrable timing.
        """
        r_obs = strategy_returns(pos, self.ret, costs)
        m_obs = metrics(r_obs, pos)
        a = pos.to_numpy().astype(bool)
        T = len(a)
        rng = np.random.default_rng(self.seed + 3)
        durations = _block_durations(a)
        nulls = {k: [] for k in ("cagr_pct", "sharpe", "calmar", "max_dd_pct")}
        for _ in range(int(n)):
            if mode == "blocks":
                b = _relocate(durations, T, rng)
            else:
                k = int(rng.integers(DAYS_PER_YEAR // 4, T - DAYS_PER_YEAR // 4))
                b = np.roll(a, k)
            p = pd.Series(b, index=pos.index)
            m = metrics(strategy_returns(p, self.ret, costs))
            for k2 in nulls:
                nulls[k2].append(m.get(k2) if m.get(k2) is not None else np.nan)
        out = {"n": int(n), "mode": mode, "costs": costs.label,
               "observed": m_obs, "null": {}, "p_value": {}}
        for k, v in nulls.items():
            arr = np.asarray(v, float); arr = arr[np.isfinite(arr)]
            if not len(arr):
                continue
            o = m_obs.get(k)
            out["null"][k] = {"p5": round(float(np.percentile(arr, 5)), 2),
                              "median": round(float(np.median(arr)), 2),
                              "p95": round(float(np.percentile(arr, 95)), 2)}
            if o is None:
                continue
            # "better" = higher value for all these metrics, including the
            # drawdown (which is negative: -17% is better than -34%)
            better = arr >= o
            out["p_value"][k] = round(float((1 + better.sum()) / (1 + len(arr))), 3)
        return out

    # ================================================================== #
    #  7. STABILITY ACROSS SUB-PERIODS                                   #
    # ================================================================== #
    def subperiods(self, pos: pd.Series, costs: Costs = Costs(bps_side=5.0),
                   crises: dict[str, tuple[str, str]] | None = None) -> dict:
        """Where the result comes from: per year, per regime, in the crises,
        and how concentrated it is in a few episodes."""
        r_s = strategy_returns(pos, self.ret, costs)
        r_b = self.benchmark(costs)
        years = []
        for year, g in r_s.groupby(r_s.index.year):
            gb = r_b.loc[g.index]
            ms, mb = metrics(g, pos.loc[g.index]), metrics(gb)
            years.append({"year": int(year),
                          "strategy_pct": ms["cagr_pct"] and round(
                              (float((1 + g).prod()) - 1) * 100, 1),
                          "market_pct": round((float((1 + gb).prod()) - 1) * 100, 1),
                          "strategy_dd_pct": ms["max_dd_pct"],
                          "market_dd_pct": mb["max_dd_pct"],
                          "in_market_pct": ms.get("time_in_market_pct")})
        for a in years:
            a["extra_pct"] = round((a["strategy_pct"] or 0) - a["market_pct"], 1)
        # rolling 3 years
        win = int(3 * DAYS_PER_YEAR)
        roll = []
        if len(r_s) > win:
            cs = np.log1p(r_s).rolling(win).sum()
            cb = np.log1p(r_b).rolling(win).sum()
            d = (cs - cb).dropna()
            roll = {"windows": int(len(d)),
                    "share_windows_above_market_pct": round(float((d > 0).mean()) * 100, 1),
                    "worst_extra_3y_pct": round(float(np.expm1(d.min())) * 100, 1),
                    "best_extra_3y_pct": round(float(np.expm1(d.max())) * 100, 1),
                    "median_extra_3y_pct": round(float(np.expm1(d.median())) * 100, 1)}
        # per quadrant
        per_quad = []
        for q in sorted(self.quad.unique()):
            m = self.quad == q
            if m.sum() < 60:
                continue
            ms = metrics(r_s[m], pos[m]); mb = metrics(r_b[m])
            per_quad.append({"quad": int(q), "days": int(m.sum()),
                             "strategy_cagr_pct": ms["cagr_pct"],
                             "market_cagr_pct": mb["cagr_pct"],
                             "in_market_pct": ms.get("time_in_market_pct"),
                             "sharpe": ms["sharpe"]})
        # crises
        crises = crises or {
            "2008 financial crisis": ("2007-10-09", "2009-03-09"),
            "2010 flash crash": ("2010-04-23", "2010-07-02"),
            "2011 US debt": ("2011-04-29", "2011-10-03"),
            "2015-16 China": ("2015-05-21", "2016-02-11"),
            "2018 Q4": ("2018-09-20", "2018-12-24"),
            "2020 Covid": ("2020-02-19", "2020-03-23"),
            "2022 inflation": ("2022-01-03", "2022-10-12"),
            "2025 tariffs": ("2025-02-19", "2025-04-08"),
        }
        ev = []
        for name, (d0, d1) in crises.items():
            try:
                s = r_s.loc[d0:d1]; b = r_b.loc[d0:d1]
            except KeyError:
                continue
            if len(s) < 5:
                continue
            ev.append({"event": name, "from": d0, "to": d1, "sessions": int(len(s)),
                       "strategy_pct": round((float((1 + s).prod()) - 1) * 100, 1),
                       "market_pct": round((float((1 + b).prod()) - 1) * 100, 1),
                       "in_market_pct": round(float(pos.loc[d0:d1].mean()) * 100, 1)})
        # concentration of the advantage: best months and annual jackknife
        ex_m = (np.log1p(r_s) - np.log1p(r_b)).resample("ME").sum()
        tot = float(ex_m.sum())
        top5 = float(ex_m.nlargest(5).sum())
        jack = []
        for year in sorted({d.year for d in r_s.index}):
            m = r_s.index.year != year
            ms, mb = metrics(r_s[m]), metrics(r_b[m])
            jack.append({"without_year": int(year),
                         "cagr_pct": ms["cagr_pct"], "calmar": ms["calmar"],
                         "extra_cagr_pct": round((ms["cagr_pct"] or 0)
                                                 - (mb["cagr_pct"] or 0), 2)})
        worst = min(jack, key=lambda x: x["extra_cagr_pct"]) if jack else None
        return {
            "costs": costs.label,
            "per_year": years, "rolling_3y": roll, "per_quadrant": per_quad,
            "crises": ev,
            "concentration": {
                "extra_total_log": round(tot, 3),
                "extra_top5_months_log": round(top5, 3),
                "extra_without_top5_log": round(tot - top5, 3),
                # share of the advantage due to the 5 best months. With a
                # negative total advantage the percentage is meaningless:
                # read the "without top5" row (how much worse it gets
                # removing them).
                "top5_months_share_pct": (round(100.0 * top5 / tot, 1)
                                          if tot > 1e-9 else None),
                "months_for_half_advantage": (
                    int((ex_m.nlargest(len(ex_m)).cumsum() < tot / 2).sum() + 1)
                    if tot > 1e-9 else None),
                "best_months": [{"month": str(i)[:7], "extra_log": round(float(v), 3)}
                                for i, v in ex_m.nlargest(5).items()],
                "worst_months": [{"month": str(i)[:7], "extra_log": round(float(v), 3)}
                                 for i, v in ex_m.nsmallest(3).items()],
            },
            "annual_jackknife": jack,
            "most_decisive_year": worst,
        }

    # ================================================================== #
    #  data coverage: when all the inputs really exist                    #
    # ================================================================== #
    def data_coverage(self) -> dict:
        """Since when each input really exists.

        It serves to unmask the worst case of all: a period in which the bot
        "dodged" a crash only because it was BLIND (missing data -> no
        signal -> flat). That non-trade is not merit: it is a gap."""
        out = {}
        for c in ("spx", "vix", "rsi_smooth", "vrp_smooth", "tdex_smooth"):
            if c not in self.ind.columns:
                continue
            s = self.ind[c]
            fv = s.first_valid_index()
            out[c] = {"first_data": str(fv.date()) if fv is not None else None,
                      "valid_days": int(s.notna().sum()),
                      "gaps_after_start": int(s.loc[fv:].isna().sum()) if fv is not None else None}
        firsts = [v["first_data"] for v in out.values() if v["first_data"]]
        out["history_fully_covered_from"] = max(firsts) if firsts else None
        return out

    def compare_windows(self, configs: dict[str, SignalConfig],
                        windows: dict[str, tuple[str, str]],
                        costs: Costs = Costs(bps_side=5.0)) -> pd.DataFrame:
        """Same configuration over different windows (e.g. the whole history
        vs only the period with TDEX present): if the numbers change nature,
        the result was an artefact of the period, not a property."""
        rows = []
        for name, c in configs.items():
            pos = self.position(c)
            r = strategy_returns(pos, self.ret, costs)
            b = self.benchmark(costs)
            for wn, (d0, d1) in windows.items():
                s = r.loc[d0:d1]
                if len(s) < 60:
                    continue
                m = metrics(s, pos.loc[d0:d1]); mb = metrics(b.loc[d0:d1])
                rows.append({"config": name, "window": wn,
                             **{k: v for k, v in m.items() if k != "label"},
                             "market_cagr_pct": mb["cagr_pct"],
                             "market_dd_pct": mb["max_dd_pct"],
                             "market_calmar": mb["calmar"]})
        return pd.DataFrame(rows)

    def rolling_start(self, pos: pd.Series, step_months: int = 6,
                      min_years: float = 5.0,
                      costs: Costs = Costs(bps_side=5.0)) -> dict:
        """Sensitivity to the START DATE: the same strategy evaluated from
        every possible start (every `step_months`) up to today. If the
        verdict changes depending on when you would have started, the
        historical result is chance, not an expectation."""
        r = strategy_returns(pos, self.ret, costs)
        b = self.benchmark(costs)
        end = r.index[-1]
        starts = pd.date_range(r.index[0], end - pd.DateOffset(years=int(min_years)),
                               freq=f"{step_months}MS")
        rows = []
        for d in starts:
            s = r.loc[d:]
            if len(s) < min_years * DAYS_PER_YEAR * 0.9:
                continue
            m, mb = metrics(s), metrics(b.loc[d:])
            rows.append({"from": str(d.date()), "cagr_pct": m["cagr_pct"],
                         "max_dd_pct": m["max_dd_pct"], "calmar": m["calmar"],
                         "sharpe": m["sharpe"],
                         "extra_cagr_pct": round((m["cagr_pct"] or 0)
                                                 - (mb["cagr_pct"] or 0), 2)})
        df = pd.DataFrame(rows)
        if df.empty:
            return {"rows": []}
        return {
            "rows": rows, "n_starts": len(df),
            "cagr": {"min": float(df.cagr_pct.min()), "median": float(df.cagr_pct.median()),
                     "max": float(df.cagr_pct.max())},
            "share_starts_above_market_pct": round(float((df.extra_cagr_pct > 0).mean()) * 100, 1),
            "share_starts_negative_cagr_pct": round(float((df.cagr_pct < 0).mean()) * 100, 1),
        }


# ====================================================================== #
#  8. SENSITIVITY TO THE STRUCTURAL PARAMETERS (needs engine recompute)  #
# ====================================================================== #
def parameter_sensitivity(build: "callable", p_base: dict,
                          perturbations: dict[str, list],
                          costs: Costs = Costs(bps_side=5.0),
                          method: str | None = None) -> pd.DataFrame:
    """Perturbs the STRUCTURAL parameters and re-measures everything.

    `build(p) -> (ind, params_by_quad, cfg)` must rebuild the indicators
    with the modified parameters (the regime depends on the RSI period,
    EMA, vote threshold and moving-average windows: the heavy run must be
    redone, ~1 s).

    `perturbations` is {"path.in.the.yaml": [values]}, e.g.
        {"regime.rsi_period": [19, 21, 23],
         "regime.ema_period": [14, 16, 18],
         "regime.vote_threshold": [4, 5, 6],
         "vix.tdex_gate_buy_spx": [-1.0, -0.5, 0.0]}

    Reading: what you are looking for is a PLATEAU. If moving a parameter
    by ±1 the CAGR moves little, the exact value is not critical (and a
    transcription error is not fatal). If it jumps, that parameter is the
    fragile point of the strategy and must be verified at the source.
    """
    import copy
    rows = []
    for path, values in perturbations.items():
        for v in values:
            p = copy.deepcopy(p_base)
            node = p
            *branches, leaf = path.split(".")
            for r_ in branches:
                node = node[r_]
            original = node.get(leaf)
            node[leaf] = v
            try:
                ind, pbq, cfg = build(p)
                if method:
                    cfg = replace(cfg, method=method)
                rb = Robustness(ind, pbq, cfg)
                pos = rb.position()
                m = metrics(strategy_returns(pos, rb.ret, costs), pos)
                mb = metrics(rb.benchmark(costs))
                rows.append({"parameter": path, "value": v,
                             "current": v == original,
                             **{k: x for k, x in m.items() if k != "label"},
                             "extra_cagr_pct": round((m["cagr_pct"] or 0)
                                                     - (mb["cagr_pct"] or 0), 2)})
            except Exception as exc:                        # pragma: no cover
                rows.append({"parameter": path, "value": v,
                             "error": str(exc)[:120]})
    df = pd.DataFrame(rows)
    if df.empty or "cagr_pct" not in df.columns:
        return df
    # range = how much that parameter matters (spread of the CAGR over the values tried)
    amp = df.groupby("parameter")["cagr_pct"].agg(lambda s: float(s.max() - s.min()))
    df["cagr_range_pct"] = df["parameter"].map(amp).round(2)
    return df.sort_values(["cagr_range_pct", "parameter", "value"],
                          ascending=[False, True, True])


# ====================================================================== #
#  Utilities                                                             #
# ====================================================================== #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(q: float) -> float:
    """Inverse of the standard normal (Acklam, error < 1.15e-9)."""
    q = min(max(q, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if q < pl:
        r = math.sqrt(-2 * math.log(q))
        return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
               ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    if q > ph:
        r = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
                ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    r = q - 0.5
    s = r * r
    return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / \
           (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)


def _block_durations(a: np.ndarray) -> list[int]:
    """Durations of the 'in' stretches (for the relocated-blocks null)."""
    out, n = [], 0
    for x in a:
        if x:
            n += 1
        elif n:
            out.append(n); n = 0
    if n:
        out.append(n)
    return out


def _relocate(durations: list[int], T: int, rng) -> np.ndarray:
    """Relocates blocks of the same durations at random on an empty calendar."""
    b = np.zeros(T, bool)
    for d in sorted(durations, reverse=True):
        for _ in range(40):
            s = int(rng.integers(0, max(T - d, 1)))
            if not b[max(s - 1, 0):s + d + 1].any():
                b[s:s + d] = True
                break
    return b
