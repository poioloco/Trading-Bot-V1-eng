"""Consensus model + portfolio simulation.

CONSENSUS: every signal votes long(+1)/short(-1)/neutral(0). If the share of
agreeing votes exceeds the threshold, the day's target is the risk-on asset
(long consensus) or the risk-off asset (short consensus); otherwise CASH
(wait).

SIMULATION: a single capital pool fed by the user's deposits. Every day the
bot is 100% invested in the target asset (or in cash). When the target
changes, it closes the position and opens a new one: that is a trade, with
P&L in euro and in %, MFE/MAE excursions, regime and indicators at entry.
The series of trades gives the portfolio curve (net of deposits).

Designed for N signals: today there are two legs, more can be added later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

ON, OFF, CASH = "risk_on", "risk_off", "cash"


# ---- votes and consensus ------------------------------------------------ #
def leg_votes(timeline: pd.DataFrame, legs: dict | None = None,
              tdex_threshold: float = 4.0, vol_inverted: bool = True,
              market_vote: str = "binary") -> pd.DataFrame:
    """+1/-1 vote (0 = abstain) of each ACTIVE leg.
    Market (market_vote):
      "binary" (legacy) = in position +1, out -1 (no abstention: the
        absence of a signal counts as a full risk-off vote).
      "ternary" = in position +1, ACTIVE exit (Excel column U:
        rsi<0|vrp<0|tdex_off) -1, otherwise 0 (abstains). Faithful to the
        Excel, where "Do Nothing" is not a directional vote: on ambiguous
        days the market leg stays silent and the others decide.
    Volatility: 'long volatility' (stress) = -1 (risk-off) if inverted.
    TDEX: smoothed z <= threshold = -1 (risk-off), > threshold = +1
          (risk-on), NaN (no data, e.g. pre-2013) = 0 (abstains)."""
    legs = legs or {"market": True, "volatility": True, "tdex": False}
    v = pd.DataFrame(index=timeline.index)
    if legs.get("market", True):
        in_pos = timeline["pos_market"] > 0
        if market_vote == "ternary" and "spx_sell_state" in timeline.columns:
            exit_ = timeline["spx_sell_state"].to_numpy()
            v["market"] = np.where(in_pos, 1, np.where(exit_, -1, 0))
        else:
            v["market"] = np.where(in_pos, 1, -1)
    if legs.get("volatility", True):
        sign = -1 if vol_inverted else 1
        v["volatility"] = np.where(timeline["pos_volatility"] > 0, sign, -sign)
    if legs.get("tdex", False) and "tdex_smooth" in timeline.columns:
        ts = timeline["tdex_smooth"]
        v["tdex"] = np.where(ts.isna(), 0, np.where(ts <= tdex_threshold, -1, 1))
    return v


def consensus_target(votes: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """RAW signal of the day: ON (long consensus), OFF (short consensus) or
    CASH. The threshold applies to the votes CAST only (abstentions do not
    count): with 3 active legs 0.5 = majority; with 2, unanimity."""
    if votes.shape[1] == 0:
        return pd.Series(CASH, index=votes.index, name="signal")
    arr = votes.to_numpy()
    longs = (arr > 0).sum(axis=1)
    shorts = (arr < 0).sum(axis=1)
    n = (arr != 0).sum(axis=1)                 # votes cast (no abstentions)
    n_safe = np.where(n > 0, n, 1)
    out = np.full(len(arr), CASH, dtype=object)
    out[longs / n_safe > threshold] = ON
    out[shorts / n_safe > threshold] = OFF
    out[n == 0] = CASH
    return pd.Series(out, index=votes.index, name="signal")


def apply_hysteresis(raw: pd.Series) -> pd.Series:
    """Hysteresis on the position: you enter with a consensus (ON/OFF); once
    in, you do NOT exit on days without consensus, you hold the position
    until the OPPOSITE consensus arrives. Cash only remains while flat."""
    state = CASH
    out = []
    for r in raw.to_numpy():
        if r in (ON, OFF):
            state = r
        out.append(state)        # raw CASH = keep the current state
    return pd.Series(out, index=raw.index, name="target")


# ---- result ------------------------------------------------------------- #
@dataclass
class Trade:
    asset: str
    side: str                 # risk_on | risk_off
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    entry_price: float
    exit_price: float | None
    capital: float            # euro invested (including deposits during the trade)
    proceeds: float | None
    pnl: float | None
    return_pct: float | None
    mfe_pct: float
    mae_pct: float
    days: int | None
    regime_in: str
    rsi_in: float
    vrp_in: float
    is_open: bool = False


@dataclass
class PortfolioResult:
    trades: pd.DataFrame
    equity: pd.DataFrame      # value, cumulative deposits, twr
    target: pd.Series
    stats: dict
    notes: list[str] = field(default_factory=list)


def load_contributions(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["amount_eur"].astype(float).sort_index()


# ---- simulator ---------------------------------------------------------- #
class PortfolioSimulator:
    def __init__(self, asset_on: str, asset_off: str, threshold: float = 0.5,
                 vol_inverted: bool = True, hysteresis: bool = True,
                 legs: dict | None = None, tdex_threshold: float = 4.0,
                 market_vote: str = "binary") -> None:
        self.asset_on = asset_on
        self.asset_off = asset_off
        self.threshold = threshold
        self.vol_inverted = vol_inverted
        self.hysteresis = hysteresis
        self.legs = legs or {"market": True, "volatility": True, "tdex": False}
        self.tdex_threshold = tdex_threshold
        self.market_vote = market_vote

    def run(self, timeline: pd.DataFrame, price_on: pd.Series,
            price_off: pd.Series, contributions: pd.Series) -> PortfolioResult:
        votes = leg_votes(timeline, self.legs, self.tdex_threshold,
                          self.vol_inverted, self.market_vote)
        raw = consensus_target(votes, self.threshold)
        target = apply_hysteresis(raw) if self.hysteresis else raw

        px = pd.DataFrame({ON: price_on, OFF: price_off}).reindex(timeline.index).ffill()
        dates = timeline.index[px.notna().all(axis=1)]
        target = target.reindex(dates)
        contr = contributions.reindex(dates).fillna(0.0)

        cash = 0.0
        side = None                 # ON | OFF | None
        units = 0.0
        cur = None                  # open Trade
        peak = trough = np.nan
        trades: list[Trade] = []
        eq_rows = []
        dep_cum = 0.0
        prev_val = np.nan

        for d in dates:
            price_today = {ON: float(px.at[d, ON]), OFF: float(px.at[d, OFF])}
            dep = float(contr.at[d])
            dep_cum += dep
            # 1) deposit: into cash, or invested in the current asset
            if dep > 0 and side is not None:
                units += dep / price_today[side]
                cur.capital += dep
            elif dep > 0:
                cash += dep

            tgt = target.at[d]
            tgt_side = None if tgt == CASH else (ON if tgt == ON else OFF)

            if tgt_side != side:                       # target change -> trade
                if side is not None:                   # close
                    pxe = price_today[side]
                    proceeds = units * pxe
                    cur.exit_date = d
                    cur.exit_price = pxe
                    cur.proceeds = proceeds
                    cur.pnl = proceeds - cur.capital
                    cur.return_pct = proceeds / cur.capital - 1.0 if cur.capital else 0.0
                    cur.mfe_pct = peak / cur.entry_price - 1.0 if cur.entry_price else 0.0
                    cur.mae_pct = trough / cur.entry_price - 1.0 if cur.entry_price else 0.0
                    cur.days = int((d - cur.entry_date).days)
                    trades.append(cur)
                    cash = proceeds
                    side, units, cur = None, 0.0, None
                if tgt_side is not None and cash > 0:    # open only if there is capital
                    pxo = price_today[tgt_side]
                    if pxo > 0:
                        units = cash / pxo
                        asset = self.asset_on if tgt_side == ON else self.asset_off
                        cur = Trade(asset=asset, side=tgt_side, entry_date=d, exit_date=None,
                                    entry_price=pxo, exit_price=None, capital=cash, proceeds=None,
                                    pnl=None, return_pct=None, mfe_pct=0.0, mae_pct=0.0,
                                    days=None, regime_in=str(timeline.at[d, "regime"]),
                                    rsi_in=float(timeline.at[d, "rsi_smooth"]),
                                    vrp_in=float(timeline.at[d, "vrp_smooth"]))
                        cash = 0.0
                        side = tgt_side
                        peak = trough = pxo

            if side is not None:                        # update MFE/MAE
                p = price_today[side]
                peak = max(peak, p)
                trough = min(trough, p)

            val = cash + (units * price_today[side] if side else 0.0)
            # TWR: daily return excluding the day's deposit
            if not np.isnan(prev_val) and prev_val > 0:
                twr_ret = (val - dep) / prev_val - 1.0
            else:
                twr_ret = 0.0
            eq_rows.append((d, val, dep_cum, twr_ret))
            prev_val = val

        eq = pd.DataFrame(eq_rows, columns=["date", "value", "deposits", "twr_ret"]).set_index("date")
        eq["twr"] = (1.0 + eq["twr_ret"]).cumprod()
        # position still open at the end of the period: show it in the log (unrealised P&L)
        if cur is not None and side is not None:
            d = dates[-1]
            pxe = float(px.at[d, side])
            cur.exit_date = None
            cur.exit_price = pxe
            cur.proceeds = units * pxe
            cur.pnl = cur.proceeds - cur.capital
            cur.return_pct = cur.proceeds / cur.capital - 1.0 if cur.capital else 0.0
            cur.mfe_pct = peak / cur.entry_price - 1.0
            cur.mae_pct = trough / cur.entry_price - 1.0
            cur.days = int((d - cur.entry_date).days)
            cur.is_open = True
            trades.append(cur)
        tdf = self._trades_df(trades)
        stats = self._stats(eq, tdf, target)
        notes = [
            "REAL P&L: uses the true historical prices of "
            f"{self.asset_on} (risk-on) and {self.asset_off} (risk-off).",
            "Positions are always LONG the chosen asset: 'short consensus' "
            "means moving to the risk-off asset, not short selling.",
            "TWR = time-weighted return (quality of the strategy, neutral "
            "with respect to WHEN you deposit); the euro P&L also depends on the deposits.",
        ]
        return PortfolioResult(tdf, eq, target, stats, notes)

    # ------------------------------------------------------------------ #
    def _trades_df(self, trades: list[Trade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        rows = [t.__dict__ for t in trades]
        return pd.DataFrame(rows)

    def _stats(self, eq: pd.DataFrame, tdf: pd.DataFrame, target: pd.Series) -> dict:
        dep = float(eq["deposits"].iloc[-1])
        val = float(eq["value"].iloc[-1])
        closed = tdf[~tdf["is_open"]] if not tdf.empty else tdf
        wins = closed[closed["pnl"] > 0] if not closed.empty else closed
        losses = closed[closed["pnl"] <= 0] if not closed.empty else closed
        gross_win = float(wins["pnl"].sum()) if not wins.empty else 0.0
        gross_loss = float(-losses["pnl"].sum()) if not losses.empty else 0.0
        dist = (target.value_counts(normalize=True) * 100).round(1).to_dict()
        return {
            "start": eq.index[0].date().isoformat(), "end": eq.index[-1].date().isoformat(),
            "total_deposited": round(dep, 2),
            "final_value": round(val, 2),
            "total_pnl": round(val - dep, 2),
            "return_on_deposits_pct": round((val / dep - 1) * 100, 1) if dep else None,
            "twr_multiple": round(float(eq["twr"].iloc[-1]), 3),
            "n_trades": int(len(closed)),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if len(closed) else None,
            "avg_win": round(float(wins["pnl"].mean()), 2) if not wins.empty else 0.0,
            "avg_loss": round(float(losses["pnl"].mean()), 2) if not losses.empty else 0.0,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
            "avg_duration_days": round(float(closed["days"].mean()), 0) if not closed.empty else None,
            "time_risk_on_pct": dist.get(ON, 0.0),
            "time_risk_off_pct": dist.get(OFF, 0.0),
            "time_cash_pct": dist.get(CASH, 0.0),
        }
