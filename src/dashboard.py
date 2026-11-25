"""The bot's HTML dashboard: ONE page with everything needed to understand
how it is working and how it performs.

Contents:
  - TODAY's consensus signal, clearly visible (risk-on / cash / risk-off);
  - the votes of the three legs (market, volatility, TDEX) and the position state;
  - the current regime (quadrant) and the key numbers of the day;
  - backtest performance with PERIOD BUTTONS (all/10/5/3/1 years);
  - real P&L on the deposits COMPARED with the S&P 500 (the same deposits
    put on the index and held), full trade log;
  - SLIDER for the date of the first deposit: the P&L section is recomputed
    in the browser as if you had started on another day (SIM_JS).
All dates on the page are in dd/mm/yyyy format.

It is ONLY presentation: it uses the existing engines (Backtester +
PortfolioSimulator) without changing any of the logic. A single
self-contained file (embedded PNGs, no external libraries), openable with a
double click even offline, regenerated at every signal computation.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import pandas as pd

# colours consistent with the rest of the project
QUAD_STYLE = {
    "disinflation": ("Q1 · Disinflation", "#cfe8ff", "#1a5f9e"),
    "reflation":    ("Q2 · Reflation",    "#d6f5d6", "#157f3d"),
    "stagflation":  ("Q3 · Stagflation",  "#ffe6c7", "#b45309"),
    "deflation":    ("Q4 · Deflation",    "#f3d4d4", "#a13030"),
}
TGT_STYLE = {   # consensus target -> (label, text colour, background)
    "risk_on":  ("RISK-ON",  "#157f3d", "#e7f6ec"),
    "risk_off": ("RISK-OFF", "#b45309", "#fff3e0"),
    "cash":     ("CASH",     "#555555", "#efefef"),
}
VOTE_LABEL = {1: ("▲ risk-on", "#157f3d"), -1: ("▼ risk-off", "#b45309"),
              0: ("— abstains", "#888888")}
BLUE, GREY, ORANGE = "#1668dc", "#999999", "#e08a1e"

# period buttons: (key, label, days back; None = all)
PERIODS_BT = [("all", "All", None), ("10y", "10 years", 3652),
              ("5y", "5 years", 1826), ("3y", "3 years", 1095),
              ("1y", "1 year", 365)]
PERIODS_PF = [("all", "All", None), ("3y", "3 years", 1095),
              ("1y", "1 year", 365), ("6m", "6 months", 183)]


def _d(x) -> str:
    """Date in dd/mm/yyyy format (accepts dates, Timestamps or 'YYYY-MM-DD'
    strings). Used for ALL the dates shown on the page."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        return pd.Timestamp(x).strftime("%d/%m/%Y")
    except (ValueError, TypeError):
        return str(x)


# ---------------------------------------------------------------------- #
# Charts (base64 PNG)                                                     #
# ---------------------------------------------------------------------- #
def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def chart_backtest(res_bt, start=None) -> str:
    """Strategy vs buy&hold equity (log) + drawdown, with 'in market' bands.
    With `start` it shows only that period and REBASES both curves to 1×
    at the start of the window (fair comparison over the period)."""
    eq, df = res_bt.equity, res_bt.timeline
    title = "Backtest over the whole history"
    if start is not None:
        eq = eq.loc[eq.index >= start]
        df = df.loc[df.index >= start]
        if len(eq) < 5:
            raise ValueError("period too short")
        eq = eq / eq.iloc[0]                       # rebase to 1× over the period
        title = f"Backtest from {_d(eq.index[0])} (curves rebased to 1×)"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 6.2), height_ratios=[2.2, 1], sharex=True)

    # green bands = days with the market leg IN position
    pos = df["pos_market"]
    changes = pos[pos != pos.shift(1)].index
    blocks = list(changes) + [df.index[-1]]
    for a, b in zip(blocks[:-1], blocks[1:]):
        if pos.loc[a] > 0:
            ax1.axvspan(a, b, color="#d6f5d6", alpha=0.55, lw=0)

    ax1.set_yscale("log")
    top = float(eq.max().max())
    bot = float(eq.min().min())
    candid = (0.3, 0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5,
              3, 4, 6, 8, 12, 20, 30, 50, 80, 120, 200)
    ticks = [t for t in candid if bot * 0.9 <= t <= top * 1.1]
    while len(ticks) > 8:               # long windows: sparser ticks
        ticks = ticks[::2]
    ax1.set_yticks(ticks)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.plot(eq.index, eq["buy_and_hold_spx"], color=GREY, lw=1.1,
             label="Buy & Hold SPX")
    ax1.plot(eq.index, eq["timing_market"], color=BLUE, lw=1.4,
             label="Strategy (market timing)")
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    ax1.set_title(f"{title} — capital (log scale; green bands = in the market)",
                  fontsize=10, loc="left")
    _clean(ax1)

    for col, colr, lab in (("buy_and_hold_spx", GREY, "Buy & Hold"),
                           ("timing_market", BLUE, "Strategy")):
        dd = eq[col] / eq[col].cummax() - 1.0
        ax2.fill_between(dd.index, dd * 100, 0, color=colr, alpha=0.35,
                         label=lab, lw=0)
    ax2.set_ylabel("drawdown %", fontsize=8)
    ax2.legend(loc="lower left", frameon=False, fontsize=8)
    _clean(ax2)
    fig.autofmt_xdate()
    return _b64(fig)


def chart_portfolio(res_pf, bench: pd.Series | None = None,
                    start=None) -> str:
    """Portfolio value vs deposits (and vs the S&P 500 with the same
    deposits), with target bands. Starts from the first deposit."""
    eq = res_pf.equity
    active = eq.index[eq["deposits"] > 0]
    if len(active):
        i0 = max(0, eq.index.get_loc(active[0]) - 10)
        eq = eq.iloc[i0:]
    if start is not None:
        eq = eq.loc[eq.index >= start]
        if len(eq) < 5:
            raise ValueError("period too short")
    tgt = res_pf.target.reindex(eq.index).ffill()

    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    s0, cur = eq.index[0], tgt.iloc[0]
    band = {"risk_on": "#d6f5d6", "risk_off": "#ffe0c2", "cash": None}
    for i in range(1, len(tgt) + 1):
        if i == len(tgt) or tgt.iloc[i] != cur:
            c = band.get(cur)
            if c:
                ax.axvspan(s0, eq.index[min(i, len(tgt) - 1)], color=c,
                           alpha=0.5, lw=0)
            if i < len(tgt):
                s0, cur = eq.index[i], tgt.iloc[i]

    ax.plot(eq.index, eq["deposits"], color=GREY, lw=1.2, ls="--",
            label="Cumulative deposits")
    if bench is not None:
        b = bench.reindex(eq.index)
        ax.plot(b.index, b, color=ORANGE, lw=1.3,
                label="S&P 500 (same deposits)")
    ax.plot(eq.index, eq["value"], color=BLUE, lw=1.5,
            label="Portfolio value (bot)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Real P&L — bot vs S&P 500 with the same deposits "
                 "(bands: green = risk-on, orange = risk-off)",
                 fontsize=10, loc="left")
    _clean(ax)
    fig.autofmt_xdate()
    return _b64(fig)


def benchmark_spx(spx_close: pd.Series, contributions: pd.Series,
                  index: pd.DatetimeIndex) -> pd.Series | None:
    """If I had put THE SAME deposits on the S&P 500 and nothing else:
    units bought at each deposit at that day's price, never sold."""
    if contributions is None or contributions.empty or len(index) == 0:
        return None
    px = spx_close.reindex(index).ffill().bfill()
    dep = pd.Series(0.0, index=index)
    for d, amt in contributions.items():
        pos = index.searchsorted(pd.Timestamp(d))
        if pos < len(index):
            dep.iloc[pos] += float(amt)
    if dep.sum() <= 0:
        return None
    units = (dep / px).cumsum()
    return units * px


# ---------------------------------------------------------------------- #
# Page pieces                                                             #
# ---------------------------------------------------------------------- #
def _fmt_eur(x: float) -> str:
    s = f"{x:,.0f}"
    return f"{s} €"


def _num(x, fmt: str, suffix: str = "") -> str:
    """Formats a number tolerating None (e.g. profit factor with no losses)."""
    return "n/a" if x is None else f"{x:{fmt}}{suffix}"


def _since(pos: pd.Series) -> str:
    """'since 12/05/2026 (43 sessions)' for the current position state."""
    changes = pos[pos != pos.shift(1)].index
    start = changes[-1] if len(changes) else pos.index[0]
    sessions = len(pos.loc[start:])
    return f"since {_d(start)} ({sessions} sessions)"


def _chip(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="chip"><div class="lab">{label}</div>'
            f'<div class="val">{value}</div>{sub_html}</div>')


def _duration_text(days: float) -> str:
    """'27 months, 1 d' from a number of calendar days."""
    g = int(round(days))
    months, rest = divmod(g, 30)
    if months <= 0:
        return f"{rest} d"
    return f"{months} months, {rest} d"


def _trade_metrics(res_pf) -> dict:
    """Trading metrics in a 'trade metrics' panel style, computed from the
    CLOSED trades and from the TWR curve (drawdown neutral to deposits)."""
    t = res_pf.trades
    eq = res_pf.equity
    twr = eq["twr"]
    val = eq["value"]
    closed = t[~t["is_open"]] if not t.empty and "is_open" in t.columns else t
    rp = closed["return_pct"].dropna() if not closed.empty else pd.Series(dtype=float)
    dd_days = closed["days"].dropna() if not closed.empty else pd.Series(dtype=float)

    win = rp[rp > 0]
    loss = rp[rp <= 0]
    hit = float(len(win) / len(rp)) if len(rp) else 0.0

    # Kelly: f = W - (1-W)/R, with R = average win / average loss (absolute)
    kelly = None
    diversif = None
    if len(win) and len(loss):
        avg_w = float(win.mean())
        avg_l = float(-loss.mean())
        if avg_l > 0:
            R = avg_w / avg_l
            f = hit - (1 - hit) / R
            kelly = f
            diversif = (1.0 / f) if f > 0 else None

    # maximum drawdown and duration (peak -> recovery) on the TWR curve
    peak = twr.cummax()
    dd = (twr / peak - 1.0)
    maxdd = float(dd.min()) if len(dd) else 0.0
    longest = 0
    start_peak = None
    prev = None
    under = twr < peak
    for ts, u in under.items():
        if u and start_peak is None:
            start_peak = prev if prev is not None else ts
        elif not u and start_peak is not None:
            longest = max(longest, (ts - start_peak).days)
            start_peak = None
        prev = ts
    if start_peak is not None:
        longest = max(longest, (under.index[-1] - start_peak).days)

    # years since the FIRST DEPOSIT (not since the start of the history:
    # before the deposit no trades are made, and the trades/year count was
    # underestimated)
    active = eq.index[eq["deposits"] > 0] if "deposits" in eq else eq.index
    t0 = active[0] if len(active) else eq.index[0]
    years = max((eq.index[-1] - t0).days / 365.25, 1e-9)

    return {
        "n_trades": int(len(closed)),
        "hit_rate": hit * 100,
        "worst": float(rp.min()) * 100 if len(rp) else 0.0,
        "best": float(rp.max()) * 100 if len(rp) else 0.0,
        "max_dd_pct": maxdd * 100,
        "max_dd_duration": _duration_text(longest),
        "kelly": kelly,
        "diversif": diversif,
        "balance_deposited": float(res_pf.stats.get("total_deposited", 0.0)),
        "balance_current": float(val.iloc[-1]) if len(val) else 0.0,
        "balance_peak": float(val.cummax().iloc[-1]) if len(val) else 0.0,
        "days_min": int(dd_days.min()) if len(dd_days) else 0,
        "days_max": int(dd_days.max()) if len(dd_days) else 0,
        "days_avg": float(dd_days.mean()) if len(dd_days) else 0.0,
        "trades_per_year": len(closed) / years,
    }


def _metric_row(label: str, value: str, col: str = "") -> str:
    style = f' style="color:{col}"' if col else ""
    return (f'<tr><td class="mk">{label}</td>'
            f'<td class="mv"{style}>{value}</td></tr>')


def _metrics_panel(res_pf, title: str = "Trade Metrics") -> str:
    m = _trade_metrics(res_pf)
    green, red = "#157f3d", "#a13030"
    kelly_txt = (f'{m["kelly"]:.2f}'
                 if m["kelly"] is not None else "n/a")
    div_txt = (f'{m["diversif"]:.2f}' + " pos."
               if m["diversif"] is not None else "—")
    rows = "".join([
        _metric_row("Total trades", f'{m["n_trades"]}'),
        _metric_row("Hit rate", f'{m["hit_rate"]:.2f}%'),
        _metric_row("Worst trade",
                    f'{m["worst"]:.2f}%', red),
        _metric_row("Best trade",
                    f'{m["best"]:.2f}%', green),
        _metric_row("Max drawdown",
                    f'{m["max_dd_pct"]:.2f}%', red),
        _metric_row("Max drawdown duration", m["max_dd_duration"]),
        _metric_row("Kelly · full diversif.", f'{kelly_txt} · {div_txt}'),
        _metric_row("Balance · deposited", _fmt_eur(m["balance_deposited"])),
        _metric_row("Balance · current", _fmt_eur(m["balance_current"]), green),
        _metric_row("Balance · peak (ATH)", _fmt_eur(m["balance_peak"])),
        _metric_row("Days in trade · min", f'{m["days_min"]}'),
        _metric_row("Days in trade · max", f'{m["days_max"]}'),
        _metric_row("Days in trade · avg",
                    f'{m["days_avg"]:.2f}'),
        _metric_row("Trades / year",
                    f'{m["trades_per_year"]:.2f}'),
    ])
    return (f'<div class="card metrics">'
            f'<div class="mtitle">{title}</div>'
            f'<table class="mtab"><tbody id="pf-metrics">{rows}</tbody></table>'
            f'<p class="sub" style="margin:10px 2px 0">Computed on the closed '
            f'trades; drawdown and duration on the TWR curve (neutral to '
            f'deposits). Kelly = optimal fraction of capital per trade.</p>'
            f'</div>')


# ---------------------------------------------------------------------- #
# Exposure: how much time the capital spends in equities, in TLT, in cash #
# ---------------------------------------------------------------------- #
# (strong colour for the bar, soft colour for backgrounds)
EXP_STYLE = {
    "risk_on":  ("#2f9e5a", "#e7f6ec"),
    "risk_off": ("#e08a1e", "#fff3e0"),
    "cash":     ("#c2c8d0", "#f1f2f4"),
}
STATES = ("risk_on", "risk_off", "cash")


def _state_blocks(s: pd.Series) -> list[dict]:
    """Contiguous periods with the same state: [{state, from, to, sessions}, ...]."""
    if s is None or not len(s):
        return []
    v = s.to_numpy()
    idx = s.index
    out, start = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[start]:
            out.append({"state": str(v[start]), "from": idx[start],
                        "to": idx[i - 1], "sessions": i - start})
            start = i
    return out


def _exposure(target: pd.Series) -> dict:
    """'Where is the capital' statistics over the whole backtest history.

    For each of the three states (equities / bonds / cash): how many
    sessions, what percentage of the time, in how many distinct periods,
    average duration and longest period. Plus: current state (since when)
    and year-by-year breakdown."""
    n = int(len(target))
    blocks = _state_blocks(target)
    per_state = {}
    for st in STATES:
        b = [x for x in blocks if x["state"] == st]
        sessions = sum(x["sessions"] for x in b)
        longest = max(b, key=lambda x: x["sessions"]) if b else None
        per_state[st] = {
            "sessions": sessions,
            "pct": round(sessions / n * 100, 1) if n else 0.0,
            "periods": len(b),
            "avg": round(sessions / len(b)) if b else 0,
            "max": longest["sessions"] if longest else 0,
            "max_from": longest["from"].date().isoformat() if longest else None,
            "max_to": longest["to"].date().isoformat() if longest else None,
        }
    # year-by-year breakdown (percentages of the year's sessions)
    years = []
    for year, sub in target.groupby(target.index.year):
        q = len(sub)
        row = {"year": int(year), "sessions": q}
        for st in STATES:
            row[st] = round(float((sub == st).sum()) / q * 100, 1) if q else 0.0
        years.append(row)
    last = blocks[-1] if blocks else None
    return {
        "n": n,
        "from": target.index[0].date().isoformat() if n else None,
        "to": target.index[-1].date().isoformat() if n else None,
        "per_state": per_state,
        "invested_pct": round(per_state["risk_on"]["pct"]
                              + per_state["risk_off"]["pct"], 1),
        "blocks": blocks,
        "years": years,
        "now": {"state": last["state"], "from": last["from"].date().isoformat(),
                "sessions": last["sessions"]} if last else None,
    }


def _exposure_bar(exp: dict) -> str:
    """Continuous timeline bar: one coloured band per period, with the year
    ticks below. Uses a hard-stop gradient (no image), so it copes with
    hundreds of state changes."""
    n = exp["n"]
    if not n:
        return ""
    stop, acc = [], 0
    for b in exp["blocks"]:
        a = acc / n * 100
        acc += b["sessions"]
        z = acc / n * 100
        stop.append(f'{EXP_STYLE[b["state"]][0]} {a:.4f}% {z:.4f}%')
    grad = "linear-gradient(90deg," + ",".join(stop) + ")"
    # ticks: start of every year (labels thinned out if the history is long)
    years = [r["year"] for r in exp["years"]]
    step = 1 if len(years) <= 12 else (2 if len(years) <= 24 else 5)
    pos, acc = [], 0
    for r in exp["years"]:
        if r["year"] % step == 0 or step == 1:
            pos.append((r["year"], acc / n * 100))
        acc += r["sessions"]
    ticks = "".join(
        f'<span style="position:absolute;left:{p:.3f}%;transform:'
        f'translateX(-50%);font-size:10.5px;color:#888">{a}</span>'
        for a, p in pos)
    return (f'<div class="expbar" style="background:{grad}"></div>'
            f'<div style="position:relative;height:16px;margin-top:2px">'
            f'{ticks}</div>')


def _exposure_card(exp: dict, asset_on: str, asset_off: str,
                   source: str) -> str:
    """The «Where is the capital» section of the dashboard."""
    if not exp["n"]:
        return ""
    ps = exp["per_state"]
    names = {"risk_on": f"Equities · {asset_on}",
             "risk_off": f"Bonds · {asset_off}",
             "cash": "Cash (no consensus)"}

    def _sub(st):
        d = ps[st]
        if not d["periods"]:
            return "never"
        return (f'{d["sessions"]} sessions · {d["periods"]} periods · '
                f'avg {d["avg"]} sessions')

    chips = "".join([
        _chip(names["risk_on"], f'{ps["risk_on"]["pct"]}%', _sub("risk_on")),
        _chip(names["risk_off"], f'{ps["risk_off"]["pct"]}%', _sub("risk_off")),
        _chip(names["cash"], f'{ps["cash"]["pct"]}%', _sub("cash")),
        _chip("Invested (stocks + bonds)", f'{exp["invested_pct"]}%',
              f'cash {ps["cash"]["pct"]}% of the time'),
    ])

    now = exp["now"]
    lab_now = {"risk_on": f"in <b>{asset_on}</b> (equities)",
               "risk_off": f"in <b>{asset_off}</b> (bonds)",
               "cash": "in <b>cash</b>"}[now["state"]]
    col_now = EXP_STYLE[now["state"]][0]
    state_now = (f'<div class="nowbox" style="border-left:4px solid {col_now}">'
                 f'<b>Right now</b> the capital would be {lab_now}, '
                 f'since <b>{_d(now["from"])}</b> ({now["sessions"]} sessions in a row).'
                 f'</div>')

    legend = "".join(
        f'<span class="tag" style="background:{EXP_STYLE[st][1]};'
        f'border-left:10px solid {EXP_STYLE[st][0]}">{names[st]}</span>'
        for st in STATES)

    # longest periods spent in each state
    longest = " · ".join(
        f'{names[st].split(" · ")[-1]}: {ps[st]["max"]} sessions '
        f'({_d(ps[st]["max_from"])} → {_d(ps[st]["max_to"])})'
        for st in STATES if ps[st]["max"])

    rows = ""
    for r in exp["years"]:
        bar = "".join(
            f'<span style="display:inline-block;height:9px;'
            f'width:{r[st]:.2f}%;background:{EXP_STYLE[st][0]}"></span>'
            for st in STATES if r[st] > 0)
        rows += (f'<tr><td><b>{r["year"]}</b></td>'
                 f'<td>{r["risk_on"]:.0f}%</td>'
                 f'<td>{r["risk_off"]:.0f}%</td>'
                 f'<td>{r["cash"]:.0f}%</td>'
                 f'<td style="width:45%;line-height:0">{bar}</td></tr>')
    table = (f'<div class="tablewrap" style="max-height:300px"><table>'
             f'<thead><tr><th>Year</th><th>{asset_on}</th>'
             f'<th>{asset_off}</th><th>Cash</th>'
             f'<th>Breakdown</th></tr></thead>'
             f'<tbody>{rows}</tbody></table></div>')

    return f"""
<div class="card">
  <h2>Where is the capital — time in {asset_on}, in {asset_off} and in cash</h2>
  <div class="chips">{chips}</div>
  {state_now}
  <h3>Exposure timeline ({_d(exp['from'])} → {_d(exp['to'])},
      {exp['n']} sessions)</h3>
  {_exposure_bar(exp)}
  <div style="margin-top:6px">{legend}</div>
  <p class="sub">Longest uninterrupted periods — {longest}.</p>
  <h3>Year by year</h3>
  {table}
  <p class="sub">Percentages computed on the trading sessions of the whole
  backtest history. {source} «Cash» = the legs' votes do not reach the
  required consensus, so the capital stays liquid and waits.</p>
</div>"""


def _vote_card(name: str, vote: int | None, detail: str) -> str:
    if vote is None:
        txt, col = "disabled", "#bbbbbb"
    else:
        txt, col = VOTE_LABEL[int(vote)]
    return (f'<div class="vote"><div class="lab">{name}</div>'
            f'<div class="val" style="color:{col}">{txt}</div>'
            f'<div class="sub">{detail}</div></div>')


def _chart_group(group: str, charts: list[tuple[str, str, str]],
                 alt: str) -> str:
    """Period buttons + stacked images (only one is visible)."""
    if not charts:
        return ""
    buttons = "".join(
        f'<button type="button" class="pbtn{" active" if i == 0 else ""}" '
        f"onclick=\"showPeriod('{group}','{k}',this)\">{lab}</button>"
        for i, (k, lab, _) in enumerate(charts))
    imgs = "".join(
        f'<img data-g="{group}" data-k="{k}" alt="{alt}" '
        f'src="data:image/png;base64,{b}"'
        + ("" if i == 0 else ' style="display:none"') + ">"
        for i, (k, _, b) in enumerate(charts))
    return f'<div class="periods">{buttons}</div>{imgs}'


def _period_charts(periods, render, as_of) -> list[tuple[str, str, str]]:
    """Generates (key, label, png) for every renderable period."""
    out = []
    for k, lab, days in periods:
        try:
            start = (None if days is None
                     else pd.Timestamp(as_of) - pd.Timedelta(days=days))
            out.append((k, lab, render(start)))
        except Exception:
            continue                    # window too short: button omitted
    return out


# ---------------------------------------------------------------------- #
# INTERACTIVE charts (ECharts via CDN): free zoom with the wheel,         #
# drag to pan, slider below, double click = reset.                        #
# ---------------------------------------------------------------------- #
GREEN_BAND = "rgba(214,245,214,0.55)"
ORANGE_BAND = "rgba(255,224,194,0.55)"


def _bands(mask: pd.Series, colour: str) -> list[list[str]]:
    """Intervals [start, end, colour] where the mask is true."""
    mask = mask.fillna(False).astype(bool)
    changes = mask[mask != mask.shift(1)].index
    blocks = list(changes) + [mask.index[-1]]
    out = []
    for a, b in zip(blocks[:-1], blocks[1:]):
        if bool(mask.loc[a]):
            out.append([a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"),
                        colour])
    return out


def _blob_backtest(res_bt, tlt: pd.Series | None = None,
                   real_cash: pd.Series | None = None) -> dict:
    """Data of the backtest chart: equity (log) + drawdown, in-market bands.
    `tlt` (optional): TLT closes, rebased to 1x on the first available day
    of the history, as a bond comparison (column 5).
    `real_cash` (optional): purchasing power of 1 held in cash since the
    first day, net of inflation (column 6, dashed line). The flat line at
    1x is drawn by the JavaScript."""
    eq, df = res_bt.equity, res_bt.timeline
    bh, tm = eq["buy_and_hold_spx"], eq["timing_market"]
    dd_bh = (bh / bh.cummax() - 1.0) * 100
    dd_tm = (tm / tm.cummax() - 1.0) * 100
    t = None
    if tlt is not None and len(tlt):
        t = tlt.reindex(eq.index).ffill()
        valid = t.dropna()
        t = (t / float(valid.iloc[0])) if len(valid) else None
    cr = None
    if real_cash is not None and real_cash.notna().any():
        cr = real_cash.reindex(eq.index)
    rows = []
    for i, d in enumerate(eq.index):
        r = [d.strftime("%Y-%m-%d"), round(float(bh.iloc[i]), 4),
             round(float(tm.iloc[i]), 4), round(float(dd_bh.iloc[i]), 2),
             round(float(dd_tm.iloc[i]), 2)]
        v = t.iloc[i] if t is not None else None
        r.append(None if v is None or pd.isna(v) else round(float(v), 4))
        if cr is not None:
            c = cr.iloc[i]
            r.append(None if pd.isna(c) else round(float(c), 4))
        rows.append(r)
    return {"rows": rows,
            "bands": _bands(df["pos_market"] > 0, GREEN_BAND),
            "has_tlt": bool(t is not None),
            "has_infl": bool(cr is not None)}


def _blob_simulator(res_pf, timeline: pd.DataFrame, px_on: pd.Series,
                    px_off: pd.Series, spx: pd.Series | None,
                    contr: pd.Series, asset_on: str, asset_off: str) -> dict:
    """Ingredients for the «date of the first deposit» SLIDER.

    The page's JavaScript redoes the same simulation as
    PortfolioSimulator.run (same order of operations, same prices, same
    daily target) starting from the chosen date: here only the raw data is
    passed, day by day, on the same sessions as the simulator. At the real
    date in contributions.csv the numbers match the ones computed in Python."""
    idx = res_pf.equity.index
    # prices exactly as the simulator sees them (ffill on the timeline)
    px = (pd.DataFrame({"on": px_on, "off": px_off})
          .reindex(timeline.index).ffill().reindex(idx))
    tgt = res_pf.target.reindex(idx)
    code = {"risk_on": "1", "risk_off": "2"}             # otherwise 0 = cash
    reg = timeline["regime"].reindex(idx).astype(str)
    labels = sorted(reg.unique())
    pos = {r: i for i, r in enumerate(labels)}
    # benchmark: same rules as benchmark_spx (ffill + bfill)
    spx_al = spx.reindex(idx).ffill().bfill() if spx is not None else None
    if spx_al is not None and spx_al.isna().any():
        spx_al = None
    return {
        "date": [d.strftime("%Y-%m-%d") for d in idx],
        "tgt": "".join(code.get(str(t), "0") for t in tgt),
        "on": [round(float(v), 4) for v in px["on"]],
        "off": [round(float(v), 4) for v in px["off"]],
        "spx": ([round(float(v), 4) for v in spx_al]
                if spx_al is not None else None),
        "reg": [pos[r] for r in reg],
        "reg_lab": [QUAD_STYLE.get(r, (r,))[0] for r in labels],
        "plan": [[pd.Timestamp(d).strftime("%Y-%m-%d"), float(a)]
                 for d, a in contr.sort_index().items()],
        "asset_on": asset_on, "asset_off": asset_off,
    }


# The page's JavaScript (raw strings, NOT f-strings: the braces are JS).
# Order in the page: COMMON_JS (shared helpers) -> ECHARTS_JS (backtest
# chart) -> SIM_JS (first-deposit slider + P&L section).

COMMON_JS = r"""
// Dates in dd/mm/yyyy format everywhere (axes, tooltips, slider).
var fmtDate = function (v) {
  if (typeof v === 'string') {
    var p = v.split('-');
    return p.length === 3 ? p[2] + '/' + p[1] + '/' + p[0] : v;
  }
  var d = new Date(v), z = function (n) { return (n < 10 ? '0' : '') + n; };
  return z(d.getDate()) + '/' + z(d.getMonth() + 1) + '/' + d.getFullYear();
};
var DMY = '{dd}/{MM}/{yyyy}';
// axis ticks: years as '2013', everything else dd/mm/yyyy
var AXIS_DATE = {year: '{yyyy}', month: DMY, day: DMY, hour: DMY,
                 minute: DMY, second: DMY, millisecond: DMY, none: DMY};
// pointer label and tooltip header
var POINTER_DATE = {label: {formatter: function (p) { return fmtDate(p.value); }}};
var HINT_ZOOM = 'wheel = zoom · drag = pan · double click = whole period';
function ecBands(bands) {
  return {silent: true, data: bands.map(function (b) {
    return [{xAxis: b[0], itemStyle: {color: b[2]}}, {xAxis: b[1]}];
  })};
}
function ecZoomReset(c) {
  c.getZr().on('dblclick', function () {
    c.dispatchAction({type: 'dataZoom', start: 0, end: 100});
  });
  window.addEventListener('resize', function () { c.resize(); });
}
"""

ECHARTS_JS = r"""
(function () {
  if (typeof echarts === 'undefined') {
    document.querySelectorAll('.ichart').forEach(function (d) {
      d.innerHTML = '<p style="color:#a13030;font-size:13px">Interactive ' +
        'charts not available: an internet connection is needed (library from the CDN). ' +
        'Regenerate the dashboard when you are online.</p>';
    });
    return;
  }
  if (window.BT) {
    var legBT = ['Buy & Hold SPX', 'Strategy (market timing)'];
    var serTLT = [];
    if (BT.has_tlt) {
      legBT.push('TLT (rebased)');
      serTLT.push({name: 'TLT (rebased)', type: 'line', encode: {x: 0, y: 5},
        showSymbol: false, sampling: 'lttb', connectNulls: false,
        lineStyle: {width: 1.1, color: '#8a56c9'},
        itemStyle: {color: '#8a56c9'}});
    }
    // Cash: flat line at 1x (money held still, not invested) and the same
    // line DASHED net of inflation (purchasing power).
    var CASH = 'Cash (held at 1×)';
    var CASH_R = 'Cash net of inflation ' + (BT.infl_name || '');
    var CASH_COL = '#3d3d3d';
    var FLAT = BT.rows.map(function (r) { return [r[0], 1]; });
    legBT.push(CASH);
    var serCash = [{name: CASH, type: 'line', datasetIndex: 2,
      encode: {x: 0, y: 1}, showSymbol: false,
      lineStyle: {width: 1.2, color: CASH_COL},
      itemStyle: {color: CASH_COL}}];
    if (BT.has_infl) {
      // in the legend: a dash without a dot, so it differs from the flat line
      legBT.push({name: CASH_R, itemStyle: {opacity: 0},
                  lineStyle: {type: [4, 3], width: 2, color: CASH_COL}});
      serCash.push({name: CASH_R, type: 'line', encode: {x: 0, y: 6},
        showSymbol: false, sampling: 'lttb', connectNulls: false,
        lineStyle: {width: 1.4, color: CASH_COL, type: [6, 4]},
        itemStyle: {color: CASH_COL}});
    }
    // Drawdown split into 3 stacked, non-overlapping bands:
    // 'both' = the common part (from 0 to the shallower DD of the two);
    // 'SPX only' / 'strategy only' = the extra part of whichever is deeper.
    var r2 = function (x) { return Math.round(x * 100) / 100; };
    var DD = BT.rows.map(function (r) {
      var a = r[3], b = r[4];
      return [r[0], r2(Math.max(a, b)), r2(Math.min(a - b, 0)),
              r2(Math.min(b - a, 0))];
    });
    var legDD = ['Both in drawdown', 'SPX only', 'Strategy only'];
    var ddArea = function (name, col, colour, opac) {
      return {name: name, type: 'line', xAxisIndex: 1, yAxisIndex: 1,
        datasetIndex: 1, encode: {x: 0, y: col}, stack: 'dd',
        stackStrategy: 'all', showSymbol: false, tooltip: {show: false},
        lineStyle: {width: 0}, areaStyle: {color: colour, opacity: opac},
        itemStyle: {color: colour, opacity: opac}};
    };
    var cb = echarts.init(document.getElementById('chbt'));
    cb.setOption({
      animation: false,
      dataset: [{source: BT.rows}, {source: DD}, {source: FLAT}],
      legend: [{top: 0, data: legBT},
               {top: '61%', right: 18, data: legDD, icon: 'rect',
                itemWidth: 12, itemHeight: 8, selectedMode: false,
                textStyle: {fontSize: 10}}],
      tooltip: {trigger: 'axis', axisPointer: {type: 'cross'}},
      axisPointer: {link: [{xAxisIndex: 'all'}]},
      grid: [{left: 64, right: 18, top: 32, height: '50%'},
             {left: 64, right: 18, top: '66%', height: '17%'}],
      xAxis: [{type: 'time', gridIndex: 0, axisLabel: {show: false},
               axisPointer: POINTER_DATE},
              {type: 'time', gridIndex: 1, axisLabel: {formatter: AXIS_DATE},
               axisPointer: POINTER_DATE}],
      yAxis: [{type: 'log', gridIndex: 0, min: 'dataMin', max: 'dataMax',
               splitLine: {lineStyle: {opacity: .4}},
               axisLabel: {formatter: function (v) {
                 return (v >= 1 ? Math.round(v * 100) / 100
                                : Math.round(v * 1000) / 1000) + '×'; }}},
              {type: 'value', gridIndex: 1, name: 'drawdown %',
               nameTextStyle: {fontSize: 10},
               axisLabel: {formatter: '{value}%'}}],
      dataZoom: [{type: 'inside', xAxisIndex: [0, 1], filterMode: 'filter'},
                 {type: 'slider', xAxisIndex: [0, 1], bottom: 2, height: 18,
                  labelFormatter: function (v) { return fmtDate(v); }}],
      series: [
        {name: 'Buy & Hold SPX', type: 'line', encode: {x: 0, y: 1},
         showSymbol: false, sampling: 'lttb',
         lineStyle: {width: 1.1, color: '#999999'},
         itemStyle: {color: '#999999'}, markArea: ecBands(BT.bands)},
        {name: 'Strategy (market timing)', type: 'line', encode: {x: 0, y: 2},
         showSymbol: false, sampling: 'lttb',
         lineStyle: {width: 1.6, color: '#1668dc'},
         itemStyle: {color: '#1668dc'}},
        ddArea(legDD[0], 1, '#d64541', .55),
        ddArea(legDD[1], 2, '#999999', .40),
        ddArea(legDD[2], 3, '#1668dc', .35),
        {name: 'DD Buy & Hold', type: 'line', xAxisIndex: 1, yAxisIndex: 1,
         encode: {x: 0, y: 3}, showSymbol: false,
         lineStyle: {width: .6, color: '#888888'},
         itemStyle: {color: '#999999'}},
        {name: 'DD Strategy', type: 'line', xAxisIndex: 1, yAxisIndex: 1,
         encode: {x: 0, y: 4}, showSymbol: false,
         lineStyle: {width: .6, color: '#1668dc'},
         itemStyle: {color: '#1668dc'}}].concat(serTLT).concat(serCash)
    });
    ecZoomReset(cb);
    document.getElementById('chbt-hint').textContent = HINT_ZOOM;
  }
})();
"""

# «Date of the first deposit» slider: redoes in JavaScript the simulation of
# PortfolioSimulator.run (src/portfolio) starting from the chosen date and
# redraws the WHOLE P&L section (boxes, chart, trade log, metrics).
# Same order of operations as the Python: at the real date in
# contributions.csv the numbers match those computed by the bot.
SIM_JS = r"""
(function () {
  var S = window.PF;
  if (!S || !S.date || !S.date.length || !S.plan || !S.plan.length) return;
  var N = S.date.length, D = S.date, TG = S.tgt, ON = S.on, OFF = S.off,
      SPX = S.spx, EC = (typeof echarts !== 'undefined');
  var BLUE = '#1668dc', ORANGE = '#e08a1e', GREEN = '#157f3d', RED = '#a13030';
  var el = function (id) { return document.getElementById(id); };

  // ---- formatting (identical to the Python part) -----------------------
  function thousands(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function eur(x) { return (x < 0 ? '-' : '') + thousands(Math.round(Math.abs(x))) + ' €'; }
  function eurS(x) { return (x < 0 ? '-' : '+') + thousands(Math.round(Math.abs(x))) + ' €'; }
  function pct1(x) { var v = (x * 100).toFixed(1); return (v.charAt(0) === '-' ? '' : '+') + v + '%'; }
  function dec(x, n) { return x.toFixed(n); }
  function pctS(x) { var v = dec(x, 1); return (v.charAt(0) === '-' ? '' : '+') + v + '%'; }
  function duration(g) {
    g = Math.round(g); var m = Math.floor(g / 30), r = g - m * 30;
    return m <= 0 ? r + ' d' : m + ' months, ' + r + ' d';
  }
  function days(a, b) { return Math.round((Date.parse(D[b]) - Date.parse(D[a])) / 864e5); }
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
  function z2(n) { return (n < 10 ? '0' : '') + n; }
  // first session >= ISO date (like pandas searchsorted)
  function findIdx(iso) {
    var lo = 0, hi = N;
    while (lo < hi) { var m = (lo + hi) >> 1; if (D[m] < iso) lo = m + 1; else hi = m; }
    return lo;
  }

  // ---- deposit plan: it shifts AS A BLOCK with the first one ----------
  var k0 = findIdx(S.plan[0][0]);
  var PLAN = S.plan.map(function (p) { return {off: findIdx(p[0]) - k0, amt: p[1]}; });
  var START0 = Math.min(k0, N - 1);

  // ---- simulation: same logic as PortfolioSimulator.run ---------------
  function simulate(s) {
    var dep = new Float64Array(N);
    PLAN.forEach(function (p) { var k = s + p.off; if (k >= 0 && k < N) dep[k] += p.amt; });
    var cash = 0, side = 0, units = 0, cur = null, peak = NaN, trough = NaN;
    var trades = [], depCum = 0, prev = NaN, g = 1;
    var val = new Float64Array(N), dc = new Float64Array(N), twr = new Float64Array(N);
    for (var i = 0; i < N; i++) {
      var pr = [0, ON[i], OFF[i]], d = dep[i];
      depCum += d;
      if (d > 0 && side) { units += d / pr[side]; cur.capital += d; }   // 1) deposit
      else if (d > 0) cash += d;
      var ts = +TG.charAt(i);                                            // 0 cash, 1 on, 2 off
      if (ts !== side) {                                                 // target change -> trade
        if (side) {                                                      // close
          var pxe = pr[side], proc = units * pxe;
          cur.exit = i; cur.pnl = proc - cur.capital;
          cur.ret = cur.capital ? proc / cur.capital - 1 : 0;
          cur.mfe = cur.epx ? peak / cur.epx - 1 : 0;
          cur.mae = cur.epx ? trough / cur.epx - 1 : 0;
          cur.days = days(cur.entry, i);
          trades.push(cur);
          cash = proc; side = 0; units = 0; cur = null;
        }
        if (ts && cash > 0) {                                            // open
          var pxo = pr[ts];
          if (pxo > 0) {
            units = cash / pxo;
            cur = {asset: ts === 1 ? S.asset_on : S.asset_off, entry: i,
                   exit: null, epx: pxo, capital: cash, reg: S.reg[i], open: false};
            cash = 0; side = ts; peak = trough = pxo;
          }
        }
      }
      if (side) { var pp = pr[side]; if (pp > peak) peak = pp; if (pp < trough) trough = pp; }
      var v = cash + (side ? units * pr[side] : 0);
      var r = (!isNaN(prev) && prev > 0) ? (v - d) / prev - 1 : 0;
      g *= 1 + r; val[i] = v; dc[i] = depCum; twr[i] = g; prev = v;
    }
    if (cur && side) {                                                   // position still open
      var pxl = side === 1 ? ON[N - 1] : OFF[N - 1], proc2 = units * pxl;
      cur.pnl = proc2 - cur.capital;
      cur.ret = cur.capital ? proc2 / cur.capital - 1 : 0;
      cur.mfe = peak / cur.epx - 1; cur.mae = trough / cur.epx - 1;
      cur.days = days(cur.entry, N - 1); cur.open = true;
      trades.push(cur);
    }
    var bench = null;                                                    // S&P 500, same deposits
    if (SPX) {
      bench = new Float64Array(N); var u = 0;
      for (i = 0; i < N; i++) { if (dep[i] > 0) u += dep[i] / SPX[i]; bench[i] = u * SPX[i]; }
    }
    return {s: s, val: val, dc: dc, twr: twr, trades: trades, bench: bench};
  }

  // ---- statistics and metrics (like _stats and _trade_metrics) ---------
  function metrics(R) {
    var closed = R.trades.filter(function (t) { return !t.open; });
    var pnl = closed.map(function (t) { return t.pnl; });
    var wins = pnl.filter(function (x) { return x > 0; });
    var loss = pnl.filter(function (x) { return x <= 0; });
    var gw = sum(wins), gl = -sum(loss);
    var deposited = R.dc[N - 1], value = R.val[N - 1];
    var rp = closed.map(function (t) { return t.ret; });
    var gg = closed.map(function (t) { return t.days; });
    var w = rp.filter(function (x) { return x > 0; });
    var l = rp.filter(function (x) { return x <= 0; });
    var hit = rp.length ? w.length / rp.length : 0;
    var kelly = null, div = null;
    if (w.length && l.length) {
      var aw = sum(w) / w.length, al = -sum(l) / l.length;
      if (al > 0) { var f = hit - (1 - hit) / (aw / al); kelly = f; div = f > 0 ? 1 / f : null; }
    }
    // maximum drawdown and duration (peak -> recovery) on the TWR curve
    var pk = -Infinity, maxdd = 0, longest = 0, sp = -1;
    for (var i = 0; i < N; i++) {
      var x = R.twr[i]; if (x > pk) pk = x;
      var dd = x / pk - 1; if (dd < maxdd) maxdd = dd;
      var u = x < pk;
      if (u && sp < 0) sp = i > 0 ? i - 1 : i;
      else if (!u && sp >= 0) { longest = Math.max(longest, days(sp, i)); sp = -1; }
    }
    if (sp >= 0) longest = Math.max(longest, days(sp, N - 1));
    var peakVal = 0; for (i = 0; i < N; i++) if (R.val[i] > peakVal) peakVal = R.val[i];
    var years = Math.max(days(R.s, N - 1) / 365.25, 1e-9);
    return {
      deposited: deposited, value: value, pnl: value - deposited,
      rend: deposited ? (value / deposited - 1) * 100 : null,
      twr: Math.round(R.twr[N - 1] * 1000) / 1000,
      n: closed.length,
      win: closed.length ? Math.round(wins.length / closed.length * 1000) / 10 : null,
      pf: gl ? Math.round(gw / gl * 100) / 100 : null,
      hit: hit * 100,
      worst: rp.length ? Math.min.apply(null, rp) * 100 : 0,
      best: rp.length ? Math.max.apply(null, rp) * 100 : 0,
      maxdd: maxdd * 100, duration: duration(longest), kelly: kelly, div: div,
      peak: peakVal,
      dmin: gg.length ? Math.min.apply(null, gg) : 0,
      dmax: gg.length ? Math.max.apply(null, gg) : 0,
      davg: gg.length ? sum(gg) / gg.length : 0,
      tyear: closed.length / years, years: years
    };
  }

  // ---- HTML pieces (same as the Python part) ----------------------------
  function chip(lab, val, sub) {
    return '<div class="chip"><div class="lab">' + lab + '</div><div class="val">' +
      val + '</div>' + (sub ? '<div class="sub">' + sub + '</div>' : '') + '</div>';
  }
  function row(lab, val, col) {
    return '<tr><td class="mk">' + lab + '</td><td class="mv"' +
      (col ? ' style="color:' + col + '"' : '') + '>' + val + '</td></tr>';
  }
  function htmlChips(R, M) {
    var pcol = M.pnl >= 0 ? GREEN : RED;
    var c = [chip('Deposited', eur(M.deposited)),
             chip('Value today (bot)', eur(M.value))];
    if (R.bench) {
      var bf = R.bench[N - 1], diff = M.value - bf;
      c.push(chip('S&amp;P 500 (same deposits)', eur(bf),
        '<span style="color:' + (diff >= 0 ? GREEN : RED) + '">bot ' + eurS(diff) +
        ' vs the index</span>'));
    }
    var tw = String(M.twr); if (tw.indexOf('.') < 0) tw += '.0';
    c.push(chip('Profit/loss', '<span style="color:' + pcol + '">' + eurS(M.pnl) + '</span>',
                M.rend === null ? 'n/a' : (M.rend >= 0 ? '+' : '') + M.rend.toFixed(1) + '% on deposits'));
    c.push(chip('Strategy (TWR)', tw + '×'));
    c.push(chip('Trades', String(M.n), 'winning ' + (M.win === null ? 'n/a' : M.win.toFixed(0) + '%') +
                ' · PF ' + (M.pf === null ? 'n/a' : M.pf.toFixed(2))));
    return c.join('');
  }
  function htmlTable(R) {
    if (!R.trades.length) return "<p class='sub'>No trades yet.</p>";
    var rows = '';
    for (var j = R.trades.length - 1; j >= 0; j--) {
      var t = R.trades[j], pc = t.pnl >= 0 ? GREEN : RED;
      rows += '<tr><td>' + fmtDate(D[t.entry]) + '</td><td>' +
        (t.open ? '<i>open</i>' : fmtDate(D[t.exit])) + '</td><td>' + t.asset +
        '</td><td>' + t.days + '</td><td style="color:' + pc + '">' + eurS(t.pnl) +
        '</td><td style="color:' + pc + '">' + pct1(t.ret) + '</td><td>' + pct1(t.mfe) +
        '</td><td>' + pct1(t.mae) + '</td><td>' + S.reg_lab[t.reg] + '</td></tr>';
    }
    return '<div class="tablewrap"><table><thead><tr><th>Entry</th><th>Exit</th>' +
      '<th>Asset</th><th>Days</th><th>P&amp;L</th><th>P&amp;L %</th><th>MFE</th>' +
      '<th>MAE</th><th>Entry regime</th></tr></thead><tbody>' + rows +
      '</tbody></table></div>';
  }
  function htmlMetrics(M) {
    var k = M.kelly === null ? 'n/a' : dec(M.kelly, 2);
    var dv = M.div === null ? '—' : dec(M.div, 2) + ' pos.';
    return [row('Total trades', String(M.n)),
            row('Hit rate', dec(M.hit, 2) + '%'),
            row('Worst trade', dec(M.worst, 2) + '%', RED),
            row('Best trade', dec(M.best, 2) + '%', GREEN),
            row('Max drawdown', dec(M.maxdd, 2) + '%', RED),
            row('Max drawdown duration', M.duration),
            row('Kelly · full diversif.', k + ' · ' + dv),
            row('Balance · deposited', eur(M.deposited)),
            row('Balance · current', eur(M.value), GREEN),
            row('Balance · peak (ATH)', eur(M.peak)),
            row('Days in trade · min', String(M.dmin)),
            row('Days in trade · max', String(M.dmax)),
            row('Days in trade · avg', dec(M.davg, 2)),
            row('Trades / year', dec(M.tyear, 2))].join('');
  }
  function maxDD(arr, from) {
    var pk = -Infinity, m = 0;
    for (var i = from; i < N; i++) { if (arr[i] > pk) pk = arr[i]; var d = arr[i] / pk - 1; if (d < m) m = d; }
    return m;
  }
  function htmlOutcome(R, M) {
    var s = R.s, a = M.years, bot = R.twr[N - 1] / R.twr[s];
    var spx = SPX ? SPX[N - 1] / SPX[s] : null;
    var annual = a >= 1;
    var f = function (x) { return pctS((annual ? Math.pow(x, 1 / a) - 1 : x - 1) * 100); };
    var h = 'From <b>' + fmtDate(D[s]) + '</b> to <b>' + fmtDate(D[N - 1]) + '</b> (' +
      dec(a, 1) + ' years) — ' + (annual ? 'compound annual' : 'total') +
      ' return: <b style="color:' + BLUE + '">bot ' + f(bot) + '</b>';
    if (spx !== null) h += ' · <b style="color:' + ORANGE + '">S&amp;P 500 ' + f(spx) + '</b>';
    h += '<br>Max drawdown over the period: bot ' + pctS(maxDD(R.twr, s) * 100);
    if (SPX) h += ' · S&amp;P 500 ' + pctS(maxDD(SPX, s) * 100);
    return h;
  }

  // ---- P&L chart (ECharts) ---------------------------------------------
  var cp = null, cc = null, CAGR_FROM = -1, CAGR_TO = -1;
  function bands(i0, code, col) {
    var out = [], a = i0;
    for (var i = i0 + 1; i <= N; i++) {
      if (i === N || (TG.charAt(i) === code) !== (TG.charAt(a) === code)) {
        if (TG.charAt(a) === code) out.push([D[a], D[i === N ? N - 1 : i], col]);
        a = i;
      }
    }
    return out;
  }
  function chartPF(R) {
    if (!EC || !el('chpf')) return;
    var i0 = Math.max(0, R.s - 10), rows = [];
    for (var i = i0; i < N; i++) {
      var r = [D[i], R.dc[i], R.val[i]];
      if (R.bench) r.push(R.bench[i]);
      rows.push(r);
    }
    var bb = bands(i0, '1', 'rgba(214,245,214,0.55)').concat(bands(i0, '2', 'rgba(255,224,194,0.55)'));
    if (!cp) {
      var ser = [{name: 'Cumulative deposits', type: 'line', encode: {x: 0, y: 1},
        showSymbol: false, lineStyle: {width: 1.2, type: 'dashed', color: '#999999'},
        itemStyle: {color: '#999999'}, markArea: ecBands(bb)}];
      if (R.bench) ser.push({name: 'S&P 500 (same deposits)', type: 'line',
        encode: {x: 0, y: 3}, showSymbol: false, sampling: 'lttb',
        lineStyle: {width: 1.3, color: ORANGE}, itemStyle: {color: ORANGE}});
      ser.push({name: 'Portfolio value (bot)', type: 'line', encode: {x: 0, y: 2},
        showSymbol: false, sampling: 'lttb', lineStyle: {width: 1.6, color: BLUE},
        itemStyle: {color: BLUE}});
      cp = echarts.init(el('chpf'));
      cp.setOption({
        animation: false, dataset: {source: rows}, legend: {top: 0},
        tooltip: {trigger: 'axis', valueFormatter: eur},
        grid: {left: 74, right: 18, top: 32, bottom: 56},
        xAxis: {type: 'time', axisLabel: {formatter: AXIS_DATE}, axisPointer: POINTER_DATE},
        yAxis: {type: 'value', scale: true, axisLabel: {formatter: function (v) { return eur(v); }}},
        dataZoom: [{type: 'inside', filterMode: 'filter'},
                   {type: 'slider', bottom: 2, height: 18,
                    labelFormatter: function (v) { return fmtDate(v); }}],
        series: ser
      });
      ecZoomReset(cp);
      el('chpf-hint').textContent = HINT_ZOOM;
    } else {
      cp.setOption({dataset: {source: rows}, series: [{markArea: ecBands(bb)}],
                    dataZoom: [{start: 0, end: 100}, {start: 0, end: 100}]});
    }
  }

  // ---- «annual return depending on the start date» chart --------------
  // Growth of the bot for 1 euro invested from the first day: in every
  // session it returns like the asset of the previous day's target
  // (cash = 0%), exactly like the simulator. From any date: G[end] / G[start].
  function chartCAGR() {
    if (!EC || !el('chcagr')) return;
    var G = new Float64Array(N); G[0] = 1;
    for (var i = 1; i < N; i++) {
      var t = TG.charAt(i - 1);
      G[i] = G[i - 1] * (t === '1' ? ON[i] / ON[i - 1] : t === '2' ? OFF[i] / OFF[i - 1] : 1);
    }
    var rows = [], e = N - 1;
    for (i = 0; i < N; i++) {
      var gg = days(i, e);
      if (gg < 365) break;                       // starts at least one year ago
      var y = 365.25 / gg, r = [D[i], Math.round((Math.pow(G[e] / G[i], y) - 1) * 10000) / 100];
      if (SPX) r.push(Math.round((Math.pow(SPX[e] / SPX[i], y) - 1) * 10000) / 100);
      rows.push(r);
      if (CAGR_FROM < 0) CAGR_FROM = i;
      CAGR_TO = i;
    }
    if (!rows.length) { el('chcagr').style.display = 'none'; return; }
    var pf = function (v) { return pctS(v); };
    var ser = [{name: 'Bot', type: 'line', encode: {x: 0, y: 1}, showSymbol: false,
      lineStyle: {width: 1.4, color: BLUE}, itemStyle: {color: BLUE},
      markLine: {silent: true, symbol: ['none', 'none'], data: [],
        lineStyle: {color: '#333', type: 'dashed', width: 1},
        label: {formatter: 'chosen start', fontSize: 10, color: '#333'}}}];
    if (SPX) ser.push({name: 'S&P 500', type: 'line', encode: {x: 0, y: 2}, showSymbol: false,
      lineStyle: {width: 1.2, color: ORANGE}, itemStyle: {color: ORANGE}});
    cc = echarts.init(el('chcagr'));
    cc.setOption({
      animation: false, dataset: {source: rows}, legend: {top: 0},
      tooltip: {trigger: 'axis', valueFormatter: pf},
      grid: {left: 54, right: 18, top: 30, bottom: 26},
      xAxis: {type: 'time', axisLabel: {formatter: AXIS_DATE}, axisPointer: POINTER_DATE},
      yAxis: {type: 'value', axisLabel: {formatter: '{value}%'}},
      series: ser
    });
    cc.getZr().on('click', function (ev) {                // click = choose that start
      var px = [ev.offsetX, ev.offsetY];
      if (!cc.containPixel({gridIndex: 0}, px)) return;
      var d = new Date(cc.convertFromPixel({gridIndex: 0}, px)[0]);
      go(findIdx(d.getFullYear() + '-' + z2(d.getMonth() + 1) + '-' + z2(d.getDate())));
    });
    window.addEventListener('resize', function () { cc.resize(); });
  }
  function lineCAGR(s) {
    if (!cc) return;
    var inside = s >= CAGR_FROM && s <= CAGR_TO;
    cc.setOption({series: [{markLine: {data: inside ? [{xAxis: D[s]}] : []}}]});
  }

  // ---- slider ----------------------------------------------------------
  var rng = el('sim-range'), cur = -1, pending = -1, raf = 0;
  function render(s) {
    cur = s; rng.value = s;
    var R = simulate(s), M = metrics(R);
    el('sim-date').textContent = fmtDate(D[s]);
    var real = (s === START0);
    el('sim-info').textContent = real ? 'real date of your first deposit (contributions.csv)'
      : 'simulation: contributions.csv is not touched';
    el('simbox').classList.toggle('moved', !real);
    el('sim-outcome').innerHTML = htmlOutcome(R, M);
    el('pf-chips').innerHTML = htmlChips(R, M);
    el('pf-ntrades').textContent = M.n;
    el('pf-table').innerHTML = htmlTable(R);
    if (el('pf-metrics')) el('pf-metrics').innerHTML = htmlMetrics(M);
    chartPF(R);
    lineCAGR(s);
  }
  function go(k) {
    k = Math.max(0, Math.min(N - 1, k));
    if (k !== cur) render(k);
  }
  function shift(months) {
    var p = D[cur].split('-'), y = +p[0], m = +p[1] - 1 + months;
    y += Math.floor(m / 12); m = ((m % 12) + 12) % 12;
    go(findIdx(y + '-' + z2(m + 1) + '-' + p[2]));
  }
  rng.addEventListener('input', function () {
    pending = +rng.value;
    if (!raf) raf = requestAnimationFrame(function () { raf = 0; go(pending); });
  });
  el('sim-m12').onclick = function () { shift(-12); };
  el('sim-m1').onclick = function () { shift(-1); };
  el('sim-p1').onclick = function () { shift(1); };
  el('sim-p12').onclick = function () { shift(12); };
  el('sim-reset').onclick = function () { go(START0); };
  chartCAGR();
  render(START0);
})();
"""


# ---------------------------------------------------------------------- #
# Building the dashboard                                                  #
# ---------------------------------------------------------------------- #
def build_dashboard(p: dict, db, store, end: date | None = None,
                    out_path: str | Path = None) -> Path:
    from src.app import build_backtester, build_portfolio_simulator
    from src.data.calendar import current_trading_date
    from src.portfolio import leg_votes, consensus_target, apply_hysteresis

    end = end or current_trading_date()
    out = Path(out_path) if out_path else Path("output") / "dashboard.html"
    out.parent.mkdir(exist_ok=True)

    # 1) engine: whole history (same classes as live, ~1 second)
    res_bt = build_backtester(p, store).run(end)
    df = res_bt.timeline
    last = df.iloc[-1]
    as_of = df.index[-1].date()

    # 2) today's consensus (same functions as the portfolio)
    pf = p.get("portfolio", {})
    legs = p.get("legs", {"market": True, "volatility": True, "tdex": True})
    votes = leg_votes(df, legs, float(pf.get("tdex_vote_threshold", 4.0)),
                      bool(pf.get("invert_volatility_leg", True)),
                      str(pf.get("market_vote", "binary")))
    raw = consensus_target(votes, float(pf.get("consensus_threshold", 0.5)))
    target = apply_hysteresis(raw) if pf.get("hold_position") else raw
    tgt = str(target.iloc[-1])
    tgt_lab, tgt_col, tgt_bg = TGT_STYLE.get(tgt, TGT_STYLE["cash"])
    asset_tgt = {"risk_on": pf.get("asset_risk_on", "—"),
                 "risk_off": pf.get("asset_risk_off", "—")}.get(tgt)
    tgt_sub = (f"the capital goes into <b>{asset_tgt}</b>" if asset_tgt
               else "no consensus: stay in cash and wait")

    # 3) real P&L + S&P 500 benchmark (only if there are deposits)
    res_pf, bench = None, None
    blob_sim = None                  # data for the first-deposit slider
    contr_path = Path(pf.get("contributions_file", "contributions.csv"))
    if contr_path.exists() and pf:
        try:
            from src.portfolio import load_contributions
            on, off = pf["asset_risk_on"], pf["asset_risk_off"]
            px_on = store.get_daily_bars(on, date(2004, 1, 1), end)["close"]
            px_off = store.get_daily_bars(off, date(2004, 1, 1), end)["close"]
            contr = load_contributions(str(contr_path))
            res_pf = build_portfolio_simulator(p).run(df, px_on, px_off, contr)
            try:
                spx = store.get_daily_bars(p["vix"]["spx_symbol"],
                                           date(2004, 1, 1), end)["close"]
            except Exception:
                spx = None
            bench = (benchmark_spx(spx, contr, res_pf.equity.index)
                     if spx is not None else None)
            if len(contr) and len(res_pf.equity):
                try:
                    blob_sim = _blob_simulator(res_pf, df, px_on, px_off, spx,
                                               contr, on, off)
                except Exception:
                    blob_sim = None  # no slider, the rest of the page stays
        except Exception:
            res_pf = None            # the dashboard must never fail here

    # 4) HTML pieces ------------------------------------------------------
    quad_lab, quad_bg, quad_col = QUAD_STYLE.get(
        str(last["regime"]), (str(last["regime"]), "#eee", "#555"))

    m = {k: last.get(k) for k in ("spx", "vix", "vrp_smooth", "rsi_smooth",
                                  "tdex_smooth")}
    tdex_txt = f"{m['tdex_smooth']:.2f}" if pd.notna(m["tdex_smooth"]) else "n/a"
    v_last = votes.iloc[-1] if not votes.empty else pd.Series(dtype=float)
    in_mkt = last["pos_market"] > 0
    in_vol = last["pos_volatility"] > 0
    votes_html = "".join([
        _vote_card("MARKET leg",
                   int(v_last["market"]) if "market" in v_last else None,
                   ("in position " if in_mkt else "out of the market ")
                   + _since(df["pos_market"])),
        _vote_card("VOLATILITY leg",
                   int(v_last["volatility"]) if "volatility" in v_last else None,
                   ("long-vol active " if in_vol else "no stress ")
                   + _since(df["pos_volatility"])),
        _vote_card("TDEX leg",
                   int(v_last["tdex"]) if "tdex" in v_last else None,
                   f"smoothed z-score: {tdex_txt}"),
    ])

    s = res_bt.stats
    chips_bt = "".join([
        _chip("Strategy", f"{s['timing_multiple']:.1f}×",
              f"max DD {s['timing_max_drawdown']:.0%}"),
        _chip("Buy & Hold", f"{s['bh_multiple']:.1f}×",
              f"max DD {s['bh_max_drawdown']:.0%}"),
        _chip("Time in the market", f"{s['time_in_market_pct']}%",
              "market leg (not the portfolio)"),
        _chip("Time in volatility", f"{s['time_in_volatility_pct']}%",
              "volatility leg"),
        _chip("Period", f"{s['days']} days", f"{_d(s['start'])} → {_d(s['end'])}"),
    ])
    reg_chips = "".join(
        f'<span class="tag" style="background:{QUAD_STYLE.get(k, ("", "#eee", ""))[1]}">'
        f'{QUAD_STYLE.get(k, (k,))[0]}: {v}%</span>'
        for k, v in s["regime_distribution_pct"].items())

    # exposure: prefer the simulator's series (prices available for both
    # assets); otherwise the one computed here on the timeline.
    if res_pf is not None and len(res_pf.target):
        tgt_exp, exp_source = res_pf.target, (
            "Series taken from the portfolio simulator, the same one that generates "
            "the trades below.")
    else:
        tgt_exp, exp_source = target, (
            "Series rebuilt from the legs' consensus over the whole timeline.")
    exp_section = _exposure_card(
        _exposure(tgt_exp), str(pf.get("asset_risk_on", "^SPX")),
        str(pf.get("asset_risk_off", "TLT")), exp_source)

    # Cash net of inflation (chart only, does not touch the signals):
    # price index from FRED, local copy in data/ for offline use.
    from src.data.inflation import (DEFAULT_SERIES, price_index,
                                    series_name, purchasing_power)
    infl_series = str((p.get("dashboard") or {}).get("inflation_series")
                      or DEFAULT_SERIES).strip().upper()
    infl_name = series_name(infl_series)
    real_cash = None
    try:
        folder = Path(p.get("data", {}).get("database",
                                            "data/market.duckdb")).parent
        ind = price_index(infl_series, folder)
        if ind is not None:
            real_cash = purchasing_power(ind, res_bt.equity.index)
    except Exception:
        real_cash = None
    if real_cash is not None and real_cash.notna().any():
        cr_end = float(real_cash.dropna().iloc[-1])
        cr_txt = f"{cr_end:.2f}"
        cash_text = (
            'Dark grey line: cash held still (stays at 1×). '
            'Dashed: the same cash net of '
            f'{infl_name} inflation — 1 held in cash since '
            f'{_d(res_bt.equity.index[0])} today buys what {cr_txt} bought '
            f'back then ({(cr_end - 1) * 100:+.0f}% purchasing power). ')
    else:
        real_cash = None
        cash_text = (
            'Dark grey line: cash held still (stays at 1×). '
            f'<i>Inflation line not available: an internet connection is needed to '
            f'download the {infl_series} series from FRED.</i> ')

    charts_bt = (
        '<p class="sub" style="margin:8px 0 0">Backtest over the whole history '
        '— capital (log scale; green bands = in the market) and drawdown '
        '(red = SPX and strategy both in drawdown; grey = SPX only; '
        'blue = strategy only). '
        'Purple line: TLT (long-term bonds, rebased to 1× at the start of the '
        'history). ' + cash_text + '<span id="chbt-hint"></span></p>'
        '<div id="chbt" class="ichart" style="height:520px"></div>')

    pf_section = ""
    if res_pf is not None:
        sp = res_pf.stats
        pnl = sp["total_pnl"]
        pnl_col = "#157f3d" if pnl >= 0 else "#a13030"
        chips_pf = [
            _chip("Deposited", _fmt_eur(sp["total_deposited"])),
            _chip("Value today (bot)", _fmt_eur(sp["final_value"])),
            _chip("Profit/loss",
                  f'<span style="color:{pnl_col}">{pnl:+,.0f} €</span>',
                  _num(sp["return_on_deposits_pct"], "+.1f", "% on deposits")),
            _chip("Strategy (TWR)", f"{sp['twr_multiple']}×"),
            _chip("Trades", f"{sp['n_trades']}",
                  f"winning {_num(sp['win_rate_pct'], '.0f', '%')} · "
                  f"PF {_num(sp['profit_factor'], '.2f')}"),
        ]
        if bench is not None and len(bench):
            b_end = float(bench.iloc[-1])
            diff = sp["final_value"] - b_end
            d_col = "#157f3d" if diff >= 0 else "#a13030"
            chips_pf.insert(2, _chip(
                "S&P 500 (same deposits)", _fmt_eur(b_end),
                f'<span style="color:{d_col}">bot {diff:+,.0f} € '
                f'vs the index</span>'))
        charts_pf = (
            '<p class="sub" style="margin:8px 0 0">Real P&L — bot vs S&P '
            '500 with the same deposits (bands: green = risk-on, orange = '
            'risk-off). <span id="chpf-hint"></span></p>'
            '<div id="chpf" class="ichart" style="height:400px"></div>')

        rows = ""
        if not res_pf.trades.empty:
            for _, t in res_pf.trades.iloc[::-1].iterrows():
                exit_ = ("<i>open</i>" if t.get("is_open")
                         else _d(t["exit_date"]))
                pcol = "#157f3d" if (t["pnl"] or 0) >= 0 else "#a13030"
                reg = QUAD_STYLE.get(str(t.get("regime_in", "")), None)
                reg_txt = reg[0] if reg else str(t.get("regime_in", "—"))
                rows += (f"<tr><td>{_d(t['entry_date'])}</td>"
                         f"<td>{exit_}</td><td>{t['asset']}</td>"
                         f"<td>{t['days']}</td>"
                         f"<td style='color:{pcol}'>{(t['pnl'] or 0):+,.0f} €</td>"
                         f"<td style='color:{pcol}'>{(t['return_pct'] or 0):+.1%}</td>"
                         f"<td>{(t['mfe_pct'] or 0):+.1%}</td>"
                         f"<td>{(t['mae_pct'] or 0):+.1%}</td>"
                         f"<td>{reg_txt}</td></tr>")
        table = (f'<div class="tablewrap"><table><thead><tr><th>Entry</th>'
                 f'<th>Exit</th><th>Asset</th><th>Days</th><th>P&L</th>'
                 f'<th>P&L %</th><th>MFE</th><th>MAE</th>'
                 f'<th>Entry regime</th></tr></thead>'
                 f'<tbody>{rows}</tbody></table></div>'
                 if rows else "<p class='sub'>No trades yet.</p>")

        # «date of the first deposit» slider (the logic is in SIM_JS)
        box_sim = ""
        if blob_sim is not None:
            import bisect
            ds = blob_sim["date"]
            k0 = min(bisect.bisect_left(ds, blob_sim["plan"][0][0]), len(ds) - 1)
            box_sim = f"""
    <div class="simbox" id="simbox">
      <div class="simrow">
        <span class="lab">First deposit on</span>
        <b class="simdate" id="sim-date">{_d(ds[k0])}</b>
        <span class="sub" id="sim-info">real date of your first deposit
          (contributions.csv)</span>
        <span class="simbtn">
          <button type="button" class="pbtn" id="sim-m12">− 1 year</button>
          <button type="button" class="pbtn" id="sim-m1">− 1 month</button>
          <button type="button" class="pbtn" id="sim-p1">+ 1 month</button>
          <button type="button" class="pbtn" id="sim-p12">+ 1 year</button>
          <button type="button" class="pbtn" id="sim-reset">Real date</button>
        </span>
      </div>
      <input type="range" id="sim-range" min="0" max="{len(ds) - 1}" step="1"
             value="{k0}" aria-label="Date of the first deposit">
      <div class="simscale"><span>{_d(ds[0])}</span><span>{_d(ds[-1])}</span></div>
      <div class="simoutcome" id="sim-outcome"></div>
      <p class="sub" style="margin:10px 0 0">Compound annual return depending
      on the start date (a single deposit held until the last data point;
      starts at least one year ago). Click on the chart = start from that date.</p>
      <div id="chcagr" class="ichart" style="height:230px"></div>
      <p class="sub" style="margin:6px 0 0">Drag the slider or use the buttons:
      boxes, chart, trade log and metrics of this section are
      recomputed with the same rules as the simulator, as if the first
      deposit had been made on that day (if contributions.csv contains
      others, they all move together). The contributions.csv file is not
      modified.</p>
    </div>"""
        pf_section = f"""
  <div class="pf-wrap">
  <div class="card pf-main">
    <h2>Real P&L — the bot against the S&P 500 (same deposits)</h2>
    {box_sim}
    <div class="chips" id="pf-chips">{''.join(chips_pf)}</div>
    {charts_pf}
    <p class="sub">The orange line is the honest comparison: the same euros,
    deposited on the same days, put on the S&P 500 and never touched again
    (price index, dividends excluded).</p>
    <h3>Full trade log (<span id="pf-ntrades">{sp['n_trades']}</span> closed)</h3>
    <div id="pf-table">{table}</div>
    <p class="sub">MFE = maximum favourable excursion during the trade;
    MAE = maximum adverse excursion. They help to understand whether the exits
    leave gains on the table or cut losses too late.</p>
  </div>
  {_metrics_panel(res_pf)}
  </div>"""
    else:
        pf_section = """
  <div class="card">
    <h2>Real P&L — the bot against the S&P 500</h2>
    <p class="sub">No deposits recorded: use menu option 6
    («Add a deposit») and this section will fill in by itself.</p>
  </div>"""

    demo = bool(db.coverage()["synthetic"].any())
    banner = ('<div class="demo">⚠ Synthetic DEMO data: these numbers are NOT '
              'real market data.</div>' if demo else "")

    notes = "".join(f"<li>{n}</li>" for n in
                    res_bt.notes + (res_pf.notes if res_pf is not None and
                                    hasattr(res_pf, "notes") else []))

    gen = datetime.now().strftime("%d/%m/%Y %H:%M")

    # data of the interactive charts (compact JSON) + library from the CDN
    # TLT in the backtest chart: price series from the DB (it is one of the
    # 40 regime ETFs, so it is always there); if missing, no line is drawn.
    try:
        tlt_px = store.get_daily_bars("TLT", date(2004, 1, 1), end)["close"]
    except Exception:
        tlt_px = None
    data_bt = _blob_backtest(res_bt, tlt_px, real_cash)
    data_bt["infl_name"] = infl_name
    blob_bt = json.dumps(data_bt, separators=(",", ":"))
    blob_pf = (json.dumps(blob_sim, separators=(",", ":"))
               if res_pf is not None and blob_sim is not None else "null")
    chart_scripts = (
        '<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/'
        'echarts.min.js"></script>\n'
        f"<script>window.BT={blob_bt};window.PF={blob_pf};</script>\n"
        f"<script>{COMMON_JS}</script>\n"
        f"<script>{ECHARTS_JS}</script>\n"
        f"<script>{SIM_JS}</script>")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tactical Bot — Dashboard</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, "Segoe UI", Roboto, sans-serif; margin: 0;
         background: #f4f5f7; color: #222; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 22px 16px 40px; }}
  header h1 {{ font-size: 20px; margin: 0 0 2px; }}
  header .sub {{ color: #666; font-size: 13px; }}
  .demo {{ background: #fff3cd; border: 1px solid #eedd88; color: #7a5d00;
          padding: 8px 12px; border-radius: 8px; margin: 12px 0;
          font-size: 13px; }}
  .hero {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 16px 0; }}
  .signal {{ flex: 2 1 320px; border-radius: 14px; padding: 18px 22px;
            background: {tgt_bg}; border: 1px solid rgba(0,0,0,.06); }}
  .signal .big {{ font-size: 34px; font-weight: 800; color: {tgt_col};
                 letter-spacing: .5px; }}
  .signal .sub {{ font-size: 14px; color: #444; margin-top: 4px; }}
  .regime {{ flex: 1 1 200px; border-radius: 14px; padding: 18px 22px;
            background: {quad_bg}; border: 1px solid rgba(0,0,0,.06); }}
  .regime .big {{ font-size: 20px; font-weight: 700; color: {quad_col}; }}
  .lab {{ font-size: 11px; text-transform: uppercase; letter-spacing: .8px;
         color: #777; }}
  .card {{ background: #fff; border-radius: 14px; padding: 18px 22px;
          margin: 14px 0; box-shadow: 0 1px 3px rgba(0,0,0,.07); }}
  .card h2 {{ font-size: 15px; margin: 0 0 10px; }}
  .card h3 {{ font-size: 13px; margin: 14px 0 6px; color: #555; }}
  .card img {{ width: 100%; height: auto; border-radius: 8px; margin-top: 8px; }}
  .pf-wrap {{ display: flex; gap: 14px; align-items: flex-start;
             flex-wrap: wrap; }}
  .pf-main {{ flex: 3 1 540px; min-width: 0; }}
  .card.metrics {{ flex: 1 1 250px; padding: 0; overflow: hidden;
                  align-self: stretch; }}
  .metrics .mtitle {{ background: {BLUE}; color: #fff; font-weight: 700;
                     font-size: 13px; padding: 12px 16px; letter-spacing: .3px; }}
  table.mtab {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
  table.mtab td {{ padding: 8px 14px; border-bottom: 1px solid #eef0f2; }}
  table.mtab tr:nth-child(even) {{ background: #f7f8fa; }}
  table.mtab .mk {{ color: #555; }}
  table.mtab .mv {{ text-align: right; font-weight: 700; color: #222;
                   white-space: nowrap; }}
  .metrics p.sub {{ padding: 0 14px 14px; }}
  .votes {{ display: grid; grid-template-columns: repeat(auto-fit,
           minmax(220px, 1fr)); gap: 12px; }}
  .vote {{ border: 1px solid #ececec; border-radius: 10px; padding: 12px 14px; }}
  .vote .val {{ font-size: 18px; font-weight: 700; margin: 2px 0; }}
  .vote .sub, .chip .sub, p.sub {{ font-size: 12px; color: #777; }}
  .chips {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }}
  .chip {{ background: #f7f8fa; border: 1px solid #ececec; border-radius: 10px;
          padding: 8px 14px; min-width: 120px; }}
  .chip .val {{ font-size: 17px; font-weight: 700; }}
  .kpis {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px;
          color: #555; margin-top: 10px; }}
  .kpis b {{ color: #222; }}
  .tag {{ display: inline-block; border-radius: 20px; padding: 3px 10px;
         font-size: 12px; margin: 2px 4px 2px 0; }}
  .periods {{ margin-top: 10px; }}
  .pbtn {{ border: 1px solid #d8dbe0; background: #f7f8fa; color: #444;
          border-radius: 8px; padding: 5px 14px; margin-right: 6px;
          font-size: 12.5px; cursor: pointer; }}
  .pbtn.active {{ background: {BLUE}; border-color: {BLUE}; color: #fff;
                 font-weight: 600; }}
  .expbar {{ height: 26px; border-radius: 6px; border: 1px solid #e6e8ea;
            width: 100%; }}
  .nowbox {{ background: #f7f8fa; border-radius: 8px; padding: 10px 14px;
            margin: 10px 0 4px; font-size: 13px; color: #444; }}
  .tablewrap {{ max-height: 380px; overflow-y: auto; border: 1px solid #f0f0f0;
               border-radius: 8px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 10px;
           border-bottom: 1px solid #f0f0f0; }}
  th {{ color: #777; font-weight: 600; font-size: 12px; position: sticky;
       top: 0; background: #fff; }}
  .simbox {{ background: #f7f9fc; border: 1px solid #e3e8f0;
            border-left: 4px solid {BLUE}; border-radius: 10px;
            padding: 12px 14px; margin: 4px 0 14px; }}
  .simbox.moved {{ border-left-color: {ORANGE}; background: #fffaf3; }}
  .simrow {{ display: flex; flex-wrap: wrap; align-items: center;
            gap: 6px 12px; }}
  .simdate {{ font-size: 19px; }}
  .simrow .sub {{ font-size: 12px; color: #777; }}
  .simbtn {{ margin-left: auto; display: flex; flex-wrap: wrap; gap: 6px; }}
  .simbtn .pbtn {{ margin: 0; padding: 4px 10px; }}
  #sim-range {{ width: 100%; margin: 12px 0 2px; accent-color: {BLUE};
               cursor: pointer; }}
  .simscale {{ display: flex; justify-content: space-between; font-size: 11px;
              color: #888; }}
  .simoutcome {{ font-size: 13px; color: #333; margin: 8px 0 0;
                line-height: 1.6; }}
  footer {{ color: #999; font-size: 11.5px; margin-top: 18px; }}
  footer ul {{ margin: 6px 0 0 18px; padding: 0; }}
</style>
<script>
function showPeriod(group, key, btn) {{
  document.querySelectorAll('img[data-g="' + group + '"]').forEach(function (i) {{
    i.style.display = (i.dataset.k === key) ? '' : 'none';
  }});
  btn.parentElement.querySelectorAll('.pbtn').forEach(function (b) {{
    b.classList.toggle('active', b === btn);
  }});
}}
</script></head><body><div class="wrap">
<header>
  <h1>Tactical Trading Bot · V1 — Dashboard</h1>
  <div class="sub">Market data as of <b>{_d(as_of)}</b> · page generated on {gen}
  · regenerated at every signal computation (option 3)</div>
</header>
{banner}
<div class="hero">
  <div class="signal">
    <div class="lab">Today's consensus</div>
    <div class="big">{tgt_lab}</div>
    <div class="sub">{tgt_sub}</div>
  </div>
  <div class="regime">
    <div class="lab">Current regime</div>
    <div class="big">{quad_lab}</div>
    <div class="kpis" style="margin-top:8px">
      <span>SPX <b>{m['spx']:.0f}</b></span> <span>VIX <b>{m['vix']:.2f}</b></span>
    </div>
  </div>
</div>
<div class="card">
  <h2>The legs' votes (the majority decides)</h2>
  <div class="votes">{votes_html}</div>
  <div class="kpis">
    <span>Smoothed VRP <b>{m['vrp_smooth']:.2f}</b></span>
    <span>Smoothed RSI <b>{m['rsi_smooth']:.2f}</b></span>
    <span>TDEX z <b>{tdex_txt}</b></span>
  </div>
</div>
<div class="card">
  <h2>How the strategy performs (backtest, same rules as live)</h2>
  <div class="chips">{chips_bt}</div>
  <div>{reg_chips}</div>
  {charts_bt}
</div>
{exp_section}
{pf_section}
<footer>
  Method notes:<ul>{notes}</ul>
  Detailed reports (if generated): output/backtest_report.html ·
  output/portfolio_report.html
</footer>
{chart_scripts}
</div></body></html>"""

    out.write_text(html, encoding="utf-8")
    return out
