"""Backtest HTML report: chart (matplotlib, embedded PNG) + statistics
table + method notes. A single file you can open with a double click."""
from __future__ import annotations

import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from . import BacktestResult

# colours for the 4 quadrants (regime)
QUAD_COLORS = {
    "disinflation": "#cfe8ff", "reflation": "#d6f5d6",
    "stagflation": "#ffe6c7", "deflation": "#f3d4d4",
}
QUAD_LABEL = {
    "disinflation": "Q1 Disinflation", "reflation": "Q2 Reflation",
    "stagflation": "Q3 Stagflation", "deflation": "Q4 Deflation",
}


def _chart_png(res: BacktestResult, long_asset: str) -> str:
    df, eq = res.timeline, res.equity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1], sharex=True)

    # ---- panel 1: SPX (log) + regime bands + markers ------------------
    ax1.set_yscale("log")
    ax1.plot(df.index, df["spx"], color="#222", lw=0.9, label="SPX")
    # regime bands: contiguous blocks of the same quad
    block_start = df.index[0]
    cur = df["regime"].iloc[0]
    for i in range(1, len(df)):
        if df["regime"].iloc[i] != cur:
            ax1.axvspan(block_start, df.index[i], color=QUAD_COLORS.get(cur, "#eee"), alpha=0.5, lw=0)
            block_start = df.index[i]
            cur = df["regime"].iloc[i]
    ax1.axvspan(block_start, df.index[-1], color=QUAD_COLORS.get(cur, "#eee"), alpha=0.5, lw=0)
    # market leg markers
    bspx = df[df["buy_spx"]]
    sspx = df[df["sell_spx"]]
    ax1.scatter(bspx.index, bspx["spx"], marker="^", s=22, color="#159c4a", zorder=5,
                label=f"BUY {long_asset}")
    ax1.scatter(sspx.index, sspx["spx"], marker="v", s=22, color="#c0392b", zorder=5,
                label=f"SELL {long_asset}")
    ax1.set_title("SPX, market regime and market-leg signals", fontsize=12)
    ax1.set_ylabel("SPX (log scale)")
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.grid(True, which="both", alpha=0.15)

    # ---- panel 2: timing curves (illustrative) -------------------------
    ax2.plot(eq.index, eq["buy_and_hold_spx"], color="#888", lw=1.0, label="Buy & Hold SPX")
    ax2.plot(eq.index, eq["timing_market"], color="#1f6fb2", lw=1.2,
             label="Market-leg timing (illustrative)")
    ax2.set_ylabel("Growth of €1")
    ax2.set_yscale("log")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, which="both", alpha=0.15)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_html_report(res: BacktestResult, path: str, long_asset: str = "TQQQ",
                      vol_asset: str = "UVIX") -> str:
    png = _chart_png(res, long_asset)
    s = res.stats
    reg_rows = "".join(
        f"<tr><td>{QUAD_LABEL.get(k, k)}</td><td>{v}%</td></tr>"
        for k, v in sorted(s["regime_distribution_pct"].items(), key=lambda kv: -kv[1])
    )
    ev = s["events"]
    notes = "".join(f"<li>{n}</li>" for n in res.notes)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Backtest – Tactical Trading Bot V1</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1c1f23}}
 .wrap{{max-width:980px;margin:0 auto;padding:28px 22px 60px}}
 h1{{font-size:22px;margin:0 0 2px}} .sub{{color:#6b7280;margin:0 0 22px;font-size:14px}}
 .card{{background:#fff;border:1px solid #e6e8eb;border-radius:12px;padding:18px 20px;margin:0 0 18px}}
 .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 18px}}
 .kpi{{background:#fff;border:1px solid #e6e8eb;border-radius:12px;padding:14px 16px}}
 .kpi .n{{font-size:22px;font-weight:650}} .kpi .l{{color:#6b7280;font-size:12px;margin-top:3px}}
 table{{border-collapse:collapse;width:100%;font-size:14px}}
 td,th{{padding:7px 8px;border-bottom:1px solid #eef0f2;text-align:left}}
 th{{color:#6b7280;font-weight:600}}
 img{{width:100%;border-radius:8px}}
 .note{{background:#fff8e6;border:1px solid #f3e2a8;border-radius:10px;padding:12px 16px;font-size:13px;color:#5b4b16}}
 .note ul{{margin:6px 0 0;padding-left:18px}}
</style></head><body><div class="wrap">
<h1>Backtest – Tactical Trading Bot V1</h1>
<p class="sub">Period {s['start']} → {s['end']} · {s['days']} trading days ·
generated on {datetime.now():%d/%m/%Y %H:%M}</p>

<div class="grid">
 <div class="kpi"><div class="n">{ev['buy_spx']}/{ev['sell_spx']}</div><div class="l">BUY/SELL {long_asset}</div></div>
 <div class="kpi"><div class="n">{ev['buy_vix']}/{ev['sell_vix']}</div><div class="l">BUY/SELL {vol_asset}</div></div>
 <div class="kpi"><div class="n">{s['time_in_market_pct']}%</div><div class="l">time in the market</div></div>
 <div class="kpi"><div class="n">{s['time_in_volatility_pct']}%</div><div class="l">time in volatility</div></div>
</div>

<div class="card"><img src="data:image/png;base64,{png}" alt="backtest chart"></div>

<div class="card">
 <h3 style="margin:0 0 10px">Market-leg timing (illustrative, SPX proxy)</h3>
 <table>
  <tr><th></th><th>Buy &amp; Hold SPX</th><th>Market timing</th></tr>
  <tr><td>Growth of €1</td><td>{s['bh_multiple']:.2f}×</td><td>{s['timing_multiple']:.2f}×</td></tr>
  <tr><td>Max drawdown</td><td>{_fmt_pct(s['bh_max_drawdown'])}</td><td>{_fmt_pct(s['timing_max_drawdown'])}</td></tr>
 </table>
</div>

<div class="card">
 <h3 style="margin:0 0 10px">Regime distribution</h3>
 <table><tr><th>Regime</th><th>% of the period</th></tr>{reg_rows}</table>
</div>

<div class="note"><b>Method notes</b><ul>{notes}</ul></div>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
