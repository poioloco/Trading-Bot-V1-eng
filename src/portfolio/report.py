"""Portfolio HTML report: capital curve vs deposits, target bands, KPIs,
trade log. A single file you can open with a double click."""
from __future__ import annotations

import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from . import CASH, OFF, ON, PortfolioResult

BAND = {ON: "#d6f5d6", OFF: "#ffe0c2", CASH: "#ececec"}


def _chart_png(res: PortfolioResult) -> str:
    eq = res.equity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), height_ratios=[2, 1], sharex=True)

    # target bands
    tgt = res.target.reindex(eq.index).ffill()
    start = eq.index[0]
    cur = tgt.iloc[0]
    for i in range(1, len(tgt)):
        if tgt.iloc[i] != cur:
            ax1.axvspan(start, eq.index[i], color=BAND.get(cur, "#eee"), alpha=0.5, lw=0)
            start = eq.index[i]
            cur = tgt.iloc[i]
    ax1.axvspan(start, eq.index[-1], color=BAND.get(cur, "#eee"), alpha=0.5, lw=0)

    ax1.plot(eq.index, eq["value"], color="#16527a", lw=1.3, label="Portfolio value")
    ax1.plot(eq.index, eq["deposits"], color="#888", lw=1.0, ls="--", label="Deposited (cumulative)")
    ax1.fill_between(eq.index, eq["deposits"], eq["value"],
                     where=(eq["value"] >= eq["deposits"]), color="#16527a", alpha=0.12)
    ax1.set_ylabel("Euro")
    ax1.set_title("Portfolio vs deposits  ·  bands: green=risk-on, grey=cash, orange=risk-off",
                  fontsize=11)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.15)

    ax2.plot(eq.index, eq["twr"], color="#1f6fb2", lw=1.2, label="Strategy return (TWR)")
    ax2.axhline(1.0, color="#bbb", lw=0.8)
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


def _eur(x) -> str:
    return "—" if x is None else f"{x:,.0f} €".replace(",", ".")


def _trade_rows(tdf: pd.DataFrame, limit: int = 60) -> str:
    if tdf.empty:
        return '<tr><td colspan="8">No trades in the period.</td></tr>'
    rows = []
    for _, t in tdf.tail(limit).iloc[::-1].iterrows():
        pnl = t["pnl"]
        col = "#159c4a" if (pnl is not None and pnl > 0) else "#c0392b"
        ret = "" if t["return_pct"] is None else f"{t['return_pct'] * 100:+.1f}%"
        rows.append(
            f"<tr><td>{t['asset']}</td>"
            f"<td>{pd.Timestamp(t['entry_date']).date()}</td>"
            f"<td>{pd.Timestamp(t['exit_date']).date() if pd.notna(t['exit_date']) else 'open'}</td>"
            f"<td>{'' if t['days'] is None else int(t['days'])}</td>"
            f"<td>{t['regime_in']}</td>"
            f"<td style='color:{col}'>{_eur(pnl)}</td>"
            f"<td style='color:{col}'>{ret}</td>"
            f"<td>{t['mfe_pct'] * 100:+.0f}% / {t['mae_pct'] * 100:+.0f}%</td></tr>"
        )
    return "".join(rows)


def build_html_report(res: PortfolioResult, path: str) -> str:
    s = res.stats
    png = _chart_png(res)
    notes = "".join(f"<li>{n}</li>" for n in res.notes)
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Portfolio – Tactical Trading Bot V1</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1c1f23}}
 .wrap{{max-width:1000px;margin:0 auto;padding:28px 22px 60px}}
 h1{{font-size:22px;margin:0 0 2px}} .sub{{color:#6b7280;margin:0 0 22px;font-size:14px}}
 .card{{background:#fff;border:1px solid #e6e8eb;border-radius:12px;padding:18px 20px;margin:0 0 18px}}
 .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 18px}}
 .kpi{{background:#fff;border:1px solid #e6e8eb;border-radius:12px;padding:14px 16px}}
 .kpi .n{{font-size:21px;font-weight:650}} .kpi .l{{color:#6b7280;font-size:12px;margin-top:3px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 td,th{{padding:6px 8px;border-bottom:1px solid #eef0f2;text-align:left}}
 th{{color:#6b7280;font-weight:600}}
 img{{width:100%;border-radius:8px}}
 .note{{background:#fff8e6;border:1px solid #f3e2a8;border-radius:10px;padding:12px 16px;font-size:13px;color:#5b4b16}}
 .note ul{{margin:6px 0 0;padding-left:18px}} .pos{{color:#159c4a}} .neg{{color:#c0392b}}
</style></head><body><div class="wrap">
<h1>Simulated portfolio – Tactical Trading Bot V1</h1>
<p class="sub">{s['start']} → {s['end']} · generated on {datetime.now():%d/%m/%Y %H:%M}</p>

<div class="grid">
 <div class="kpi"><div class="n">{_eur(s['total_deposited'])}</div><div class="l">total deposited</div></div>
 <div class="kpi"><div class="n">{_eur(s['final_value'])}</div><div class="l">final value</div></div>
 <div class="kpi"><div class="n {'pos' if s['total_pnl']>=0 else 'neg'}">{_eur(s['total_pnl'])}</div><div class="l">profit/loss</div></div>
 <div class="kpi"><div class="n">{s['twr_multiple']}×</div><div class="l">strategy return (TWR)</div></div>
 <div class="kpi"><div class="n">{s['n_trades']}</div><div class="l">closed trades</div></div>
 <div class="kpi"><div class="n">{s['win_rate_pct']}%</div><div class="l">winning trades</div></div>
 <div class="kpi"><div class="n">{s['profit_factor']}</div><div class="l">profit factor</div></div>
 <div class="kpi"><div class="n">{s['time_risk_on_pct']}/{s['time_cash_pct']}/{s['time_risk_off_pct']}%</div><div class="l">time risk-on/cash/risk-off</div></div>
</div>

<div class="card"><img src="data:image/png;base64,{png}" alt="portfolio chart"></div>

<div class="card">
 <h3 style="margin:0 0 10px">Trade log (most recent)</h3>
 <table>
  <tr><th>Asset</th><th>Entry</th><th>Exit</th><th>Days</th><th>Regime</th><th>P&amp;L</th><th>Return</th><th>MFE/MAE</th></tr>
  {_trade_rows(res.trades)}
 </table>
</div>

<div class="note"><b>Please read</b><ul>{notes}
 <li>IN-SAMPLE results, without costs or slippage: the model's behaviour, not a promise of returns.</li>
</ul></div>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
