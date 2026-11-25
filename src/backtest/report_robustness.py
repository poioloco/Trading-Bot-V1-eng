"""HTML report of the robustness battery.

No calculations in here: it takes the dictionaries produced by
`robustness.py` and lays them out, with next to each number the explanation
of how to read it and a TRAFFIC LIGHT (green / yellow / red) with the
declared criterion. The traffic light is not an opinion: it is the
threshold written under the number.
"""
from __future__ import annotations

import html
import json
from datetime import datetime

import pandas as pd

CSS = """
:root{--green:#1a7f37;--yellow:#9a6700;--red:#cf222e;--bd:#d0d7de;--bg:#f6f8fa}
*{box-sizing:border-box}
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:0;
     padding:28px;color:#1f2328;max-width:1180px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:38px 0 6px;padding-bottom:6px;border-bottom:2px solid var(--bd)}
h3{font-size:16px;margin:22px 0 6px}
.sub{color:#656d76;margin:0 0 18px}
p.why{background:var(--bg);border-left:3px solid #8250df;padding:9px 12px;
     margin:8px 0 14px;font-size:13.5px;color:#424a53}
table{border-collapse:collapse;width:100%;margin:10px 0 6px;font-size:13.5px}
th,td{border:1px solid var(--bd);padding:5px 8px;text-align:right}
th{background:var(--bg);text-align:right;font-weight:600}
td:first-child,th:first-child{text-align:left}
tr.hl td{background:#fff8c5;font-weight:600}
.light{display:inline-block;padding:2px 9px;border-radius:11px;color:#fff;
     font-size:12px;font-weight:600;margin-right:6px}
.light.green{background:var(--green)}.light.yellow{background:var(--yellow)}.light.red{background:var(--red)}
.box{border:1px solid var(--bd);border-radius:6px;padding:12px 14px;margin:12px 0}
.box.green{border-left:5px solid var(--green)}.box.yellow{border-left:5px solid var(--yellow)}
.box.red{border-left:5px solid var(--red)}
.crit{font-size:12.5px;color:#656d76;margin:4px 0 0}
.big{font-size:22px;font-weight:700}
ul{margin:6px 0 6px 20px;padding:0}li{margin:3px 0}
footer{margin-top:40px;color:#656d76;font-size:12px;border-top:1px solid var(--bd);
     padding-top:12px}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _light(level: str, text: str) -> str:
    return f'<span class="light {level}">{_e(text)}</span>'


def _tab(df: pd.DataFrame, highlight=None) -> str:
    if df is None or len(df) == 0:
        return "<p><i>no data</i></p>"
    cols = list(df.columns)
    out = ["<table><thead><tr>"] + [f"<th>{_e(c)}</th>" for c in cols] + ["</tr></thead><tbody>"]
    for _, r in df.iterrows():
        cl = ' class="hl"' if highlight and highlight(r) else ""
        out.append(f"<tr{cl}>")
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = f"{v:,.2f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
            out.append(f"<td>{_e('' if v is None or (isinstance(v,float) and pd.isna(v)) else v)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _box(level: str, title: str, body: str, criterion: str = "") -> str:
    c = f'<p class="crit">Criterion: {_e(criterion)}</p>' if criterion else ""
    return (f'<div class="box {level}">{_light(level, title)}{body}{c}</div>')


# ---------------------------------------------------------------------- #
def build_html(res: dict) -> str:
    """`res` is the full dictionary produced by `cmd_robustness`."""
    o = [f"<!doctype html><meta charset='utf-8'><title>Robustness</title>"
         f"<style>{CSS}</style>",
         "<h1>Robustness of the strategy</h1>",
         f"<p class='sub'>{_e(res['meta']['start'])} → {_e(res['meta']['end'])} · "
         f"{res['meta']['sessions']:,} sessions · costs {_e(res['meta']['costs'])} · "
         f"generated on {_e(res['meta']['generated'])}</p>"]

    # ---------------- summary ---------------- #
    o.append("<h2>In short</h2>")
    for v in res.get("verdicts", []):
        o.append(_box(v["level"], v["title"], f"<p>{v['text']}</p>", v.get("criterion", "")))

    # ---------------- 0. data coverage ---------------- #
    o.append("<h2>0 · Data coverage</h2>")
    o.append("<p class='why'>Before measuring skill, measure "
             "VISION. Missing data produces abstention, and an abstention "
             "during a crash looks like a brilliant exit without being one. "
             "Here you can see since when each input really exists.</p>")
    cov = res["coverage"]
    o.append(_tab(pd.DataFrame([{"series": k, **v} for k, v in cov.items()
                                if isinstance(v, dict)])))
    o.append(f"<p><b>History with all inputs present: from "
             f"{_e(cov.get('history_fully_covered_from'))}.</b></p>")
    if "windows" in res:
        o.append("<h3>The same configuration over different windows</h3>")
        o.append("<p class='why'>If the numbers change NATURE going from the "
                 "covered period to the uncovered one, the overall result is "
                 "not a property of the strategy: it is an artefact of the "
                 "period.</p>")
        o.append(_tab(pd.DataFrame(res["windows"])))

    # ---------------- 1. costs and latency ---------------- #
    o.append("<h2>1 · Costs, latency, leverage</h2>")
    o.append("<p class='why'>In the <b>bps</b> column what matters is the slope "
             "(how much CAGR each basis point eats: proportional to the number of "
             "trades). In the <b>latency</b> column what matters is the collapse: a "
             "healthy strategy loses little going from T+1 to T+2; if instead the "
             "numbers fall apart, the signal lives inside one day of noise. "
             "<b>Leverage</b> matters because the real asset is 3x leveraged, "
             "not SPX: the unleveraged proxy hides volatility drag.</p>")
    o.append(_tab(pd.DataFrame(res["costs_latency"])))

    # ---------------- 2. walk-forward ---------------- #
    o.append("<h2>2 · Anchored walk-forward</h2>")
    o.append("<p class='why'>Each fold recalibrates on the past only and is "
             "judged on the next, unseen block. The two questions: does the "
             "calibration beat the base configuration out of sample? And "
             "how much does it DEGRADE going from in sample to out of sample? "
             "A systematic degradation is the signature of overfitting.</p>")
    wf = res["walk_forward"]
    rows = []
    for f in wf["fold"]:
        if "skip" in f:
            continue
        rows.append({
            "fold": f["fold"], "train up to": f["train_to"],
            "test": f"{f['test_from']} → {f['test_to']}",
            f"IS calibrated ({wf['objective']})": f["train_calibrated"].get(wf["objective"]),
            f"OOS calibrated": f["test_calibrated"].get(wf["objective"]),
            "OOS base": f["test_base"].get(wf["objective"]),
            "OOS buy&hold": f["test_buyhold"].get(wf["objective"]),
            "beats the base": "yes" if f["beats_base"] else "no"})
    o.append(_tab(pd.DataFrame(rows), highlight=lambda r: r["beats the base"] == "no"))
    o.append("<h3>Stitched curve of the out-of-sample blocks only</h3>")
    o.append(_tab(pd.DataFrame([{"what": k, **{k2: v2 for k2, v2 in v.items()
                                               if k2 != "label"}}
                                for k, v in wf["oos_stitched"].items()])))
    o.append(f"<p>Folds in which the calibration beats the base: "
             f"<b>{wf['folds_beating_base']} of {wf['valid_folds']}</b>. "
             f"Median IS→OOS degradation: <b>{wf['median_degradation']}</b>.</p>")

    # ---------------- 3-4. overfitting ---------------- #
    o.append("<h2>3-4 · Overfitting: PBO and Deflated Sharpe</h2>")
    o.append("<p class='why'>Trying N configurations and taking the "
             "best, the winner's result is the MAXIMUM of N "
             "draws: inflated by construction. <b>PBO</b> measures the "
             "probability that the selection procedure produces a loser "
             "out of sample. The <b>Deflated Sharpe</b> discounts the "
             "winner's Sharpe for the number of trials and says what Sharpe "
             "pure chance would have produced anyway.</p>")
    for name, block in res["overfitting"].items():
        o.append(f"<h3>{_e(name)}</h3>")
        pb = block["pbo"]
        o.append(_tab(pd.DataFrame([{k: v for k, v in pb.items()
                                     if not k.startswith("_")}])))
        ds = block["dsr"]
        o.append(_tab(pd.DataFrame([ds])))

    # ---------------- 5. bootstrap ---------------- #
    o.append("<h2>5 · Block bootstrap</h2>")
    o.append("<p class='why'>History is ONE realisation. Resampling "
             "blocks of one trading month (which preserve autocorrelation and "
             "volatility clustering) yields thousands of plausible "
             "histories. The <b>extra_*</b> rows are the difference against the "
             "market: if the interval crosses zero, the claim "
             "«it beats the market» is not proven. Max drawdown depends "
             "on the order of the days, which the bootstrap reshuffles: read it "
             "as indicative.</p>")
    for name, bs in res["bootstrap"].items():
        o.append(f"<h3>{_e(name)}</h3>")
        rows = [{"metric": k, **v} for k, v in bs["intervals"].items()]
        obs = bs["observed"]
        for r in rows:
            r["observed"] = obs.get(r["metric"])
        o.append(_tab(pd.DataFrame(rows)))
        o.append(f"<p>Simulated histories in which the strategy beats the market: "
                 f"CAGR <b>{bs['prob_extra_cagr_positive_pct']}%</b> · "
                 f"Sharpe <b>{bs['prob_extra_sharpe_positive_pct']}%</b> "
                 f"<span class='crit'>(≥95% is needed to speak of a proven "
                 f"advantage)</span></p>")

    # ---------------- 6. permutation ---------------- #
    o.append("<h2>6 · Permutation test (chance, at equal presence)</h2>")
    o.append("<p class='why'>The right null hypothesis for a timing "
             "strategy is not «zero return»: it is «being in x% of the "
             "time in a rising market». The position series is rotated "
             "cyclically — same time in the market, same number of "
             "trades, same durations — and only the alignment with the "
             "returns is destroyed. The p-value is the share of random histories "
             "that do at least as well as the strategy: p=0.30 means a die "
             "with the same market presence beats the strategy 3 times out of "
             "10.</p>")
    for name, pm in res["permutation"].items():
        o.append(f"<h3>{_e(name)}</h3>")
        rows = []
        for k, v in pm["null"].items():
            rows.append({"metric": k, "strategy": pm["observed"].get(k),
                         "chance p5": v["p5"], "chance median": v["median"],
                         "chance p95": v["p95"],
                         "p-value": pm["p_value"].get(k)})
        o.append(_tab(pd.DataFrame(rows),
                      highlight=lambda r: (r["p-value"] or 1) > 0.10))

    # ---------------- 7. sub-periods ---------------- #
    o.append("<h2>7 · Stability across sub-periods</h2>")
    o.append("<p class='why'>A number over twenty years can come from a "
             "single episode. Here we look at where it comes from: year by year, "
             "how often a 3-year window beats the market, how it "
             "behaves in the crises, and how concentrated the advantage is in "
             "a few months. The jackknife removes one year at a time: if removing "
             "a year the edge disappears, the strategy is a single number, not a "
             "method.</p>")
    sp = res["subperiods"]
    o.append("<h3>Year by year</h3>")
    o.append(_tab(pd.DataFrame(sp["per_year"]),
                  highlight=lambda r: (r.get("extra_pct") or 0) < 0))
    if sp.get("rolling_3y"):
        o.append("<h3>Rolling 3-year windows</h3>")
        o.append(_tab(pd.DataFrame([sp["rolling_3y"]])))
    o.append("<h3>In the crises</h3>")
    o.append(_tab(pd.DataFrame(sp["crises"])))
    o.append("<h3>Per regime quadrant</h3>")
    o.append(_tab(pd.DataFrame(sp["per_quadrant"])))
    o.append("<h3>Concentration of the advantage</h3>")
    c = sp["concentration"]
    o.append(_tab(pd.DataFrame([{k: v for k, v in c.items()
                                 if not isinstance(v, list)}])))
    o.append(_tab(pd.DataFrame(c["best_months"])))
    o.append("<h3>Jackknife: removing one year at a time</h3>")
    o.append(_tab(pd.DataFrame(sp["annual_jackknife"])))
    if sp.get("most_decisive_year"):
        a = sp["most_decisive_year"]
        o.append(f"<p>Most decisive year: <b>{a['without_year']}</b> — "
                 f"removing it the annual advantage over the market becomes "
                 f"<b>{a['extra_cagr_pct']}%</b>.</p>")
    if res.get("rolling_start"):
        av = res["rolling_start"]
        o.append("<h3>Sensitivity to the start date</h3>")
        o.append("<p class='why'>The same strategy evaluated from all the "
                 "possible starts. If the verdict depends on when you would have "
                 "started, the historical result is chance, not "
                 "an expectation.</p>")
        o.append(_tab(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v)
                                     for k, v in av.items() if k != "rows"}])))

    # ---------------- 8. parameter sensitivity ---------------- #
    if res.get("sensitivity") is not None and len(res["sensitivity"]):
        o.append("<h2>8 · Sensitivity to the structural parameters</h2>")
        o.append("<p class='why'>Here it is not the calibrated thresholds that are "
                 "touched but the engine parameters. We look for a PLATEAU: if moving "
                 "a parameter by ±1 the result moves little, the exact value "
                 "is not critical (and a transcription error is "
                 "harmless). If it jumps, that parameter is the fragile point and must be "
                 "verified at the source.</p>")
        o.append(_tab(pd.DataFrame(res["sensitivity"]),
                      highlight=lambda r: bool(r.get("current"))))

    o.append("<footer>Generated by the robustness battery of the Tactical "
             "Trading Bot. All curves use SPX returns as a "
             "proxy for the timing of the market leg: the proxy is declared, and "
             "the «leverage» column serves to estimate the effect of the real asset. "
             "None of these measures is a forecast.</footer>")
    return "".join(o)
