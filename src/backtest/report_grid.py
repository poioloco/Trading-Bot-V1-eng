"""Grid calibration report: CSV of the outcomes and an HTML page with tables
and colour maps (one per regime, training and unseen years)."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import QUAD_LABELS, QUADS

WINDOW_NAMES = {"train": "Training", "unseen": "Unseen years",
                "full": "Whole period"}
LABELS = {"today": "Today (automatic re-entry)",
          "start": "Trend mode, starting thresholds",
          "calibrated": "Trend mode, calibrated thresholds"}
TIEBREAK_TEXT = {
    "robust": "robust (among equivalent thresholds the one furthest from the edges of its plateau wins)",
    "nearest": "nearest (among equivalent thresholds the one closest to the starting one wins)",
    "unseen": "unseen (among thresholds tied in training the unseen years decide)",
}
ACTION_TEXT = {"changed": "changed",
               "changed_among_ties": "changed among ties · identical training, deeper plateau",
               "recentred": "recentred · same results",
               "kept": "unchanged"}
THRESHOLD_NAMES = {"rsi_buy": "RSI", "vrp_buy": "VRP", "tdex_buy": "TDEX", "rsi": "RSI", "vrp": "VRP"}


def fmt_margins(d: dict) -> str:
    """'RSI 50 · VRP 0 · TDEX 40': free 0.1 steps on both sides."""
    return " · ".join(f"{THRESHOLD_NAMES[k]} {m['min']}" for k, m in d["margins"].items())


def _num(x: float, dec: int = 2) -> str:
    return f"{x:,.{dec}f}"


def write_csv(res: dict, path: Path) -> Path:
    rows = []
    for q in QUADS:
        tab = res["per_quad"][q]["table"]
        df = pd.DataFrame({k: v for k, v in tab.items() if k != "outcome"})
        df.insert(0, "regime", QUAD_LABELS[q])
        df.insert(0, "quad", q)
        rows.append(df.sort_values(f"train_{res['key']}", ascending=False))
    out = pd.concat(rows, ignore_index=True)
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(4)
    out.to_csv(path, index=False)
    return path


def verdict(res: dict) -> str:
    """One honest sentence: does the calibration hold up in the unseen years?"""
    r = res["references"]
    p, c = r["start"]["unseen"]["calmar"], r["calibrated"]["unseen"]["calmar"]
    a_p, a_c = r["start"]["train"]["calmar"], r["calibrated"]["train"]["calmar"]
    if abs(a_c - a_p) < 1e-9 and abs(c - p) < 1e-9:
        judgement = ("no result changes: the thresholds already had the best outcome "
                     "in training" + (" (at most they were recentred in their "
                                      "plateau, with identical historical numbers)"
                                      if any(a == "recentred" for a in res.get("actions", {}).values())
                                      else ""))
    elif abs(a_c - a_p) < 1e-9 and res.get("tiebreak") != "unseen":
        judgement = ("nothing changes in training; in the unseen years it changes because, among "
                     "tied outcomes, the one with the deepest plateau was taken WITHOUT looking at "
                     "those years: the difference is a chance effect, not a proven improvement")
    elif abs(c - p) < 0.02:
        judgement = ("in the UNSEEN years the calibration changes practically "
                     "nothing: the improvement in training is not confirmed")
    elif c > p:
        judgement = ("in the UNSEEN years the calibration improves there too: it is a "
                     "sign in favour (not a proof)")
    else:
        judgement = ("in the UNSEEN years the calibration does WORSE than the start: "
                     "the improvement in training is probably fitting to noise")
    extra = (" WARNING: with the «unseen» tie-break those years were used "
             "to choose, so the comparison is no longer a clean check."
             if res.get("tiebreak") == "unseen" else "")
    return (f"Training Calmar {_num(a_p)} → {_num(a_c)}; in the unseen years "
            f"{_num(p)} → {_num(c)}: {judgement}.{extra}")


def _references_table(res: dict) -> str:
    wins = ("train", "unseen", "full")
    th = "".join(f'<th colspan="5" class="{ "key" if f == "unseen" else ""}">'
                 f'{WINDOW_NAMES[f]}<br><span class="sub">{res["windows"][f][0]} → '
                 f'{res["windows"][f][1]}</span></th>' for f in wins)
    sub = "".join("<th>return/yr</th><th>max DD</th><th>Calmar</th>"
                  "<th>time in</th><th>entries</th>" for _ in wins)
    rows = []
    trend_now = res.get("mode_today") == "trend"
    labels = dict(LABELS)
    if trend_now:
        labels["start"] = "Trend mode, current thresholds in params.yaml"
    for k in (("start", "calibrated") if trend_now else ("today", "start", "calibrated")):
        m = res["references"][k]
        cells = "".join(
            f"<td>{_num(m[f]['cagr_pct'])}%</td><td>{_num(m[f]['max_dd_pct'], 1)}%</td>"
            f"<td><b>{_num(m[f]['calmar'])}</b></td><td>{_num(m[f]['time_in_market_pct'], 0)}%</td>"
            f"<td>{int(m[f]['entries'])}</td>" for f in wins)
        rows.append(f'<tr class="{k}"><th class="name">{labels[k]}</th>{cells}</tr>')
    return (f'<div class="scroll"><table class="num"><thead><tr><th></th>{th}</tr>'
            f'<tr><th></th>{sub}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def _fmt_thresholds(d: dict) -> str:
    names = {"rsi_buy": "RSI", "vrp_buy": "VRP", "tdex_buy": "TDEX", "rsi": "RSI", "vrp": "VRP"}
    return " · ".join(f"{names[k]} {_num(v, 1)}" for k, v in d.items())


def _regimes_table(res: dict) -> str:
    part = res["part"]
    rows = []
    for q in QUADS:
        d = res["per_quad"][q]
        before, after = res["start"][part][q], res["final"][part][q]
        plat = " · ".join(f"{k.split('_')[0].upper()} {_num(a, 1)}…{_num(b, 1)}"
                          for k, (a, b) in d["plateau"].items())
        v = d["neighbours"]
        action = res.get("actions", {}).get(q, "")
        notes = d.get("notes") or []
        note = (f"<br><span class='warn'>⚠ {html.escape(', '.join(notes))}</span>" if notes else "")
        rows.append(
            f"<tr><th class='name'>Q{q} {QUAD_LABELS[q]}</th><td>{_fmt_thresholds(before)}</td>"
            f"<td><b>{_fmt_thresholds(after)}</b><br><span class='sub'>{ACTION_TEXT.get(action, '')}</span></td>"
            f"<td><b>{fmt_margins(d)}</b><br><span class='sub'>together: {d['margin']}"
            f"{'' if d['margin'] >= d['max_margin'] else ' (the innermost point: ' + str(d['max_margin']) + ')'}"
            f"</span>{note}</td>"
            f"<td>{d['combinations']:,}" + f"</td><td>{d['distinct_outcomes']:,}"
            + f"</td><td>{d['same_outcome']:,}" + f"<br><span class='sub'>{plat}</span></td>"
            f"<td>{_num(v['median'])} / {_num(v['worst'])}</td>"
            f"<td>{_num(v['unseen_median'])} / {_num(v['unseen_worst'])}</td></tr>")
    return ("<div class='scroll'><table class='num'><thead><tr><th></th><th>before</th><th>after</th>"
            "<th>margin<br><span class='sub'>free 0.1 steps on both sides</span></th>"
            "<th>combinations tried</th><th>distinct outcomes</th>"
            "<th>combinations with the same outcome as the choice<br><span class='sub'>box containing them</span></th>"
            f"<th>neighbours ±0.5: training {res['key']}<br><span class='sub'>median / worst</span></th>"
            "<th>neighbours ±0.5: unseen Calmar<br><span class='sub'>median / worst</span></th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _ties_tables(res: dict) -> str:
    """For each regime: the outcomes TIED in training with the chosen one,
    with the numbers of the unseen years marked 'informational only'."""
    ch = res["key"]
    blocks = []
    for q in QUADS:
        p = res["per_quad"][q].get("ties")
        if not p:
            continue
        rows = p["rows"]
        chosen = next((r for r in rows if r["chosen"]), rows[0])
        if p["outcomes"] <= 1:
            blocks.append(f"<p><b>Q{q} {QUAD_LABELS[q]}</b> — no ties: the chosen outcome is "
                          f"the only one with this training {html.escape(ch)}.</p>")
            continue
        mv = [r["unseen_calmar"] for r in rows]
        header = (f"<b>Q{q} {QUAD_LABELS[q]}</b> — {p['outcomes']} outcomes tied in training "
                  f"({p['combinations']:,} combinations)"
                  + f" · unseen-years Calmar from {_num(min(mv))} to {_num(max(mv))}, "
                  f"choice {_num(chosen['unseen_calmar'])}"
                  + (f" · showing the {len(rows)} largest" if p["truncated"] else ""))
        trs = []
        for r in rows:
            box = " · ".join(f"{THRESHOLD_NAMES[k]} {_num(a, 1)}…{_num(b, 1)}" if a != b
                             else f"{THRESHOLD_NAMES[k]} {_num(a, 1)}"
                             for k, (a, b) in r["box"].items())
            cls = " class='calibrated'" if r["chosen"] else ""
            trs.append(
                f"<tr{cls}><td>{'◯ chosen' if r['chosen'] else ''}</td>"
                f"<td>{r['combinations']:,}" + f"</td><td style='text-align:left'>{box}</td>"
                f"<td>{r['depth']}</td><td>{_fmt_thresholds(r['inner_point'])}</td>"
                f"<td>{_num(r['train_objective'], 3)}</td>"
                f"<td class='info'>{_num(r['unseen_calmar'])}</td>"
                f"<td class='info'>{_num(r['unseen_cagr_pct'])}%</td>"
                f"<td class='info'>{_num(r['unseen_max_dd_pct'], 1)}%</td></tr>")
        table = ("<div class='scroll'><table class='num'><thead><tr><th></th><th>combinations</th>"
                 "<th>box containing them</th><th>depth<br><span class='sub'>margin of the innermost point</span></th>"
                 "<th>innermost point</th>"
                 f"<th>training {html.escape(ch)}</th>"
                 "<th class='info'>unseen-years Calmar<br><span class='sub'>informational only</span></th>"
                 "<th class='info'>unseen return/yr<br><span class='sub'>informational only</span></th>"
                 "<th class='info'>unseen max DD<br><span class='sub'>informational only</span></th>"
                 f"</tr></thead><tbody>{''.join(trs)}</tbody></table></div>")
        opened = " open" if len(rows) <= 8 else ""
        blocks.append(f"<details{opened}><summary>{header}</summary>{table}</details>")
    return "".join(blocks)


def _rounds_table(res: dict) -> str:
    rows = []
    for s in res["history"]:
        action = s.get("action", "change" if s["changed"] else "keep")
        rows.append(f"<tr><td>{s['round']}</td><td>Q{s['quad']} {QUAD_LABELS[s['quad']]}</td>"
                    f"<td>{_fmt_thresholds(s['before'])}</td><td>{_fmt_thresholds(s['after'])}</td>"
                    f"<td>{action}</td>"
                    f"<td>{_num(s['objective_before'], 3)} → {_num(s['objective_after'], 3)}</td></tr>")
    return ("<div class='scroll'><table class='num'><thead><tr><th>round</th><th>regime</th><th>before</th>"
            "<th>after</th><th></th><th>training objective</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def write_html(res: dict, path: Path, extra_title: str = "") -> Path:
    part = res["part"]
    maps = {}
    for q in QUADS:
        d = res["per_quad"][q]
        before = res["start"][part][q]
        k0, k1 = list(before)[:2]
        maps[q] = {
            "tr": np.round(d["train_map"], 3).ravel().tolist(),
            "mv": np.round(d["unseen_map"], 3).ravel().tolist(),
            "chosen": list(d["point"]),
            "start": [int(np.clip(round((before[k0] + 10) * 10), 0, 200)),
                      int(np.clip(round((before[k1] + 10) * 10), 0, 200))],
            "slice": d["slice"], "name": f"Q{q} {QUAD_LABELS[q]}",
        }
    lab_y, lab_x = ("Entry RSI", "Entry VRP") if part == "entries" else \
                   ("Confirmation RSI", "Confirmation VRP")
    ver = res["verification"]
    what = ("ENTRIES · when to re-enter after a trend-filter exit" if part == "entries"
            else "EXITS · confirmation thresholds of the trend filter")
    combo = "8,120,601 (RSI × VRP × TDEX)" if part == "entries" else "40,401 (RSI × VRP)"
    unseen_warning = ""
    if res.get("tiebreak") == "unseen":
        unseen_warning = ('<div class="card alarm"><b>⚠ Warning: «unseen» tie-break.</b> Among the thresholds '
                          'tied in training, the Calmar of the unseen years decided. Those years '
                          '<b>are no longer a clean test</b>: their numbers below are inflated by the '
                          'choice itself and the only check left is the future.</div>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grid calibration</title>
<style>
:root {{ color-scheme: light; --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --ring:rgba(11,11,11,0.10); --accent:#2a78d6; --hl:#eef4fc; }}
@media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; --bg:#0d0d0d; --surface:#1a1a19;
  --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --ring:rgba(255,255,255,0.10);
  --accent:#3987e5; --hl:#1d2a3a; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:32px 0 8px; }}
.sub {{ color:var(--ink2); font-size:12.5px; font-weight:400; }}
.card {{ background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:16px; margin:12px 0; }}
.verdict {{ border-left:4px solid var(--accent); }}
.scroll {{ overflow-x:auto; }}
table.num {{ border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:12.5px; min-width:100%; }}
table.num th, table.num td {{ padding:5px 6px; border-bottom:1px solid var(--grid); text-align:right; white-space:nowrap; }}
table.num thead th {{ color:var(--ink2); font-weight:600; text-align:center; }}
table.num th.name {{ text-align:left; font-weight:600; }}
th.key {{ background:var(--hl); }}
tr.calibrated td, tr.calibrated th {{ background:var(--hl); }}
.maps {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:16px; }}
.map h3 {{ font-size:14px; margin:0 0 2px; }}
.map canvas {{ width:100%; max-width:420px; aspect-ratio:1; display:block; cursor:crosshair;
  border-radius:4px; image-rendering:pixelated; }}
.axis {{ display:flex; justify-content:space-between; color:var(--muted); font-size:11.5px; max-width:420px; }}
.legend {{ display:flex; align-items:center; gap:8px; font-size:12.5px; color:var(--ink2); flex-wrap:wrap; }}
.legend canvas {{ width:220px; height:12px; border-radius:3px; }}
#tip {{ position:fixed; pointer-events:none; background:var(--surface); color:var(--ink);
  border:1px solid var(--ring); border-radius:8px; padding:8px 10px; font-size:12.5px;
  box-shadow:0 4px 16px rgba(0,0,0,.15); display:none; font-variant-numeric:tabular-nums; z-index:5; }}
ul {{ padding-left:20px; }} li {{ margin:4px 0; }}
.warn {{ color:#b0302e; font-size:12.5px; }}
@media (prefers-color-scheme: dark) {{ .warn {{ color:#e66767; }} }}
.alarm {{ border-left:4px solid #b0302e; }}
td.info, th.info {{ color:var(--muted); font-style:italic; }}
details {{ margin:8px 0; }} summary {{ cursor:pointer; padding:4px 0; }}
</style></head><body><main>
<h1>Grid calibration</h1>
<div class="sub">{html.escape(what)} · trend mode · costs {_num(res['cost_bps'], 0)} basis points per side ·
objective: {html.escape(res['objective'])} in training · tie-break:
{html.escape(TIEBREAK_TEXT.get(res.get('tiebreak', 'nearest'), res.get('tiebreak', '')))} {html.escape(extra_title)}</div>
{unseen_warning}
<div class="card verdict"><b>In short.</b> {html.escape(verdict(res))}</div>

<h2>Results</h2>
<div class="sub">«Unseen years» = period the training did not use: it is the real check.
{"«Current thresholds» = those already written in params.yaml (trend mode)." if res.get("mode_today") == "trend" else "«Today» = the current params.yaml configuration (after a trend-filter exit the bot re-enters automatically)."}
All rows start flat on {res['windows']['full'][0]}.</div>
<div class="card">{_references_table(res)}</div>

<h2>Thresholds per regime</h2>
<div class="sub">For each regime ALL the combinations were tried ({combo}), each from −10 to +10 in steps of 0.1,
with the other three regimes fixed; the round was repeated {res['rounds']} {'time' if res['rounds'] == 1 else 'times'}{' until no regime changed any more' if res['converged'] else ' (round limit reached)'}.
Many combinations give exactly the same result: «distinct outcomes» counts how many distinct results there are.
If the neighbours (±0.5 on every threshold) hold, the choice sits on a plateau; if the worst collapses, it is a fragile peak.<br>
«Margin» = by how many 0.1 steps each threshold can be moved (on its own, up and down) without changing
a single trade; «together» = how many steps in total can be taken moving several thresholds at once.
The edge of the grid (−10 and +10) counts as an edge. «Recentred» = same history, threshold moved to the centre of its plateau.</div>
<div class="card">{_regimes_table(res)}</div>

<h2>Ties in training</h2>
<div class="sub">DIFFERENT outcomes that in training give exactly the same {html.escape(res['key'])} as the choice.
Depth tells how wide their plateau is. The italic columns (unseen years) are <b>informational only</b>:
{'with the «unseen» tie-break they were USED to choose.' if res.get('tiebreak') == 'unseen' else 'they were not used to choose, so the unseen years remain a clean check.'}</div>
<div class="card">{_ties_tables(res)}</div>

<h2>Maps</h2>
<div class="sub">Each square is a pair of thresholds: vertical {lab_y} (+10 at the top), horizontal {lab_x}
(+10 on the right). Colour = training {html.escape(res['key'])} (left) and Calmar in the unseen years (right):
each map has its own scale, from its lowest value (light) to its highest (dark); if a map contains
both positive and negative values, red marks losses and grey zero. ◯ = chosen thresholds, ✕ = starting thresholds.
Hover to read the values.</div>
<div class="card"><div class="legend"><span>lowest</span><canvas id="leg" width="220" height="12"></canvas><span>highest</span>
<span class="sub">· each map has its own scale</span></div>
<div class="maps" id="maps"></div></div>

<h2>How the rounds went</h2>
<div class="card">{_rounds_table(res)}</div>

<h2>Automatic checks</h2>
<div class="card"><ul>
<li>The fast engine was compared with the bot's real engine on {ver['engine_cases']} configurations
(the final choice plus random combinations): identical positions day by day.</li>
<li>The count of the combinations was verified by directly simulating {ver['dedup_cases']:,} random combinations: identical numbers.</li>
<li>Returns computed on the S&amp;P 500 without leverage (the bot trades TQQQ, 3x leveraged: real drawdowns are deeper).</li>
</ul></div>
</main><div id="tip"></div>
<script>
const M = {json.dumps(maps, separators=(",", ":"))};
const LY = {json.dumps(lab_y)}, LX = {json.dumps(lab_x)}, CH = {json.dumps(res['key'])};
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
const NEU = dark ? [56,56,53] : [240,239,236];
const POS = dark ? [134,182,239] : [16,66,129];
const NEG = dark ? [230,103,103] : [176,48,46];
const SEQ0 = dark ? [16,66,129] : [205,226,251];     // light blue (dark in dark mode)
const SEQ0N = dark ? [90,30,30] : [249,215,214];
function mix(a,b,t) {{ return a.map((x,i)=>Math.round(x+(b[i]-x)*t)); }}
function colour(v, lo, hi) {{
  if (lo >= 0 || hi <= 0) {{          // a single sign: sequential scale min -> max
    const t = hi > lo ? (v - lo) / (hi - lo) : 1;
    return lo >= 0 ? mix(SEQ0, POS, t) : mix(NEG, SEQ0N, t);
  }}
  if (v >= 0) return mix(NEU, POS, hi > 0 ? Math.min(1, v/hi) : 0);
  return mix(NEU, NEG, lo < 0 ? Math.min(1, v/lo) : 0);
}}
const fmt = (v, d=2) => v.toLocaleString("en-US", {{minimumFractionDigits:d, maximumFractionDigits:d}});
const g = i => -10 + i/10;
const tip = document.getElementById("tip");
function draw(cv, vals, m) {{
  const N = 201, S = 2, ctx = cv.getContext("2d");
  cv.width = N*S; cv.height = N*S;
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  const img = ctx.createImageData(N, N);
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {{
    const v = vals[r*N + c], col = colour(v, lo, hi), y = N-1-r, p = (y*N + c)*4;
    img.data[p]=col[0]; img.data[p+1]=col[1]; img.data[p+2]=col[2]; img.data[p+3]=255;
  }}
  const off = document.createElement("canvas"); off.width = N; off.height = N;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, N*S, N*S);
  const ink = dark ? "#f0efec" : "#0b0b0b";
  const [sr, sc] = m.chosen, [pr, pc] = m.start;
  ctx.lineWidth = 2; ctx.strokeStyle = ink;
  ctx.beginPath(); ctx.arc(sc*S+1, (N-1-sr)*S+1, 9, 0, 7); ctx.stroke();
  ctx.strokeStyle = dark ? "#fcfcfb" : "#52514e"; ctx.beginPath();
  const x = pc*S+1, y = (N-1-pr)*S+1;
  ctx.moveTo(x-6,y-6); ctx.lineTo(x+6,y+6); ctx.moveTo(x+6,y-6); ctx.lineTo(x-6,y+6); ctx.stroke();
}}
const box = document.getElementById("maps");
for (const [q, m] of Object.entries(M)) {{
  for (const [key, name] of [["tr", "training · " + CH], ["mv", "unseen years · Calmar"]]) {{
    const d = document.createElement("div"); d.className = "map";
    d.innerHTML = `<h3>${{m.name}} — ${{name}}</h3><div class="sub">${{m.slice ? "slice at " + m.slice : "&nbsp;"}}</div>`;
    const cv = document.createElement("canvas"); d.appendChild(cv);
    const ax = document.createElement("div"); ax.className = "axis";
    ax.innerHTML = `<span>${{LX}} −10</span><span>0</span><span>+10</span>`; d.appendChild(ax);
    box.appendChild(d); draw(cv, m[key], m);
    cv.addEventListener("mousemove", e => {{
      const b = cv.getBoundingClientRect();
      const c = Math.min(200, Math.max(0, Math.floor((e.clientX-b.left)/b.width*201)));
      const r = 200 - Math.min(200, Math.max(0, Math.floor((e.clientY-b.top)/b.height*201)));
      const i = r*201 + c;
      tip.innerHTML = `${{LY}} <b>${{fmt(g(r),1)}}</b> · ${{LX}} <b>${{fmt(g(c),1)}}</b><br>`
        + `training ${{CH}}: <b>${{fmt(m.tr[i])}}</b><br>unseen-years Calmar: <b>${{fmt(m.mv[i])}}</b>`;
      tip.style.display = "block";
      tip.style.left = Math.min(innerWidth-260, e.clientX+14) + "px"; tip.style.top = (e.clientY+14) + "px";
    }});
    cv.addEventListener("mouseleave", () => tip.style.display = "none");
  }}
}}
const lg = document.getElementById("leg").getContext("2d");
for (let i = 0; i < 220; i++) {{ const v = i/219, c = colour(v, 0, 1);
  lg.fillStyle = `rgb(${{c}})`; lg.fillRect(i, 0, 1, 12); }}
</script></body></html>"""
    path.write_text(page, encoding="utf-8")
    return path
