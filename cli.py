"""Command-line interface of the Tactical Trading Bot V1.

Designed to be used WITHOUT programming: usually you launch it from the menu
(double click on START), but if you want:

    python cli.py update-data       # fill/update the local history
    python cli.py signals           # compute today's signal (prints it)
    python cli.py check             # "is everything ready?" — system status
    python cli.py setup             # set up the API keys (guided)

The "demo" mode works without internet or keys.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import date, datetime

import pandas as pd

from src.app import (
    all_symbols,
    build_database,
    build_longvol_calibrator,
    build_regime_detector,
    build_signal_runner,
    build_pipeline,
    build_backtester,
    build_portfolio_simulator,
    build_store,
    etf_symbols,
    load_params,
)
from src.data.calendar import current_trading_date
from src.data.importer import import_path
from src.data.ingest import incremental_update, seed_offline_demo

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Windows console (cp1252): without this, symbols like "checkmark" and "->"
# make printing fail. Force UTF-8 with a harmless fallback.
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

GREEN, RED, YEL, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def _p(msg: str = "") -> None:
    print(msg)


# ---------------------------------------------------------------------- #
def _update_history_end(params_path: str, db) -> str | None:
    """Moves `data.history_end` forward to the last REAL date in the archive
    (synthetic symbols excluded). It is the trusted boundary: a future
    re-import of your files will not touch the days already covered by the
    bot. Surgical edit (comments untouched, .bak backup). Returns the new
    date if it changed, otherwise None."""
    from src.config import ParamsEditor
    row = db.con.execute(
        "SELECT MAX(date) FROM prices_raw WHERE NOT synthetic"
    ).fetchone()[0]
    if row is None:
        return None
    new = pd.Timestamp(row).date().isoformat()
    ed = ParamsEditor(params_path)
    current = str(ed.value("data.history_end") or "")
    if current and new <= current[:10]:
        return None
    ed.set("data.history_end", new)
    return new


def cmd_update_data(p: dict, params_path: str = "params.yaml") -> int:
    mode = p["data"].get("mode", "demo")
    db = build_database(p)
    start = pd.Timestamp(p["data"]["history_start"]).date()
    end = current_trading_date()
    _p(f"{BOLD}Data update{OFF}  —  mode: {mode}  ({start} → {end})")

    if mode == "demo":
        _p(f"{YEL}DEMO mode: generating a SYNTHETIC (not real) history, so "
           f"you can try the bot without internet or API keys.{OFF}")
        info = seed_offline_demo(db, etf_symbols(p), start, end)
        _p(f"{GREEN}✓ Done.{OFF} {info['sessions']} trading sessions, "
           f"{info['etf_bars']} ETF bars + indices (SPX/VIX/TDEX).")
        for note in info["issues"][:5]:
            _p(f"  {DIM}· {note}{OFF}")
    else:
        from src.providers.smart_provider import build_smart_provider
        from src.providers.fred_tdex import FredTdexProvider
        cfg = {"data": {
            "primary_provider": p["data"]["primary_provider"],
            "fallback_provider": p["data"]["fallback_provider"],
            "indices_provider": p["data"].get("indices_provider", "fred"),
            "alpaca": {"feed": p["data"]["alpaca_feed"]},
        }}
        provider = build_smart_provider(cfg)
        syms = [s for s in all_symbols(p) if s != "TDEX"]
        _p(f"{DIM}INCREMENTAL update: downloading only the missing days after "
           f"the last date in the archive (no bulk re-download).{OFF}")
        res = incremental_update(db, provider, syms, end, fallback_start=start)
        _p(f"{GREEN}✓{OFF} +{res['written']} new bars. "
           f"Already up to date: {len(res['skipped'])} · Failed: {len(res['failed'])}")
        if res.get("pending"):
            _p(f"{YEL}… Not yet published by the source: "
               f"{', '.join(res['pending'])}. No problem: FRED publishes "
               f"a day's close on the following business day, around "
               f"09:40 New York time (15:40 CET).{OFF}")
        # TDEX from FRED (incremental)
        try:
            last = db.con.execute(
                "SELECT MAX(date) FROM prices_raw WHERE symbol='TDEX'"
            ).fetchone()[0]
            t_start = (pd.Timestamp(last) + pd.Timedelta(days=1)).date() if last else start
            if t_start > end:
                _p(f"{GREEN}✓{OFF} TDEX: already up to date.")
            else:
                # get_series returns only the new days (t_start → end):
                # the history already in the archive is not rewritten
                tdex = FredTdexProvider().get_series(t_start, end)
                if tdex.empty:
                    _p(f"{YEL}… TDEX: no new value (not yet published "
                       f"by FRED).{OFF}")
                else:
                    tdf = pd.DataFrame({"open": tdex, "high": tdex, "low": tdex,
                                        "close": tdex, "volume": 0.0})
                    db.upsert_prices(tdf, "TDEX", "fred")
                    _p(f"{GREEN}✓{OFF} TDEX: +{len(tdex)} values from FRED.")
        except Exception as exc:
            _p(f"{YEL}TDEX not downloaded: {exc}{OFF}")
        for note in res["issues"][:5]:
            _p(f"  {DIM}· {note}{OFF}")
        # trusted boundary: after a successful real update, move
        # data.history_end forward (so a future re-import of your files
        # does not touch the days already covered by the bot)
        try:
            new = _update_history_end(params_path, db)
            if new:
                _p(f"{DIM}Trusted boundary updated: history_end → "
                   f"{new} (in {params_path}, .bak backup).{OFF}")
        except Exception as exc:
            _p(f"{YEL}history_end not updated: {exc}{OFF}")
    db.close()
    return 0


def cmd_import_history(p: dict, path: str) -> int:
    if not path:
        _p(f"{RED}Specify the path of the data: "
           f"python cli.py import-history --from <folder-or-file>{OFF}")
        return 1
    db = build_database(p)
    _p(f"{BOLD}Importing the history from:{OFF} {path}")
    cutoff = p["data"].get("history_end") or None
    if cutoff:
        _p(f"{DIM}Trusted boundary: up to {cutoff} (the bot fills in the rest).{OFF}")
    _p(f"{DIM}Your data enters with source 'import' and will never be "
       f"overwritten by the incremental update.{OFF}")
    max_date = pd.Timestamp(cutoff).date() if cutoff else None
    try:
        info = import_path(db, path, only_symbols=set(all_symbols(p)),
                           max_date=max_date)
    except Exception as exc:
        _p(f"{RED}Import failed: {exc}{OFF}")
        db.close()
        return 1
    _p(f"{GREEN}✓ Imported {len(info['symbols'])} symbols, "
       f"{info['total']} bars in total.{OFF}")
    cov = db.coverage()
    if not cov.empty:
        _p(f"  Coverage: {cov['first'].min()} → {cov['last'].max()}")
    for note in info["issues"][:8]:
        _p(f"  {DIM}· {note}{OFF}")
    # warning for expected symbols that are missing
    missing = set(all_symbols(p)) - set(cov["symbol"]) if not cov.empty else set()
    if missing:
        _p(f"  {YEL}! Still without data: {', '.join(sorted(missing)[:8])}"
           f"{'…' if len(missing) > 8 else ''}{OFF}")
    db.close()
    return 0


def cmd_signals(p: dict, recompute: bool = False) -> int:
    db = build_database(p)
    if not db.symbols():
        _p(f"{YEL}The history is empty. Run 'import-history' or 'update-data' first.{OFF}")
        db.close()
        return 1
    store = build_store(db)
    pipe = build_pipeline(p, db, store)
    end = current_trading_date()
    _p(f"{BOLD}Computing the signal{OFF}")
    if recompute:
        _p(f"{DIM}Full recompute from 2005 forced…{OFF}")
    try:
        rep = pipe.run(end, force_rebuild=recompute)
    except Exception as exc:
        _p(f"{RED}Unable to compute the signal: {exc}{OFF}")
        db.close()
        return 1
    label = {"warm-up": "first full computation", "incremental":
             f"incremental update (+{rep.new_days} days)",
             "up to date": "already up to date",
             "rebuild": "full rebuild: the history changed "
                        "retroactively, state realigned"}.get(rep.mode, rep.mode)
    _p(f"{DIM}({label}){OFF}")
    if rep.warning:
        _p(f"\n{YEL}⚠ {rep.warning}{OFF}")
    _p(f"\n  Trading date  : {rep.as_of.date()}")
    _p(f"  Regime        : quad {rep.quad} ({rep.quad_name})")
    for sym, action in rep.asset_signals.items():
        col = {"BUY": GREEN, "SELL": RED, "HOLD": DIM}.get(action, "")
        _p(f"  {sym:<6} → {col}{action}{OFF}")
    m = rep.metrics
    tdex = f"{m['tdex_smooth']:.2f}" if m.get("tdex_smooth") is not None else "n/a"
    _p(f"  {DIM}SPX {m['spx']:.0f} | VIX {m['vix']:.2f} | "
       f"VRP {m['vrp_smooth']:.2f} | RSI {m['rsi_smooth']:.2f} | TDEXz {tdex}{OFF}")
    if bool(db.coverage()["synthetic"].any()):
        _p(f"\n{YEL}⚠ Synthetic DEMO data: these numbers are NOT real market "
           f"data. They only serve to run the bot.{OFF}")
    # HTML dashboard (presentation only: it does not touch the signal logic)
    try:
        from src.dashboard import build_dashboard
        # same date as the signal (if today's VIX/SPX are missing, the last
        # complete day) so that dashboard and signal match
        dash = build_dashboard(p, db, store, rep.as_of.date())
        _p(f"  {DIM}Dashboard updated: {dash} (open it with option 13){OFF}")
    except Exception as exc:
        _p(f"{YEL}Dashboard not regenerated: {exc}{OFF}")
    db.close()
    return 0


def cmd_check(p: dict) -> int:
    _p(f"{BOLD}Check: is everything ready?{OFF}\n")
    ok = True

    # 1) database and coverage
    db = build_database(p)
    cov = db.coverage()
    if cov.empty:
        _p(f"  {RED}✗{OFF} Local history EMPTY → run 'update-data'")
        ok = False
    else:
        n = len(cov)
        synth = bool(cov["synthetic"].any())
        first, last = cov["first"].min(), cov["last"].max()
        tag = f"{YEL}(synthetic DEMO){OFF}" if synth else f"{GREEN}(real){OFF}"
        _p(f"  {GREEN}✓{OFF} History: {n} symbols, {first} → {last} {tag}")
        expected = set(all_symbols(p))
        present = set(cov["symbol"])
        missing = expected - present
        if missing:
            _p(f"  {YEL}!{OFF} {len(missing)} symbols missing: "
               f"{', '.join(sorted(missing)[:6])}…")

    # 2) last run (heartbeat)
    last_run = db.last_successful_run()
    if last_run:
        _p(f"  {GREEN}✓{OFF} Last successful update: "
           f"{last_run['kind']} ({last_run['started_at']})")
    else:
        _p(f"  {DIM}·{OFF} No run recorded yet")

    # 3) keys / mode
    mode = p["data"].get("mode")
    if mode == "demo":
        _p(f"  {YEL}!{OFF} DEMO mode (synthetic). For real data: "
           f"set 'live' in params.yaml and run 'setup'")
    else:
        need = {"ALPACA_API_KEY": "Alpaca", "TELEGRAM_BOT_TOKEN": "Telegram"}
        if (p.get("notifications", {}) or {}).get("email_enabled"):
            need["EMAIL_SMTP_USER"] = "Email (SMTP)"
        for env, label in need.items():
            mark = f"{GREEN}✓{OFF}" if os.environ.get(env) else f"{YEL}!{OFF}"
            state = "present" if os.environ.get(env) else "missing (see 'setup')"
            _p(f"  {mark} {label} key: {state}")

    db.close()
    _p(f"\n{GREEN if ok else YEL}{'All set.' if ok else 'Fix the points above.'}{OFF}")
    return 0 if ok else 1


def cmd_setup(p: dict) -> int:
    _p(f"{BOLD}Guided setup of the API keys{OFF}\n")
    _p("The bot reads the keys from a `.env` file next to this program.")
    _p("DEMO mode needs NOTHING: you can already use the bot.\n")
    _p("For REAL data you need (all free):")
    _p(f"  {DIM}1.{OFF} Alpaca  → https://alpaca.markets  (API key + secret)")
    _p(f"  {DIM}2.{OFF} Telegram→ talk to @BotFather for token + chat id")
    _p(f"  {DIM}3.{OFF} FRED    → optional, the CSV endpoint needs no key")
    _p(f"  {DIM}4.{OFF} Email   → optional (redundancy): SMTP user and password.")
    _p(f"     {DIM}With Gmail you need an 'app password':")
    _p(f"     myaccount.google.com/apppasswords{OFF}\n")
    env_path = ".env"
    if os.path.exists(env_path):
        _p(f"{YEL}A .env file already exists: leaving it untouched.{OFF}")
        return 0
    template = (
        "# Tactical Bot API keys. DO NOT share this file.\n"
        "ALPACA_API_KEY=\n"
        "ALPACA_API_SECRET=\n"
        "TELEGRAM_BOT_TOKEN=\n"
        "TELEGRAM_CHAT_ID=\n"
        "FRED_API_KEY=\n"
        "# Email (optional, for the email notification channel):\n"
        "EMAIL_SMTP_USER=\n"
        "EMAIL_SMTP_PASSWORD=\n"
    )
    with open(env_path, "w", encoding="utf-8") as fh:
        fh.write(template)
    _p(f"{GREEN}✓ Created the `.env` template file.{OFF} Open it with a text "
       f"editor, paste the keys after the '=' sign and save.")
    return 0


# ---------------------------------------------------------------------- #
def cmd_backtest(p: dict, to: str = "") -> int:
    from datetime import date as _date
    db = build_database(p)
    if not db.symbols():
        _p(f"{YEL}The history is empty. Run 'import-history' or 'update-data' first.{OFF}")
        db.close()
        return 1
    store = build_store(db)
    end = _date.fromisoformat(to) if to else current_trading_date()
    _p(f"{BOLD}Backtest{OFF} (same code as live, up to {end})")
    _p(f"{DIM}Recomputing the whole history…{OFF}")
    try:
        bt = build_backtester(p, store)
        res = bt.run(end)
        # anti-lookahead check on a few scattered dates
        idx = res.timeline.index
        sample = [idx[int(len(idx) * f)].date() for f in (0.25, 0.5, 0.75, 0.97)]
        wf = bt.walk_forward_check(sample, res.timeline)
    except Exception as exc:
        _p(f"{RED}Backtest failed: {exc}{OFF}")
        db.close()
        return 1

    out = Path("output")
    out.mkdir(exist_ok=True)
    csv_path = out / "backtest_signals.csv"
    html_path = out / "backtest_report.html"
    res.timeline.to_csv(csv_path)
    from src.backtest.report import build_html_report
    build_html_report(res, str(html_path), p["vix"]["market_asset"],
                      p["vix"]["volatility_asset"])

    s = res.stats
    _p(f"\n  Period        : {s['start']} → {s['end']} ({s['days']} days)")
    ev = s["events"]
    _p(f"  Signals       : market BUY/SELL {ev['buy_spx']}/{ev['sell_spx']} | "
       f"volatility {ev['buy_vix']}/{ev['sell_vix']}")
    _p(f"  Time in market: {s['time_in_market_pct']}%  |  in volatility: {s['time_in_volatility_pct']}%")
    _p(f"  Timing (illustr.): timing {s['timing_multiple']:.2f}× vs buy&hold {s['bh_multiple']:.2f}×")
    ok = wf["max_diff_quad"] == 0 and wf["mismatched_events"] == 0
    col = GREEN if ok else RED
    _p(f"  {col}Anti-lookahead : {wf['dates_checked']} dates checked, "
       f"max quad diff {wf['max_diff_quad']}, mismatched events {wf['mismatched_events']}{OFF}")
    _p(f"\n  {DIM}Saved: {csv_path}  and  {html_path}{OFF}")
    if bool(db.coverage()["synthetic"].any()):
        _p(f"\n{YEL}⚠ Synthetic DEMO data: NOT real market numbers.{OFF}")
    db.close()
    return 0


def cmd_deposit(p: dict, date_str: str, amount: str) -> int:
    from datetime import date as _date
    pf = p.get("portfolio", {})
    path = Path(pf.get("contributions_file", "contributions.csv"))
    try:
        d = _date.fromisoformat(date_str) if date_str else current_trading_date()
        amt = float(str(amount).replace(",", "."))
    except Exception:
        _p(f"{RED}Invalid date or amount. Example: deposit --date 2024-01-15 --amount 5000{OFF}")
        return 1
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("date,amount_eur\n")
        f.write(f"{d.isoformat()},{amt:.2f}\n")
    _p(f"{GREEN}Deposit recorded:{OFF} {amt:.2f} € on {d} → {path}")
    return 0


def cmd_portfolio(p: dict, to: str = "") -> int:
    from datetime import date as _date
    db = build_database(p)
    if not db.symbols():
        _p(f"{YEL}The history is empty. Run 'import-history' or 'update-data' first.{OFF}")
        db.close()
        return 1
    store = build_store(db)
    pf = p["portfolio"]
    end = _date.fromisoformat(to) if to else current_trading_date()
    contr_path = Path(pf.get("contributions_file", "contributions.csv"))
    _p(f"{BOLD}Portfolio / trade log{OFF} (consensus of the signals, up to {end})")
    if not contr_path.exists():
        _p(f"{YEL}No deposits file ({contr_path}). Add one with "
           f"'deposit --date YYYY-MM-DD --amount 5000' (or from the menu).{OFF}")
        db.close()
        return 1
    try:
        from src.portfolio import load_contributions
        from src.portfolio.report import build_html_report
        bt = build_backtester(p, store)
        res_bt = bt.run(end)
        on, off = pf["asset_risk_on"], pf["asset_risk_off"]
        px_on = store.get_daily_bars(on, _date(2004, 1, 1), end)["close"]
        px_off = store.get_daily_bars(off, _date(2004, 1, 1), end)["close"]
        contr = load_contributions(str(contr_path))
        sim = build_portfolio_simulator(p)
        res = sim.run(res_bt.timeline, px_on, px_off, contr)
    except Exception as exc:
        _p(f"{RED}Simulation failed: {exc}{OFF}")
        db.close()
        return 1

    out = Path("output"); out.mkdir(exist_ok=True)
    csv_path, html_path = out / "portfolio_trades.csv", out / "portfolio_report.html"
    if not res.trades.empty:
        res.trades.to_csv(csv_path, index=False)
    build_html_report(res, str(html_path))
    s = res.stats
    _p(f"\n  Period        : {s['start']} → {s['end']}  (assets: {on} / {off})")
    _p(f"  Deposited     : {s['total_deposited']:.0f} €   →   Final value: {s['final_value']:.0f} €")
    seg = GREEN if s['total_pnl'] >= 0 else RED
    _p(f"  {seg}Profit/loss: {s['total_pnl']:+.0f} €  ({s['return_on_deposits_pct']:+.1f}% on deposits){OFF}")
    _p(f"  Strategy (TWR): {s['twr_multiple']}×   |   trades: {s['n_trades']}  "
       f"(winning {s['win_rate_pct']}%, profit factor {s['profit_factor']})")
    _p(f"  Time risk-on/cash/risk-off: {s['time_risk_on_pct']}/{s['time_cash_pct']}/{s['time_risk_off_pct']}%")
    _p(f"\n  {DIM}Saved: {html_path}  and  {csv_path}{OFF}")
    if bool(db.coverage()["synthetic"].any()):
        _p(f"\n{YEL}⚠ Synthetic DEMO data: NOT real market numbers.{OFF}")
    db.close()
    return 0


LEGS_INFO = [
    ("market", "Market", "equity direction (VRP+RSI)"),
    ("volatility", "Volatility", "volatility (VRP+RSI, inverse reading)"),
    ("tdex", "TDEX", "TDEX z-score (abstains before 2013)"),
]


def _set_leg(params_path: str, key: str, value: bool) -> bool:
    """Enables/disables a leg by editing ONLY its line inside the `legs:`
    block of params.yaml, keeping comments and formatting."""
    lines = Path(params_path).read_text(encoding="utf-8").splitlines()
    in_block = False
    for i, l in enumerate(lines):
        s = l.strip()
        if not l.startswith(" ") and s.startswith("legs:"):
            in_block = True
            continue
        if in_block:
            if l and not l.startswith(" "):
                break
            if s.startswith(key + ":"):
                indent = l[: len(l) - len(l.lstrip())]
                comment = ("  #" + l.split("#", 1)[1]) if "#" in l else ""
                lines[i] = f"{indent}{key}: {str(value).lower()}{comment}"
                Path(params_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
                return True
    return False


def cmd_legs(p: dict, enable: str = "", disable: str = "",
             params_path: str = "params.yaml") -> int:
    g = p.get("legs", {})
    change = None
    if enable or disable:
        key = (enable or disable).strip().lower()
        valid = {k for k, _, _ in LEGS_INFO}
        if key not in valid:
            _p(f"{RED}Unknown leg: '{key}'. Use one of: {', '.join(valid)}{OFF}")
            return 1
        val = bool(enable)
        if _set_leg(params_path, key, val):
            g[key] = val
            change = (key, val)
        else:
            _p(f"{YEL}Could not find the '{key}' line in the legs: block of {params_path}.{OFF}")
            return 1
    active = sum(1 for k, _, _ in LEGS_INFO if g.get(k, True))
    _p(f"{BOLD}Voting legs{OFF}  (consensus: majority of the votes cast)")
    for k, label, desc in LEGS_INFO:
        on = g.get(k, True)
        state = f"{GREEN}● ACTIVE {OFF}" if on else f"{DIM}○ off    {OFF}"
        _p(f"  {state} {label:<11} {DIM}— {desc}{OFF}")
    if change:
        _p(f"\n{GREEN}Updated:{OFF} {change[0]} → {'ACTIVE' if change[1] else 'off'} "
           f"(saved in {params_path}).")
    if active == 0:
        _p(f"\n{YEL}⚠ No active leg: the bot will always stay in cash.{OFF}")
    elif active == 1:
        _p(f"\n{DIM}With a single leg the consensus follows that single vote.{OFF}")
    _p(f"\n{DIM}To change: 'legs --enable tdex' or 'legs --disable volatility' "
       f"(or from the menu). Then run 'portfolio'/'backtest' again.{OFF}")
    return 0


# ---------------------------------------------------------------------- #
def cmd_calibrate_longvol(p: dict, to: str = "", leg: str = "rsi",
                          quadrants: str = "2,4", frequencies: str = "",
                          apply: str = "", params_path: str = "params.yaml") -> int:
    """Recalibrates the long-vol thresholds (rsi_long/vrp_long) for the majority.

    The midterm thresholds were optimised for the gate (AND): in 2-of-3 they
    are too permissive (bot in volatility ~44% of the time). Here a target
    vote FREQUENCY is chosen and each quadrant's threshold becomes the
    empirical quantile of that regime (same criterion as the TDEX seeds).
    """
    from datetime import date as _date
    from src.backtest.calibrate import yaml_block

    if p["vix"].get("voting_method") != "majority":
        _p(f"{YEL}calibrate-longvol only applies with voting_method: majority "
           f"(the gate replicates the Excel and is not touched).{OFF}")
        return 1

    db = build_database(p)
    if not db.symbols():
        _p(f"{YEL}The history is empty. Run 'import-history' or 'update-data' first.{OFF}")
        db.close()
        return 1
    store = build_store(db)
    end = _date.fromisoformat(to) if to else current_trading_date()

    leg = (leg or "rsi").strip().lower()
    legs = {"rsi": ["rsi"], "vrp": ["vrp"], "both": ["rsi", "vrp"]}.get(leg)
    if legs is None:
        _p(f"{RED}--leg must be rsi, vrp or both (not '{leg}').{OFF}")
        db.close()
        return 1
    if quadrants.strip().lower() in {"all", ""}:
        quads = [1, 2, 3, 4]
    else:
        quads = sorted({int(x) for x in quadrants.split(",")})
    freqs = [float(x) for x in (frequencies or "10,15,20,25,30,40").split(",")]

    _p(f"{BOLD}Long-vol leg calibration{OFF} (majority, up to {end})")
    _p(f"{DIM}Leg: {'+'.join(legs)} · quadrants: {quads} · target frequencies: "
       f"{freqs}%\nThe market leg does NOT change: the long thresholds only act on "
       f"Buy/Sell VIX.\nRecomputing the whole history (only once)…{OFF}")
    try:
        cal = build_longvol_calibrator(p, store)
        cal.prepare(end)
        cands = cal.sweep(legs, freqs, quads)
    except Exception as exc:
        _p(f"{RED}Calibration failed: {exc}{OFF}")
        db.close()
        return 1

    # ---- table --------------------------------------------------------
    hdr = (f"\n  {'candidate':<10} {'vol%':>6} {'entries':>9} {'dur.d':>7} "
           f"{'VIX×':>6} {'SPX bp/d':>9} {'crash%':>7}   thresholds")
    _p(hdr)
    _p("  " + "-" * (len(hdr) + 4))
    for c in cands:
        s = c.stats
        thr = " ".join(
            f"Q{q}[" + ",".join(f"{k.split('_')[0]}<{v:g}" for k, v in c.overrides[q].items()) + "]"
            for q in sorted(c.overrides)
        ) or "(workbook)"
        _p(f"  {c.label:<10} {s['time_in_vol_pct']:>6} {s['entries']:>9} "
           f"{s['avg_duration_days']:>7} {s['vix_captured_x']:>6} "
           f"{s['spx_in_vol_bp_day']:>9} {s['crash_coverage_pct']:>7}   {thr}")
    ref = cands[0].stats
    _p(f"\n  {DIM}Long-vol RSI vote frequencies TODAY per quadrant: "
       f"{ref['rsi_vote_freq_pct']} · VRP: {ref['vrp_vote_freq_pct']}\n"
       f"  Legend: vol% = time in volatility · VIX× = VIX change captured while "
       f"in position (proxy)\n  SPX bp/d = average SPX return on the days in position "
       f"(better if negative)\n  crash% = share of SPX's worst 5% of days spent in "
       f"position (better if high){OFF}")

    # ---- CSV ------------------------------------------------------------
    out = Path("output"); out.mkdir(exist_ok=True)
    rows = []
    for c in cands:
        r = {"candidate": c.label, **{k: v for k, v in c.stats.items()
                                      if not isinstance(v, dict)}}
        for q in (1, 2, 3, 4):
            r[f"freq_rsi_q{q}"] = c.stats["rsi_vote_freq_pct"].get(q)
            r[f"freq_vrp_q{q}"] = c.stats["vrp_vote_freq_pct"].get(q)
            r[f"vol_q{q}"] = c.stats["time_in_vol_per_quad_pct"].get(q)
            for k, v in c.overrides.get(q, {}).items():
                r[f"{k}_q{q}"] = v
        rows.append(r)
    csv_path = out / "calibration_longvol.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _p(f"\n  {DIM}Details saved in {csv_path}{OFF}")

    # ---- YAML block / applying ------------------------------------------
    chosen = None
    if apply:
        lbl = f"f={float(apply):g}%"
        chosen = next((c for c in cands if c.label == lbl), None)
        if chosen is None:
            _p(f"{RED}--apply {apply}: frequency not among those computed "
               f"({', '.join(x.label for x in cands[1:])}).{OFF}")
            db.close()
            return 1
    if chosen is not None:
        block = yaml_block(chosen.overrides, chosen.label,
                           _date.today().isoformat())
        try:
            _apply_longvol_yaml(params_path, block)
        except Exception as exc:
            _p(f"{RED}Could not write params.yaml: {exc}{OFF}")
            _p("Block to paste by hand inside the vix section:\n" + block)
            db.close()
            return 1
        _p(f"\n{GREEN}✓ Applied {chosen.label} in {params_path} "
           f"(backup in {params_path}.bak).{OFF}\n"
           f"{DIM}params_per_quad was NOT touched: it stays the midterm "
           f"workbook. Run 'backtest' again to see the effect.{OFF}")
    else:
        best = cands[1] if len(cands) > 1 else cands[0]
        _p(f"\n{DIM}To apply a frequency: "
           f"python cli.py calibrate-longvol --apply 20 (example). YAML block "
           f"of the first candidate, if you prefer to paste it by hand:{OFF}\n"
           + yaml_block(best.overrides, best.label, _date.today().isoformat()))
    db.close()
    return 0


def cmd_calibrate_market(p: dict, to: str = "", leg: str = "both",
                         quadrants: str = "all", frequencies: str = "",
                         apply: str = "", params_path: str = "params.yaml",
                         montecarlo: str = "", objective: str = "calmar",
                         entries: bool = False) -> int:
    """Recalibrates the EXIT thresholds of the market leg for the majority.

    The Excel's sell SPX uses fixed zeros (rsi<0 OR vrp<0): with the exit
    anchored to the position (needed in majority mode) it becomes
    hyperactive and the bot is in the market ~28% of the time with trades of
    a few days. Here an exit QUANTILE is chosen: you exit a regime only when
    the indicator is in the weakest g% of THAT regime.

    Works in majority AND in gate. In gate ("calibrated gate"): the entry
    stays the Excel AND, ONLY the exit is calibrated (rsi/vrp per quadrant +
    TDEX deep fear), anchored to the position. Without the market_per_quad
    block in params.yaml the replica stays exact.
    """
    from datetime import date as _date
    from src.backtest.calibrate import yaml_block_market

    if p["vix"].get("voting_method", "gate") != "majority":
        _p(f"{YEL}GATE mode — 'calibrated gate': the entry stays the "
           f"Excel replica, ONLY the market exit is calibrated (rsi/vrp per "
           f"quadrant + TDEX deep fear), anchored to the position.\n"
           f"Without the market_per_quad block in params.yaml the gate stays "
           f"the exact replica ('current' reference in the table).{OFF}")

    db = build_database(p)
    if not db.symbols():
        _p(f"{YEL}The history is empty. Run 'import-history' or 'update-data' first.{OFF}")
        db.close()
        return 1
    store = build_store(db)
    end = _date.fromisoformat(to) if to else current_trading_date()

    leg = (leg or "both").strip().lower()
    legs = {"rsi": ["rsi"], "vrp": ["vrp"], "both": ["rsi", "vrp"]}.get(leg)
    if legs is None:
        _p(f"{RED}--leg must be rsi, vrp or both (not '{leg}').{OFF}")
        db.close()
        return 1
    if quadrants.strip().lower() in {"all", ""}:
        quads = [1, 2, 3, 4]
    else:
        quads = sorted({int(x) for x in quadrants.split(",")})

    if montecarlo:
        return _calibrate_market_montecarlo(p, db, store, end, quads,
                                            int(montecarlo), objective,
                                            apply, params_path, entries)

    qtl = [float(x) for x in (frequencies or "2,5,10,15,20").split(",")]

    _p(f"{BOLD}MARKET leg calibration{OFF} (majority, up to {end})")
    _p(f"{DIM}Exit: {'+'.join(legs)} · quadrants: {quads} · exit quantiles: "
       f"{qtl}%\nThe Buy SPX entry and the volatility leg do NOT change.\n"
       f"Recomputing the whole history (only once)…{OFF}")
    try:
        cal = build_longvol_calibrator(p, store)
        cal.prepare(end)
        cands = cal.sweep_market(legs, qtl, quads)
    except Exception as exc:
        _p(f"{RED}Calibration failed: {exc}{OFF}")
        db.close()
        return 1

    # ---- table --------------------------------------------------------
    hdr = (f"\n  {'candidate':<10} {'market%':>9} {'entries':>9} {'dur.d':>7} "
           f"{'timing×':>8} {'maxDD%':>7} {'B&H×':>6}   exit thresholds")
    _p(hdr)
    _p("  " + "-" * (len(hdr) + 4))
    for c in cands:
        s = c.stats
        thr = " ".join(
            f"Q{q}[" + ",".join(f"{k.split('_')[0]}<{v:g}" for k, v in c.overrides[q].items()) + "]"
            for q in sorted(c.overrides)
        ) or "(Excel zeros)"
        _p(f"  {c.label:<10} {s['time_in_market_pct']:>9} {s['entries']:>9} "
           f"{s['avg_duration_days']:>7} {s['timing_multiple']:>8} "
           f"{s['timing_max_drawdown_pct']:>7} {s['bh_multiple']:>6}   {thr}")
    _p(f"\n  {DIM}Legend: market% = time in position · timing× = multiple "
       f"of the strategy (real P&L on SPX closes, T+1 entry)\n"
       f"  maxDD% = maximum drawdown of the strategy · B&H× = SPX buy&hold "
       f"over the same period{OFF}")

    # ---- CSV ------------------------------------------------------------
    out = Path("output"); out.mkdir(exist_ok=True)
    rows = []
    for c in cands:
        r = {"candidate": c.label, **{k: v for k, v in c.stats.items()
                                      if not isinstance(v, dict)}}
        for q in (1, 2, 3, 4):
            r[f"market_q{q}"] = c.stats["time_per_quad_pct"].get(q)
            for k, v in c.overrides.get(q, {}).items():
                r[f"{k}_q{q}"] = v
        rows.append(r)
    csv_path = out / "calibration_market.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _p(f"\n  {DIM}Details saved in {csv_path}{OFF}")

    # ---- YAML block / applying ------------------------------------------
    chosen = None
    if apply:
        lbl = f"q={float(apply):g}%"
        chosen = next((c for c in cands if c.label == lbl), None)
        if chosen is None:
            _p(f"{RED}--apply {apply}: quantile not among those computed "
               f"({', '.join(x.label for x in cands[1:])}).{OFF}")
            db.close()
            return 1
    if chosen is not None:
        block = yaml_block_market(chosen.overrides, chosen.label,
                                  _date.today().isoformat())
        try:
            _apply_market_yaml(params_path, block)
        except Exception as exc:
            _p(f"{RED}Could not write params.yaml: {exc}{OFF}")
            _p("Block to paste by hand inside the vix section:\n" + block)
            db.close()
            return 1
        _p(f"\n{GREEN}✓ Applied {chosen.label} in {params_path} "
           f"(backup in {params_path}.bak).{OFF}\n"
           f"{DIM}params_per_quad was NOT touched: it stays the midterm "
           f"workbook. Run 'signals' and 'backtest' again to see the effect.{OFF}")
    else:
        best = cands[1] if len(cands) > 1 else cands[0]
        _p(f"\n{DIM}To apply a quantile: "
           f"python cli.py calibrate-market --apply 10 (example). YAML block "
           f"of the first candidate, if you prefer to paste it by hand:{OFF}\n"
           + yaml_block_market(best.overrides, best.label,
                               _date.today().isoformat()))
    db.close()
    return 0


def _calibrate_market_montecarlo(p: dict, db, store, end, quads: list[int],
                                 n: int, objective: str, apply: str,
                                 params_path: str, entries: bool = False) -> int:
    """Monte Carlo search of the exit thresholds (walk-forward + stability)."""
    from datetime import date as _date
    from src.backtest.calibrate import yaml_block_market

    reentry = p["vix"].get("market_reentry", "edge")
    _p(f"{BOLD}MARKET leg calibration — Monte Carlo{OFF} "
       f"({n} samples, objective: {objective}, up to {end})")
    _p(f"{DIM}Exit quantiles per quadrant (rsi, vrp and TDEX sell_spx "
       f"if the leg is active) sampled at random in [1, 30]%."
       + ("\nWith --entries the entry thresholds rsi_buy/vrp_buy are ALSO "
          "sampled in [10, 70]%." if entries else "")
       + f"\nMarket re-entry: {reentry}"
       + (" (after an exit a new edge of the entry is needed; for "
          "timely re-entry set market_reentry: state in params.yaml)"
          if reentry != "state" else " (timely re-entry)")
       + "\nThe thresholds are estimated on the first ~70% of the days (training) and "
       f"checked on the last ~30% NEVER seen (test): if the winner "
       f"holds up only in training, it is overfitting and you see it right away.\n"
       f"Recomputing the whole history once, then {n} evaluations…{OFF}")
    try:
        cal = build_longvol_calibrator(p, store)
        cal.prepare(end)
        res = cal.montecarlo_market(n=n, quads=quads, objective=objective,
                                    include_entries=entries)
    except Exception as exc:
        _p(f"{RED}Monte Carlo failed: {exc}{OFF}")
        db.close()
        return 1

    key = res["objective"]
    sp = res["split"]
    _p(f"\n  {DIM}Training {sp['train_from']} → {sp['train_to']} · "
       f"test {sp['test_from']} → {sp['test_to']}{OFF}")

    def _row(name, c):
        tr, te = c["train"], c["test"]
        return (f"  {name:<9} {tr[key]:>7} {tr['multiple']:>7} {tr['max_dd_pct']:>7} "
                f"{tr['time_in_market_pct']:>8} | {te[key]:>7} {te['multiple']:>7} "
                f"{te['max_dd_pct']:>7} {te['time_in_market_pct']:>8}")

    hdr = (f"\n  {'':<9} {'──── TRAINING ────':^32} | {'──── TEST (unseen) ────':^32}\n"
           f"  {'cand.':<9} {key:>7} {'mult×':>7} {'DD%':>7} {'mkt%':>8} "
           f"| {key:>7} {'mult×':>7} {'DD%':>7} {'mkt%':>8}")
    _p(hdr)
    _p("  " + "-" * 76)
    for i, c in enumerate(res["candidates"][:10], start=1):
        _p(_row(f"#{i}", c))
    _p("  " + "-" * 76)
    _p(_row("current", res["current"]))

    st = res["stability"]
    _p(f"\n  Stability of the winner (#1): {key} {st['winner_objective']} · "
       f"neighbours ±2pp → median {st['median_neighbour']}, worst {st['worst_neighbour']}")
    if st["worst_neighbour"] is not None and st["winner_objective"]:
        hold = st["worst_neighbour"] / st["winner_objective"] if st["winner_objective"] else 0
        if hold < 0.7:
            _p(f"  {YEL}⚠ The maximum looks like a fragile peak (the worst neighbour "
               f"loses >30%): better to choose a more stable candidate.{OFF}")
        else:
            _p(f"  {GREEN}✓ Plateau: the neighbours hold ({hold:.0%} of the winner).{OFF}")

    # ---- sensitivity: which thresholds matter and in which direction --
    sens = res.get("sensitivity") or []
    if sens:
        _p(f"\n  {BOLD}HOW MUCH EACH THRESHOLD MATTERS{OFF}  {DIM}(HIGH exit "
           f"quantile = exit the market more easily, LOW = stay in longer){OFF}")
        f0 = sens[0]["bands"]

        def _blab(f):
            return f"{f['name']} {f['from']:g}–{f['to']:g}%"

        hdr2 = (f"  {'threshold':<11} {'weight':<7}"
                + "".join(f" {_blab(f):>15}" for f in f0)
                + f"   {'best':<7} recommended (typical band)")
        _p(hdr2)
        _p("  " + "-" * (len(hdr2) - 2))
        for s in sens:
            vals = "".join(
                f" {(f[key] if f[key] is not None else '—'):>15}"
                for f in s["bands"])
            c = s["recommended"]
            _p(f"  Q{s['quad']} {s['leg']:<8} {s['strength']:<7}{vals}   "
               f"{(s['best_band'] or '—'):<7} "
               f"{c['median']:g}%  ({c['p25']:g}–{c['p75']:g}%)")
        _p(f"  {DIM}How to read it: the numbers are the median {key} of the samples "
           f"with that threshold in the low/mid/high band.\n"
           f"  weight = difference between the best and worst band moving ONLY "
           f"that threshold: if it is STRONG the threshold must be chosen carefully,\n"
           f"  if it is weak moving it changes almost nothing. "
           f"recommended = where that threshold sits in the best 20% of the "
           f"samples.{OFF}")
        if any(s["leg"] == "tdex" for s in sens):
            _p(f"  {DIM}The tdex rows are the TDEX exit threshold (sell_spx "
               f"per quadrant): exit when tdex_smooth drops below the "
               f"threshold (deep fear). TDEX longvol and buy_spx are not "
               f"touched.{OFF}")
        if any(s["leg"].endswith("_buy") for s in sens):
            _p(f"  {DIM}The rsi_buy/vrp_buy rows are the ENTRY thresholds "
               f"(sampled in [10, 70]%): LOW quantile = you enter easily, "
               f"HIGH = you enter only on the regime's real strength.{OFF}")

    # ---- recommended vs current config: concrete numbers --------------
    rec = res.get("recommended")
    if rec:
        cur = res["current"]
        _p(f"\n  {BOLD}RECOMMENDED vs CURRENT CONFIG{OFF}  {DIM}(real P&L on "
           f"SPX closes, T+1 entry){OFF}")
        _p(f"  {'':<16} {'return/yr':>11} {'max DD':>8} {'calmar':>7} "
           f"{'sharpe':>7} {'time in mkt':>12}")
        for phase, title in (("train", "training"),
                             ("test", "test (unseen)")):
            _p(f"  {title}")
            for name, cnd in (("current", cur), ("recommended", rec)):
                m = cnd[phase]
                _p(f"    {name:<14} {m['cagr_pct']:>10}% {m['max_dd_pct']:>7}% "
                   f"{m['calmar']:>7} {m['sharpe']:>7} "
                   f"{m['time_in_market_pct']:>11}%")
        d_cagr = rec["test"]["cagr_pct"] - cur["test"]["cagr_pct"]
        d_dd = rec["test"]["max_dd_pct"] - cur["test"]["max_dd_pct"]
        _p(f"\n  In short, over the test period: return "
           f"{d_cagr:+.1f} points a year and maximum drawdown "
           f"{abs(d_dd):.1f} points {'shallower' if d_dd >= 0 else 'deeper'} "
           f"than the current configuration.")

    # ---- CSV ------------------------------------------------------------
    out = Path("output"); out.mkdir(exist_ok=True)
    rows = []
    for i, c in enumerate(res["candidates"], start=1):
        r = {"rank": i}
        for phase in ("train", "test"):
            for k2, v in c[phase].items():
                r[f"{phase}_{k2}"] = v
        for q, d in (c["quantiles"] or {}).items():
            for leg, g in d.items():
                r[f"q{q}_{leg}_pct"] = g
        for q, d in c["overrides"].items():
            for k2, v in d.items():
                r[f"{k2}_q{q}"] = v
        for q, v in (c.get("tdex_sell") or {}).items():
            r[f"tdex_sell_q{q}"] = v
        rows.append(r)
    csv_path = out / "calibration_market_montecarlo.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _p(f"\n  {DIM}All {len(rows)} candidates saved in {csv_path}{OFF}")
    if sens:
        srows = []
        for s in sens:
            r = {"quad": s["quad"], "leg": s["leg"], "weight": s["effect"],
                 "strength": s["strength"], "best_band": s["best_band"],
                 "recommended_median_pct": s["recommended"]["median"],
                 "recommended_p25_pct": s["recommended"]["p25"],
                 "recommended_p75_pct": s["recommended"]["p75"]}
            for f in s["bands"]:
                pre = f"band_{f['name']}"
                r[f"{pre}_from_pct"] = f["from"]
                r[f"{pre}_to_pct"] = f["to"]
                r[f"{pre}_n"] = f["n"]
                r[f"{pre}_{key}"] = f[key]
                r[f"{pre}_cagr_pct"] = f["cagr_pct"]
                r[f"{pre}_max_dd_pct"] = f["max_dd_pct"]
            srows.append(r)
        sens_path = out / "sensitivity_montecarlo.csv"
        pd.DataFrame(srows).to_csv(sens_path, index=False)
        _p(f"  {DIM}Per-threshold sensitivity saved in {sens_path}{OFF}")

    # ---- applying --------------------------------------------------------
    chosen = None
    label = ""
    if apply:
        a = apply.strip().lower()
        if a in {"recommended"}:
            chosen = res.get("recommended")
            if chosen is None:
                _p(f"{RED}No recommended configuration available.{OFF}")
                db.close()
                return 1
            label = f"montecarlo recommended, n={n}, objective {objective}"
        else:
            rank = 1 if a in {"best", "1"} else (
                int(a) if a.isdigit() else None)
            if rank is None or not (1 <= rank <= len(res["candidates"])):
                _p(f"{RED}--apply must be 'best', 'recommended' or a "
                   f"rank 1..10.{OFF}")
                db.close()
                return 1
            chosen = res["candidates"][rank - 1]
            label = f"montecarlo #{rank}, n={n}, objective {objective}"
        block = yaml_block_market(chosen["overrides"], label,
                                  _date.today().isoformat())
        try:
            _apply_market_yaml(params_path, block)
            if chosen.get("tdex_sell"):
                _apply_tdex_sell_yaml(params_path, chosen["tdex_sell"])
        except Exception as exc:
            _p(f"{RED}Could not write params.yaml: {exc}{OFF}")
            _p("Block to paste by hand inside the vix section:\n" + block)
            if chosen.get("tdex_sell"):
                _p("And update by hand the sell_spx values of tdex_per_quad: "
                   + str(chosen["tdex_sell"]))
            db.close()
            return 1
        extra = (" — the sell_spx values of tdex_per_quad were updated too"
                 if chosen.get("tdex_sell") else "")
        _p(f"\n{GREEN}✓ Applied ({label}) in {params_path} "
           f"(.bak backup){extra}.{OFF}\n{DIM}Run 'signals' (automatic "
           f"rebuild) and 'backtest' again to see the effect.{OFF}")
    else:
        _p(f"\n{DIM}To apply: menu option 14, or\n"
           f"python cli.py calibrate-market --montecarlo {n} --apply "
           f"recommended   (or 'best', or a rank 1..10){OFF}")
    db.close()
    return 0


def cmd_calibrate_grid(p: dict, to: str = "", part: str = "entries",
                       apply: str = "", params_path: str = "params.yaml",
                       start: str = "2013-07-26", train_start: str = "2018-01-01",
                       objective: str = "calmar", bps: float = 5.0,
                       open_browser: bool = True, tiebreak: str = "robust") -> int:
    """GRID calibration of the market leg in trend mode: tries EVERY value
    from -10 to +10 in steps of 0.1 for every threshold of every regime.
    part = entries (re-entry rsi/vrp/tdex, exits fixed) or exits
    (trend-filter confirmation rsi/vrp, entries fixed).
    tiebreak = robust (default: among equivalent thresholds the one furthest
    from the edges of its plateau) | nearest (previous rule) | unseen (the
    unseen years decide: declared, it consumes the clean test)."""
    import time
    import webbrowser
    from datetime import date as _date
    from src.backtest.grid import (GridCalibrator, QUAD_LABELS, QUADS,
                                   TIEBREAKS, VerificationFailed, apply_yaml)
    from src.backtest.report_grid import (fmt_margins, write_csv, write_html,
                                          verdict)
    from src.app import signal_config

    part = (part or "entries").strip().lower()
    if part not in ("entries", "exits"):
        _p(f"{RED}--part must be 'entries' or 'exits' (not '{part}').{OFF}")
        return 1
    tiebreak = (tiebreak or "robust").strip().lower()
    if tiebreak not in TIEBREAKS:
        _p(f"{RED}--tiebreak must be one of: {', '.join(TIEBREAKS)} "
           f"(not '{tiebreak}').{OFF}")
        return 1
    db = build_database(p)
    if not db.symbols():
        _p(f"{YEL}The history is empty. Run 'import-history' or 'update-data' first.{OFF}")
        db.close()
        return 1
    store = build_store(db)
    end = _date.fromisoformat(to) if to else current_trading_date()
    n_comb = "8,120,601 (RSI × VRP × TDEX)" if part == "entries" else "40,401 (RSI × VRP)"
    _p(f"{BOLD}Grid calibration — {part.upper()}{OFF} (trend mode, up to {end})")
    _p(f"{DIM}Each threshold tries every value from -10 to +10 in steps of 0.1: "
       f"{n_comb} combinations per regime, one regime at a time, rounds repeated "
       f"until no regime changes any more.\n"
       + ("The exits (trend filter) stay those of params.yaml." if part == "entries"
          else "The entries stay those of params.yaml.")
       + f"\nEvaluation from {start} · training from {train_start} · costs {bps:g} "
       f"basis points per side · objective {objective} · tie-break {tiebreak}.\n"
       f"Recomputing the history…{OFF}")
    if tiebreak == "unseen":
        _p(f"{YEL}⚠ 'unseen' tie-break: among the thresholds tied in training "
           f"the unseen years will decide. Afterwards, those years are NO longer a "
           f"clean test: the only check left is the future.{OFF}")
    t0 = time.time()
    try:
        cal = build_longvol_calibrator(p, store)
        cal.prepare(end)
        g = GridCalibrator(cal.ind, cal.params_by_quad, signal_config(p),
                           start=start, train_start=train_start,
                           cost_bps=bps, objective=objective,
                           tiebreak=tiebreak)
        res = g.calibrate(part, progress=lambda s: _p(f"  {DIM}{s}{OFF}"))
    except VerificationFailed as exc:
        _p(f"{RED}CHECK FAILED: {exc}. No result was saved or "
           f"applied.{OFF}")
        db.close()
        return 1
    except Exception as exc:
        _p(f"{RED}Calibration failed: {exc}{OFF}")
        db.close()
        return 1
    db.close()

    win = res["windows"]
    _p(f"\n  {DIM}Done in {time.time() - t0:.0f} s · {res['rounds']} rounds · checks: "
       f"real engine on {res['verification']['engine_cases']} configurations, "
       f"{res['verification']['dedup_cases']} combinations simulated directly — all identical.{OFF}")
    _p(f"\n  {'':<32}{'TRAINING':^24}{'UNSEEN YEARS':^24}{'ALL':^24}")
    _p(f"  {'':<32}" + "".join(f"{str(win[f][0]) + ' → ' + str(win[f][1])[:4]:^24}"
                               for f in ("train", "unseen", "full")))
    _p(f"  {'':<32}" + "   ret.   maxDD Calmar " * 3)
    names = {"today": "today (automatic re-entry)", "start": "trend, starting thresholds",
             "calibrated": "trend, CALIBRATED thresholds"}
    # in trend mode the current configuration IS the start: a single row
    ref_rows = (("today", "start", "calibrated") if res.get("mode_today") != "trend"
                else ("start", "calibrated"))
    if res.get("mode_today") == "trend":
        names["start"] = "trend, CURRENT thresholds"
    for k in ref_rows:
        m = res["references"][k]
        _p(f"  {names[k]:<32}" + "".join(
            f"{m[f]['cagr_pct']:>7.1f}%{m[f]['max_dd_pct']:>6.1f}%{m[f]['calmar']:>7.2f} "
            for f in ("train", "unseen", "full")))
    _p(f"\n  {BOLD}Thresholds per regime{OFF}  (before → after · tie-break {tiebreak})")
    action_lab = {"changed": f"{BOLD}CHANGED{OFF}",
                  "changed_among_ties": f"{BOLD}changed among ties{OFF} {DIM}(identical training, "
                                        f"deeper plateau; at most the unseen years change){OFF}",
                  "recentred": f"{GREEN}recentred{OFF} {DIM}(same trades, more margin){OFF}",
                  "kept": f"{DIM}unchanged{OFF}"}
    for q in QUADS:
        bf, af = res["start"][part][q], res["final"][part][q]
        d = res["per_quad"][q]
        chg = ", ".join(f"{k} {bf[k]:g}→{af[k]:g}" if bf[k] != af[k] else f"{k} {af[k]:g}"
                        for k in af)
        v = d["neighbours"]
        obj_v = res["references"]["calibrated"]["train"][res["key"]]
        fragile = obj_v > 0 and v["worst"] < 0.7 * obj_v
        _p(f"  Q{q} {QUAD_LABELS[q]:<14} {chg}  · {action_lab.get(res['actions'][q], '')}")
        notes = d.get("notes") or []
        _p(f"  {'':<18}{DIM}margin (0.1 steps): {fmt_margins(d)} · together {d['margin']}"
           f" · neighbours ±0.5: median {v['median']:.2f}, worst {v['worst']:.2f}{OFF}"
           + (f"{YEL} ⚠ fragile peak{OFF}" if fragile else "")
           + (f"{YEL} ⚠ {', '.join(notes)}{OFF}" if notes else ""))
        tg = d.get("ties") or {}
        if tg.get("outcomes", 0) > 1:
            mv = [r["unseen_calmar"] for r in tg["rows"]]
            _p(f"  {'':<18}{DIM}{tg['outcomes']} outcomes tied in training · unseen-years "
               f"Calmar from {min(mv):.2f} to {max(mv):.2f} (informational only){OFF}")
    _p(f"\n  {BOLD}In short:{OFF} {verdict(res)}")

    out = Path("output"); out.mkdir(exist_ok=True)
    csv_path = write_csv(res, out / f"grid_{part}.csv")
    html_path = write_html(res, out / f"grid_{part}.html")
    _p(f"\n  {DIM}Report with the maps: {html_path} · all outcomes: {csv_path}{OFF}")

    if apply:
        label = (f"training from {train_start}, evaluation from {start}, "
                 f"objective {objective}, costs {bps:g} bps, tie-break {tiebreak}")
        try:
            apply_yaml(params_path, part, res["final"][part], label)
        except Exception as exc:
            _p(f"{RED}Could not write params.yaml: {exc}{OFF}")
            return 1
        _p(f"\n{GREEN}✓ {part} thresholds written to {params_path} and market_mode: trend "
           f"(backup in {params_path}.bak).{OFF}\n{DIM}Run 'signals' (it rebuilds "
           f"itself) and 'backtest' again to see the effect. To go back: "
           f"market_mode: classic.{OFF}")
    else:
        if open_browser:
            try:
                webbrowser.open(Path(html_path).absolute().as_uri())
            except Exception:
                pass
        _p(f"\n{DIM}To write these thresholds to params.yaml: menu 14, or\n"
           f"python cli.py calibrate-grid --part {part} --apply yes{OFF}")
    return 0


def _apply_tdex_sell_yaml(params_path: str, tdex_sell: dict) -> None:
    """Updates only the sell_spx values inside `tdex_per_quad` (longvol and
    buy_spx stay the current ones; if the block is missing it is created
    from the global thresholds). Must be called AFTER _apply_market_yaml:
    the .bak backup has already been made."""
    import re
    import yaml as _yaml
    src = Path(params_path).read_text(encoding="utf-8")
    vix = (_yaml.safe_load(src) or {}).get("vix") or {}
    old = {int(k): dict(v) for k, v in (vix.get("tdex_per_quad") or {}).items()}
    names = {1: "Disinflation", 2: "Reflation", 3: "Stagflation", 4: "Deflation"}
    lines = ["  tdex_per_quad:"]
    for q in (1, 2, 3, 4):
        d = old.get(q, {})
        sell = float(tdex_sell.get(q, d.get("sell_spx",
                                            vix.get("tdex_vote_sell_spx", -4.0))))
        lv = float(d.get("longvol", vix.get("tdex_vote_longvol", 4.0)))
        buy = float(d.get("buy_spx", vix.get("tdex_gate_buy_spx", -0.5)))
        lines.append(f"    {q}: {{sell_spx: {sell:g}, longvol: {lv:g}, "
                     f"buy_spx: {buy:g}}}   # {names[q]}")
    block = "\n".join(lines)
    pat = re.compile(r"^[ \t]*tdex_per_quad:[ \t]*\n(?:^[ \t]+\d+:[^\n]*\n?)*",
                     re.MULTILINE)
    if pat.search(src):
        new = pat.sub(block + "\n", src, count=1)
    else:
        anchor = re.compile(r"^([ \t]*# Parameters per quadrant)", re.MULTILINE)
        m = anchor.search(src)
        if not m:
            raise RuntimeError("anchor 'params_per_quad' not found in the YAML")
        new = src[: m.start()] + block + "\n\n" + src[m.start():]
    parsed = _yaml.safe_load(new)
    if "tdex_per_quad" not in (parsed.get("vix") or {}):
        raise RuntimeError("after the edit tdex_per_quad is not inside vix:")
    Path(params_path).write_text(new, encoding="utf-8")


def _apply_market_yaml(params_path: str, block: str) -> None:
    """Writes/replaces the market_per_quad block inside the vix section,
    with a surgical text edit (the file's comments stay) and a backup."""
    import re, shutil
    src = Path(params_path).read_text(encoding="utf-8")
    shutil.copyfile(params_path, params_path + ".bak")
    pat = re.compile(
        r"(?:^[ \t]*#[^\n]*MARKET-LEG EXIT THRESHOLDS[^\n]*\n(?:^[ \t]*#[^\n]*\n)*)?"
        r"^[ \t]*market_per_quad:[ \t]*\n(?:^[ \t]+\d+:[^\n]*\n?)*",
        re.MULTILINE,
    )
    if pat.search(src):
        new = pat.sub(block + "\n", src, count=1)
    else:
        anchor = re.compile(r"^([ \t]*# Parameters per quadrant)", re.MULTILINE)
        m = anchor.search(src)
        if not m:
            raise RuntimeError("anchor 'params_per_quad' not found in the YAML")
        new = src[: m.start()] + block + "\n\n" + src[m.start():]
    import yaml as _yaml
    parsed = _yaml.safe_load(new)
    if "market_per_quad" not in (parsed.get("vix") or {}):
        raise RuntimeError("after the edit the block is not inside vix:")
    Path(params_path).write_text(new, encoding="utf-8")


def _apply_longvol_yaml(params_path: str, block: str) -> None:
    """Writes/replaces the longvol_per_quad block inside the vix section,
    with a surgical text edit (the file's comments stay) and a backup."""
    import re, shutil
    src = Path(params_path).read_text(encoding="utf-8")
    shutil.copyfile(params_path, params_path + ".bak")
    # existing block: from the marker comment (or from the longvol_per_quad:
    # line) to the last quad line '    N: {...}'
    pat = re.compile(
        r"(?:^[ \t]*#[^\n]*LONG-VOL THRESHOLDS[^\n]*\n(?:^[ \t]*#[^\n]*\n)*)?"
        r"^[ \t]*longvol_per_quad:[ \t]*\n(?:^[ \t]+\d+:[^\n]*\n?)*",
        re.MULTILINE,
    )
    if pat.search(src):
        new = pat.sub(block + "\n", src, count=1)
    else:
        # insert the block right BEFORE the params_per_quad comment
        anchor = re.compile(r"^([ \t]*# Parameters per quadrant)", re.MULTILINE)
        m = anchor.search(src)
        if not m:
            raise RuntimeError("anchor 'params_per_quad' not found in the YAML")
        new = src[: m.start()] + block + "\n\n" + src[m.start():]
    # sanity: the file must stay valid YAML and contain the block
    import yaml as _yaml
    parsed = _yaml.safe_load(new)
    if "longvol_per_quad" not in (parsed.get("vix") or {}):
        raise RuntimeError("after the edit the block is not inside vix:")
    Path(params_path).write_text(new, encoding="utf-8")


def cmd_robustness(p: dict, to: str = "", level: str = "normal",
                   compare: str = "both", params_path: str = "params.yaml",
                   bps: float = 5.0, sensitivity: bool = False) -> int:
    """Robustness battery: costs/latency/leverage, anchored walk-forward,
    PBO, Deflated Sharpe, block bootstrap, permutation test, stability
    across sub-periods, sensitivity to the parameters.

    It changes NOTHING in the configuration: it measures and writes a report.
    """
    from dataclasses import replace as _replace
    from datetime import date as _date
    import numpy as _np
    from src.app import vix_quad_params, signal_config
    from src.backtest.robustness import (
        Costs, Robustness, metrics, strategy_returns, parameter_sensitivity)
    from src.backtest.report_robustness import build_html

    PROFILES = {  # (walk-forward MC samples, PBO candidates, bootstrap, permutations)
        "fast": (60, 120, 400, 300),
        "normal": (150, 250, 1000, 600),
        "deep": (400, 600, 4000, 2000),
    }
    n_wf, n_pbo, n_boot, n_perm = PROFILES.get(level, PROFILES["normal"])

    db = build_database(p)
    if not db.symbols():
        _p(f"{YEL}The history is empty. Run 'import-history' first.{OFF}")
        db.close()
        return 1
    store = build_store(db)
    end = _date.fromisoformat(to) if to else current_trading_date()
    cfg0 = signal_config(p)
    pbq = vix_quad_params(p)
    costs = Costs(bps_side=float(bps), latency=1)

    _p(f"{BOLD}Robustness of the strategy{OFF} — {level} profile, up to {end}")
    _p(f"{DIM}Computing the indicators (40-ETF regime + VIX chains)…{OFF}")
    cal = build_longvol_calibrator(p, store)
    cal.prepare(end)
    ind = cal.ind
    rb = Robustness(ind, pbq, cfg0)

    # configurations being compared
    confs: dict[str, object] = {}
    if compare in ("gate", "both"):
        confs["gate"] = _replace(cfg0, method="gate")
    if compare in ("majority", "both"):
        confs["majority"] = _replace(cfg0, method="majority")
    if compare == "current":
        confs[f"current ({cfg0.method})"] = cfg0
    # diagnostic variants: they isolate the contribution of the trend veto
    # and the behaviour with missing TDEX (the 2005-2013 gap)
    confs["gate without trend veto"] = _replace(cfg0, method="gate", trend_enabled=False)
    confs["gate blind pre-TDEX"] = _replace(cfg0, method="gate",
                                            gate_allow_missing_tdex=False)

    res: dict = {"meta": {
        "start": str(ind.index[0].date()), "end": str(ind.index[-1].date()),
        "sessions": int(len(ind)), "costs": costs.label,
        "profile": level, "configured_method": cfg0.method,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M")}}

    # ---- 0. data coverage + windows ------------------------------------- #
    _p(f"{DIM}0/8 data coverage…{OFF}")
    res["coverage"] = rb.data_coverage()
    cut = res["coverage"].get("history_fully_covered_from")
    windows = {"whole history": (str(ind.index[0].date()), str(ind.index[-1].date()))}
    if cut:
        windows[f"complete inputs (from {cut})"] = (cut, str(ind.index[-1].date()))
        windows[f"incomplete inputs (up to {cut})"] = (
            str(ind.index[0].date()), cut)
    res["windows"] = rb.compare_windows(confs, windows, costs).to_dict("records")

    # ---- 1. costs, latency, leverage ------------------------------------ #
    _p(f"{DIM}1/8 costs, latency, leverage…{OFF}")
    res["costs_latency"] = rb.costs_latency(
        confs, bps=(0.0, float(bps), 20.0), latencies=(1, 2, 3),
        leverages=(1.0, 3.0)).to_dict("records")

    # ---- 2. walk-forward ------------------------------------------------ #
    _p(f"{DIM}2/8 anchored walk-forward ({n_wf} samples × 6 folds)…{OFF}")
    res["walk_forward"] = rb.walk_forward(n_fold=6, n_samples=n_wf, costs=costs)

    # ---- 3-4. PBO + Deflated Sharpe ------------------------------------- #
    _p(f"{DIM}3-4/8 PBO and Deflated Sharpe ({n_pbo} candidates)…{OFF}")
    res["overfitting"] = {}
    for label, include in (("selection between base and calibrations", True),
                           ("calibration procedure alone", False)):
        r = rb.pbo(n_samples=n_pbo, s_blocks=10, objective="sharpe",
                   costs=costs, include_base=include)
        M = r.pop("_matrix"); r.pop("_labels", None)
        sd = M.std(axis=0)
        sh = M.mean(axis=0) / _np.where(sd > 0, sd, 1.0)
        best = int(_np.argmax(sh))
        res["overfitting"][label] = {
            "pbo": {k: v for k, v in r.items() if not k.startswith("_")},
            "dsr": rb.deflated_sharpe(pd.Series(M[:, best]), sh)}

    # ---- 5-6-7. for each main configuration ----------------------------- #
    main_confs = {k: v for k, v in confs.items()
                  if k in ("gate", "majority") or k.startswith("current")}
    res["bootstrap"], res["permutation"], res["subperiods_per_config"] = {}, {}, {}
    for name, c in main_confs.items():
        _p(f"{DIM}5-7/8 bootstrap · permutation · sub-periods [{name}]…{OFF}")
        pos = rb.position(c)
        res["bootstrap"][name] = rb.bootstrap(pos, n=n_boot, costs=costs)
        res["permutation"][name] = rb.permutation(pos, n=n_perm, costs=costs)
        res["subperiods_per_config"][name] = rb.subperiods(pos, costs=costs)
        # the same battery on the window with all inputs present
        if cut:
            rb2 = Robustness(ind.loc[cut:], pbq, cfg0)
            pos2 = rb2.position(c)
            res["bootstrap"][f"{name} · from {cut}"] = rb2.bootstrap(
                pos2, n=n_boot, costs=costs)
            res["permutation"][f"{name} · from {cut}"] = rb2.permutation(
                pos2, n=n_perm, costs=costs)
    # the report's "subperiods" block is the one of the configured config
    main = (f"current ({cfg0.method})" if f"current ({cfg0.method})" in main_confs
            else (cfg0.method if cfg0.method in main_confs
                  else list(main_confs)[0]))
    res["subperiods"] = res["subperiods_per_config"][main]
    res["rolling_start"] = rb.rolling_start(
        rb.position(main_confs[main]), costs=costs)

    # ---- 8. sensitivity to the structural parameters -------------------- #
    if sensitivity:
        _p(f"{DIM}8/8 sensitivity to the structural parameters…{OFF}")

        def _build(p_mod):
            cal2 = build_longvol_calibrator(p_mod, store)
            cal2.prepare(end)
            return cal2.ind, vix_quad_params(p_mod), signal_config(p_mod)

        perturb = {
            "regime.rsi_period": [17, 19, 21, 23, 25],
            "regime.ema_period": [12, 14, 16, 18, 20],
            "regime.vote_threshold": [3, 4, 5, 6, 7],
            "regime.bifurcation_threshold": [0.195, 0.215, 0.235],
            "vix.tdex_gate_buy_spx": [-1.5, -1.0, -0.5, 0.0, 0.5],
            "vix.trend_filter.window": [150, 175, 200, 225, 250],
        }
        res["sensitivity"] = parameter_sensitivity(
            _build, p, perturb, costs=costs).to_dict("records")
    else:
        res["sensitivity"] = []

    # ---- verdicts: the traffic lights, with a declared criterion ------- #
    res["verdicts"] = _robustness_verdicts(res, main)

    out = Path("output"); out.mkdir(exist_ok=True)
    html_path = out / "robustness_report.html"
    json_path = out / "robustness_data.json"
    html_path.write_text(build_html(res), encoding="utf-8")

    def _clean(x):
        if isinstance(x, dict):
            return {str(k): _clean(v) for k, v in x.items()
                    if not str(k).startswith("_")}
        if isinstance(x, (list, tuple)):
            return [_clean(v) for v in x]
        if isinstance(x, (_np.integer,)):
            return int(x)
        if isinstance(x, (_np.floating,)):
            return float(x)
        if isinstance(x, (_np.bool_,)):
            return bool(x)
        return x

    import json as _json
    json_path.write_text(_json.dumps(_clean(res), indent=1, ensure_ascii=False,
                                     default=str), encoding="utf-8")

    # ---- on-screen summary --------------------------------------------- #
    _p("")
    for v in res["verdicts"]:
        col = {"green": GREEN, "yellow": YEL, "red": RED}[v["level"]]
        _p(f"  {col}● {v['title']}{OFF}")
        _p(f"    {v['plain_text']}")
    _p(f"\n  {DIM}Report: {html_path}   ·   raw data: {json_path}{OFF}")
    db.close()
    return 0


def _robustness_verdicts(res: dict, main: str) -> list[dict]:
    """Turns the numbers into traffic lights, always with the threshold next
    to them.

    No threshold is a "market standard": they are declared conventions, so
    that one can disagree about the criterion instead of arguing about the
    number."""
    v: list[dict] = []

    # --- data coverage
    cut = res["coverage"].get("history_fully_covered_from")
    win = {(r["config"], r["window"]): r for r in res.get("windows", [])}
    uncovered = [k for k in win if "incomplete" in k[1]]
    if cut and uncovered:
        worse = [win[k] for k in uncovered if win[k]["config"] == main]
        cagr_unc = worse[0]["cagr_pct"] if worse else None
        lvl = "red" if (cagr_unc is not None and cagr_unc < 0) else "yellow"
        v.append({
            "level": lvl, "title": "Data coverage",
            "text": (f"All inputs only exist from <b>{cut}</b>. "
                     f"In the earlier period the «{main}» configuration "
                     f"returns <b>{cagr_unc}%</b> a year: the uncovered part "
                     f"of the history does not confirm the strategy, it dilutes it. "
                     f"Every conclusion must be read on the covered window."),
            "plain_text": (f"Complete inputs only from {cut}; before then "
                           f"the strategy returns {cagr_unc}% a year."),
            "criterion": "red if the return in the uncovered period is negative"})

    # --- latency
    cl = pd.DataFrame(res["costs_latency"])
    if len(cl):
        bps_r = float(res["meta"]["costs"].split("bps")[0])
        m = cl[(cl.config == main) & (cl.leverage == 1.0) & (cl.bps == bps_r)]
        t1 = m[m.latency == 1]["cagr_pct"]
        t2 = m[m.latency == 2]["cagr_pct"]
        if len(t1) and len(t2):
            d = float(t2.iloc[0] - t1.iloc[0])
            dd1 = float(m[m.latency == 1]["max_dd_pct"].iloc[0])
            dd2 = float(m[m.latency == 2]["max_dd_pct"].iloc[0])
            ddd = dd2 - dd1
            severity = max(abs(d) / 1.0, abs(ddd) / 5.0)   # 1 CAGR point ≈ 5 of DD
            lvl = "green" if severity < 1 else ("yellow" if severity < 2.5 else "red")
            v.append({
                "level": lvl, "title": "Execution latency",
                "text": (f"Going from T+1 to T+2 the CAGR changes by "
                         f"<b>{d:+.2f} points</b> and the maximum drawdown by "
                         f"<b>{ddd:+.1f} points</b> ({dd1:.1f}% → {dd2:.1f}%). "
                         f"One day of delay should not move the "
                         f"result: if it does — and all the more if it "
                         f"IMPROVES it — the daily timing contains no "
                         f"exploitable information, and the difference is "
                         f"noise disguised as signal."),
                "plain_text": (f"T+1 → T+2: CAGR {d:+.2f} points, "
                               f"drawdown {ddd:+.1f} points."),
                "criterion": "green if the shift stays below 1 CAGR point "
                             "and 5 of drawdown, red beyond two and a half times that"})

    # --- real leverage
    if len(cl):
        bps_ref = float(res["meta"]["costs"].split("bps")[0])
        l3 = cl[(cl.config == main) & (cl.leverage == 3.0)
                & (cl.latency == 1) & (cl.bps == bps_ref)]
        if len(l3):
            dd = float(l3["max_dd_pct"].iloc[0]); mu = float(l3["multiple"].iloc[0])
            lvl = "red" if dd < -70 else ("yellow" if dd < -50 else "green")
            v.append({
                "level": lvl, "title": "Leverage of the real asset",
                "text": (f"With 3x leverage (the profile of the asset the bot "
                         f"really trades) the maximum drawdown becomes "
                         f"<b>{dd:.1f}%</b> and the multiple <b>{mu:.2f}×</b>. "
                         f"The unleveraged curve does not describe the risk "
                         f"being taken."),
                "plain_text": f"At 3x leverage: DD {dd:.1f}%, multiple {mu:.2f}×.",
                "criterion": "red if the DD at 3x leverage exceeds 70%"})

    # --- walk-forward
    wf = res["walk_forward"]
    if wf.get("valid_folds"):
        won, tot = wf["folds_beating_base"], wf["valid_folds"]
        lvl = "green" if won > tot * 0.6 else ("yellow" if won >= tot * 0.4 else "red")
        st = wf.get("oos_stitched", {})
        c_cal = (st.get("calibrated") or {}).get(wf["objective"])
        c_bas = (st.get("base") or {}).get(wf["objective"])
        v.append({
            "level": lvl, "title": "Walk-forward: does the calibration add value?",
            "text": (f"The calibration beats the base configuration in "
                     f"<b>{won} folds out of {tot}</b>. On the stitched curve of the "
                     f"out-of-sample blocks only: calibrated "
                     f"<b>{c_cal}</b> against base <b>{c_bas}</b> "
                     f"({wf['objective']}). Median in→out of sample "
                     f"degradation: <b>{wf['median_degradation']}</b>."),
            "plain_text": (f"Calibration better than the base in {won}/{tot} "
                           f"folds (stitched {wf['objective']} {c_cal} vs {c_bas})."),
            "criterion": "green if it wins in more than 60% of the folds, red below 40%"})

    # --- PBO
    for label, block in res["overfitting"].items():
        pb, ds = block["pbo"], block["dsr"]
        lvl = "green" if pb["pbo_pct"] < 20 else ("yellow" if pb["pbo_pct"] < 50 else "red")
        v.append({
            "level": lvl, "title": f"PBO — {label}",
            "text": (f"Probability that the selection produces a loser "
                     f"out of sample: <b>{pb['pbo_pct']}%</b> "
                     f"({pb['n_candidates']} candidates, "
                     f"{pb['n_combinations']} splits). Deflated Sharpe: the "
                     f"winner's Sharpe is <b>{ds.get('annual_sharpe')}</b> "
                     f"against <b>{ds.get('expected_chance_sharpe_annual')}</b> "
                     f"that pure chance would have produced with the same "
                     f"number of trials (probability of a true Sharpe > 0: "
                     f"<b>{ds.get('dsr_pct')}%</b>)."),
            "plain_text": (f"PBO {pb['pbo_pct']}% · Sharpe {ds.get('annual_sharpe')} "
                           f"vs {ds.get('expected_chance_sharpe_annual')} by chance "
                           f"· DSR {ds.get('dsr_pct')}%"),
            "criterion": "green <20%, yellow <50%, red beyond; useful DSR ≥95%"})

    # --- permutation
    for name, pm in res["permutation"].items():
        pv = pm.get("p_value", {})
        p_sh, p_ca = pv.get("sharpe"), pv.get("calmar")
        worst_p = max([x for x in (p_sh, p_ca) if x is not None], default=1.0)
        lvl = "green" if worst_p <= 0.05 else ("yellow" if worst_p <= 0.20 else "red")
        v.append({
            "level": lvl, "title": f"Does the timing beat chance? — {name}",
            "text": (f"Against random histories with the SAME time in the market and "
                     f"the same durations: Sharpe p-value <b>{p_sh}</b>, "
                     f"Calmar <b>{p_ca}</b>, CAGR "
                     f"<b>{pv.get('cagr_pct')}</b>. The p-value is the share of "
                     f"random histories that do at least as well as the "
                     f"strategy."),
            "plain_text": (f"p-value: Sharpe {p_sh} · Calmar {p_ca} · "
                           f"CAGR {pv.get('cagr_pct')}"),
            "criterion": "green p≤0.05, yellow p≤0.20, red beyond"})

    # --- bootstrap
    for name, bs in res["bootstrap"].items():
        pc = bs["prob_extra_cagr_positive_pct"]
        ps = bs["prob_extra_sharpe_positive_pct"]
        lvl = "green" if max(pc, ps) >= 95 else ("yellow" if max(pc, ps) >= 70 else "red")
        ic = bs["intervals"].get("extra_cagr_pct", {})
        v.append({
            "level": lvl, "title": f"Does it beat the market? — {name}",
            "text": (f"On block-resampled histories, the strategy beats "
                     f"the market in <b>{pc}%</b> of cases on CAGR and in "
                     f"<b>{ps}%</b> on Sharpe. Interval of the annual "
                     f"advantage: from <b>{ic.get('p5')}%</b> to "
                     f"<b>{ic.get('p95')}%</b> (median "
                     f"{ic.get('p50')}%): if it crosses zero, the "
                     f"advantage is not proven."),
            "plain_text": (f"Beats the market in {pc}% of the histories (CAGR), "
                           f"{ps}% (Sharpe)."),
            "criterion": "green ≥95%, yellow ≥70%, red below"})

    # --- concentration
    sp = res["subperiods"]
    roll = sp.get("rolling_3y") or {}
    if roll:
        q = roll.get("share_windows_above_market_pct", 0)
        lvl = "green" if q >= 50 else ("yellow" if q >= 30 else "red")
        v.append({
            "level": lvl, "title": "Consistency over 3 years",
            "text": (f"Only <b>{q}%</b> of the rolling 3-year windows "
                     f"beat the market; the median 3-year advantage is "
                     f"<b>{roll.get('median_extra_3y_pct')}%</b> and the "
                     f"worst <b>{roll.get('worst_extra_3y_pct')}%</b>. "
                     f"It is the number that says how plausible it is to stay "
                     f"faithful to the strategy."),
            "plain_text": (f"{q}% of the 3-year windows above the market "
                           f"(median {roll.get('median_extra_3y_pct')}%)."),
            "criterion": "green ≥50%, yellow ≥30%, red below"})
    a = sp.get("most_decisive_year")
    if a:
        v.append({
            "level": "yellow", "title": "Concentration in a few episodes",
            "text": (f"Removing <b>{a['without_year']}</b> the annual "
                     f"advantage over the market becomes "
                     f"<b>{a['extra_cagr_pct']}%</b>. The 5 best months "
                     f"are worth <b>{sp['concentration'].get('extra_top5_months_log')}</b> "
                     f"of logarithmic extra return out of a total of "
                     f"<b>{sp['concentration'].get('extra_total_log')}</b>."),
            "plain_text": (f"Without {a['without_year']} the annual advantage is "
                           f"{a['extra_cagr_pct']}%."),
            "criterion": "informational: measures how much the result depends on one episode"})

    # --- rolling start
    av = res.get("rolling_start") or {}
    if av.get("n_starts"):
        q = av["share_starts_above_market_pct"]
        lvl = "green" if q >= 60 else ("yellow" if q >= 30 else "red")
        v.append({
            "level": lvl, "title": "Dependence on the start date",
            "text": (f"Over <b>{av['n_starts']}</b> different start dates, the "
                     f"strategy beats the market in <b>{q}%</b> of cases. "
                     f"CAGR from <b>{av['cagr']['min']}%</b> to "
                     f"<b>{av['cagr']['max']}%</b> (median "
                     f"{av['cagr']['median']}%)."),
            "plain_text": (f"Beats the market in {q}% of the start dates "
                           f"({av['n_starts']} tried)."),
            "criterion": "green ≥60%, yellow ≥30%, red below"})

    # --- parameter sensitivity
    sens = pd.DataFrame(res.get("sensitivity") or [])
    if len(sens) and "cagr_range_pct" in sens.columns:
        worst = sens.sort_values("cagr_range_pct", ascending=False).iloc[0]
        amp = float(worst["cagr_range_pct"])
        lvl = "green" if amp < 2 else ("yellow" if amp < 5 else "red")
        v.append({
            "level": lvl, "title": "Sensitivity to the structural parameters",
            "text": (f"The most critical parameter is "
                     f"<b>{worst['parameter']}</b>: over the values tried the "
                     f"CAGR varies by <b>{amp} points</b>. Large ranges "
                     f"mean that the result depends on the exact value "
                     f"— and those values must be verified at the source "
                     f"(the .xlsb), not inherited from a screenshot."),
            "plain_text": (f"Most critical parameter: {worst['parameter']} "
                           f"(CAGR range {amp} points)."),
            "criterion": "green <2 points, yellow <5, red beyond"})
    return v


def cmd_test_notify(p: dict) -> int:
    """Sends a test message on ALL the enabled notification channels."""
    from src.notify import build_notifier
    n = p.get("notifications", {}) or {}
    channels = [name for name, on in
                [("Telegram", n.get("telegram_enabled", True)),
                 ("Email", n.get("email_enabled", False))] if on]
    if not channels:
        _p(f"{YEL}No channel enabled in params.yaml (notifications block).{OFF}")
        return 1
    _p(f"{BOLD}Notification test{OFF} — enabled channels: {', '.join(channels)}")
    notifier = build_notifier(p)
    outcomes: list[str] = []
    for ch in notifier.channels:
        name = type(ch).__name__.replace("Notifier", "")
        try:
            ch.send_text("Tactical Bot: test message. "
                         "If you can read it, this channel works.")
            outcomes.append(f"  {GREEN}✓{OFF} {name}: sent")
        except Exception as exc:
            outcomes.append(f"  {RED}✗{OFF} {name}: {exc}")
    for r in outcomes:
        _p(r)
    ko = sum("✗" in r for r in outcomes)
    if ko:
        _p(f"\n{YEL}Check the keys in the .env file ('setup' command) and the "
           f"notifications block of params.yaml.{OFF}")
    return 1 if ko else 0


def cmd_dashboard(p: dict) -> int:
    import webbrowser
    db = build_database(p)
    if not db.symbols():
        _p(f"{YEL}The history is empty. Run 'import-history' or 'update-data' first.{OFF}")
        db.close()
        return 1
    store = build_store(db)
    _p(f"{BOLD}Dashboard{OFF} — generating it with the latest data…")
    try:
        from src.dashboard import build_dashboard
        from src.engine.pipeline import date_with_indices
        # same day as the signal: if today's VIX/SPX are not published yet,
        # the last complete day
        up_to, warning = date_with_indices(db, current_trading_date(),
                                           p["vix"]["spx_symbol"],
                                           p["vix"]["vix_symbol"])
        if warning:
            _p(f"{YEL}⚠ {warning}{OFF}")
        path = build_dashboard(p, db, store, up_to)
    except Exception as exc:
        _p(f"{RED}Dashboard failed: {exc}{OFF}")
        db.close()
        return 1
    db.close()
    _p(f"{GREEN}✓{OFF} {path}")
    try:
        webbrowser.open(Path(path).absolute().as_uri())
        _p(f"{DIM}Opened in the browser.{OFF}")
    except Exception:
        _p(f"{DIM}Open it with a double click: {Path(path).absolute()}{OFF}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from src.utils.envfile import load_env
    load_env()  # makes the keys of the .env file visible (Alpaca, Telegram…)
    parser = argparse.ArgumentParser(description="Tactical Trading Bot V1")
    parser.add_argument(
        "command",
        choices=["update-data", "import-history", "signals", "backtest",
                 "robustness", "calibrate-longvol", "calibrate-market",
                 "calibrate-grid",
                 "portfolio", "deposit", "legs", "check", "setup",
                 "test-notify", "config-ui", "dashboard"],
    )
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--from", dest="from_", default="",
                        help="data path for import-history")
    parser.add_argument("--to", default="", help="end date (YYYY-MM-DD)")
    parser.add_argument("--date", default="", help="deposit date (YYYY-MM-DD)")
    parser.add_argument("--amount", default="", help="deposit amount in euro")
    parser.add_argument("--enable", default="", help="leg to enable (market/volatility/tdex)")
    parser.add_argument("--disable", default="", help="leg to disable")
    parser.add_argument("--recompute", action="store_true", help="full recompute")
    parser.add_argument("--leg", default="",
                        help="calibrate-*: rsi | vrp | both "
                             "(default: rsi for longvol, both for market)")
    parser.add_argument("--quadrants", default="",
                        help="calibrate-*: e.g. '2,4' or 'all' "
                             "(default: 2,4 for longvol, all for market)")
    parser.add_argument("--montecarlo", default="",
                        help="calibrate-market: number of Monte Carlo samples (e.g. 500)")
    parser.add_argument("--entries", action="store_true",
                        help="calibrate-market --montecarlo: also sample the "
                             "ENTRY thresholds (rsi_buy/vrp_buy per quadrant)")
    parser.add_argument("--objective", default="calmar",
                        help="calibrate-market --montecarlo: calmar (default) | "
                             "return | sharpe")
    parser.add_argument("--frequencies", default="",
                        help="calibrate-longvol: target frequencies %% e.g. '10,15,20,25,30,40'")
    parser.add_argument("--level", default="normal",
                        help="robustness: fast | normal | deep "
                             "(how many simulations)")
    parser.add_argument("--compare", default="both",
                        help="robustness: gate | majority | both | current")
    parser.add_argument("--bps", default="5",
                        help="robustness: cost per side in basis points (default 5)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="robustness: adds the perturbation of the "
                             "structural parameters (slower)")
    parser.add_argument("--apply", default="",
                        help="calibrate-longvol: writes the chosen frequency to "
                             "params.yaml (e.g. 20) · calibrate-market --montecarlo: "
                             "'recommended', 'best' or a rank 1..10")
    parser.add_argument("--part", default="entries",
                        help="calibrate-grid: entries | exits")
    parser.add_argument("--start", default="2013-07-26",
                        help="calibrate-grid: first evaluated day (default "
                             "2013-07-26, first day with TDEX)")
    parser.add_argument("--train-start", default="2018-01-01",
                        help="calibrate-grid: start of the training "
                             "(default 2018-01-01: regime almost complete, "
                             "37/40 ETFs)")
    parser.add_argument("--no-browser", action="store_true",
                        help="calibrate-grid: do not open the report in the browser")
    parser.add_argument("--tiebreak", default="robust",
                        help="calibrate-grid: how to choose between equivalent "
                             "thresholds · robust (default: the furthest "
                             "from the edges) | nearest (the closest to the "
                             "starting one) | unseen (the unseen years "
                             "decide: it consumes the clean test)")
    args = parser.parse_args(argv)
    p = load_params(args.params)

    if args.command == "import-history":
        return cmd_import_history(p, args.from_)
    if args.command == "signals":
        return cmd_signals(p, recompute=args.recompute)
    if args.command == "backtest":
        return cmd_backtest(p, args.to)
    if args.command == "robustness":
        return cmd_robustness(p, args.to, args.level, args.compare,
                              args.params, float(args.bps), args.sensitivity)
    if args.command == "calibrate-longvol":
        return cmd_calibrate_longvol(p, args.to, args.leg or "rsi",
                                     args.quadrants or "2,4",
                                     args.frequencies, args.apply, args.params)
    if args.command == "calibrate-market":
        return cmd_calibrate_market(p, args.to, args.leg or "both",
                                    args.quadrants or "all", args.frequencies,
                                    args.apply, args.params,
                                    args.montecarlo, args.objective,
                                    args.entries)
    if args.command == "calibrate-grid":
        return cmd_calibrate_grid(p, args.to, args.part, args.apply,
                                  args.params, args.start, args.train_start,
                                  args.objective, float(args.bps),
                                  open_browser=not args.no_browser,
                                  tiebreak=args.tiebreak)
    if args.command == "portfolio":
        return cmd_portfolio(p, args.to)
    if args.command == "deposit":
        return cmd_deposit(p, args.date, args.amount)
    if args.command == "legs":
        return cmd_legs(p, args.enable, args.disable, args.params)
    if args.command == "update-data":
        return cmd_update_data(p, args.params)
    if args.command == "dashboard":
        return cmd_dashboard(p)
    if args.command == "config-ui":
        from src.config.webui import run_config_ui
        return run_config_ui(args.params)
    return {
        "check": cmd_check,
        "setup": cmd_setup,
        "test-notify": cmd_test_notify,
    }[args.command](p)


if __name__ == "__main__":
    sys.exit(main())
