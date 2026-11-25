"""Text menu of the Tactical Bot — to use it WITHOUT typing commands.

It normally starts by itself with a double click on START.bat (Windows),
start.command (Mac) or start.sh (Linux). It shows numbered actions: just
type the number.
"""
from __future__ import annotations

import sys

import cli

BOLD, DIM, OFF, CYAN = "\033[1m", "\033[2m", "\033[0m", "\033[36m"
GREEN, RED = "\033[32m", "\033[31m"

MENU = f"""
{BOLD}╔══════════════════════════════════════════════╗
║         TACTICAL TRADING BOT  ·  V1           ║
╚══════════════════════════════════════════════╝{OFF}

  {CYAN}1{OFF}  Import the history (the data you already have)
  {CYAN}2{OFF}  Update the data   (fill the missing days / demo)
  {CYAN}3{OFF}  Compute today's signal
  {CYAN}4{OFF}  Backtest          (whole history + HTML report)
  {CYAN}5{OFF}  Portfolio / trade log  (P&L on your deposits)
  {CYAN}6{OFF}  Add a deposit
  {CYAN}7{OFF}  Voting legs       (enable/disable market · volatility · TDEX)
  {CYAN}8{OFF}  Check: is everything ready?
  {CYAN}9{OFF}  API key setup (guided)
  {CYAN}10{OFF} Calibrate the volatility leg  (long-vol thresholds for the majority)
  {CYAN}11{OFF} Control panel          (edit the parameters from the browser)
  {CYAN}12{OFF} Test the notifications (Telegram / email)
  {CYAN}13{OFF} Dashboard              (signal, votes and performance in the browser)
  {CYAN}14{OFF} Calibrate the market leg  (grid: entries / exits)
  {CYAN}15{OFF} Robustness        (does it hold? walk-forward, overfitting, chance)
  {CYAN}16{OFF} Full run          (in a row: 2 → 3 → 4 → 13, up to the dashboard)
  {CYAN}0{OFF}  Exit
"""


def _run(args: list[str]) -> int:
    """Runs a bot command. Returns 0 if it went well, otherwise a number
    other than 0 (the full run needs it to know whether to stop)."""
    try:
        return cli.main(args) or 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except Exception as exc:  # pragma: no cover
        print(f"\n\033[31mError: {exc}\033[0m")
        return 1


# Item 16: items 2 → 3 → 4 → 13 one after the other, without pressing Enter
# between steps. (menu item, name, command)
FULL_RUN = [
    ("2", "Update the data", ["update-data"]),
    ("3", "Compute today's signal", ["signals"]),
    ("4", "Backtest", ["backtest"]),
    ("13", "Dashboard", ["dashboard"]),
]


def _full_run() -> None:
    """Runs items 2, 3, 4 and 13 in a row. If a step fails it stops there
    and says which one, so signal and dashboard are not built on top of a
    skipped step."""
    n = len(FULL_RUN)
    print(f"\n{BOLD}Full run{OFF}  {DIM}(2 → 3 → 4 → 13 — the backtest may "
          f"take a few minutes){OFF}")
    for i, (item, name, command) in enumerate(FULL_RUN, 1):
        print(f"\n{CYAN}{BOLD}── Step {i}/{n} · item {item}: {name} ──{OFF}\n")
        if _run(command) != 0:
            print(f"\n{RED}✗ Stopping here: step {i} ({name}) did not "
                  f"succeed.{OFF}\n  The following steps were not "
                  f"run: you can launch them by hand from the menu.")
            return
    print(f"\n{GREEN}✓ Full run finished:{OFF} data updated, signal "
          f"computed, backtest redone and dashboard opened in the browser.")


def main() -> int:
    while True:
        print(MENU)
        try:
            choice = input("  Choose a number and press Enter › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSee you soon!")
            return 0
        if choice == "0":
            print("See you soon!")
            return 0
        elif choice == "1":
            path = input("  Path of the data folder or file › ").strip().strip('"')
            _run(["import-history", "--from", path])
        elif choice == "2":
            _run(["update-data"])
        elif choice == "3":
            _run(["signals"])
        elif choice == "4":
            _run(["backtest"])
        elif choice == "5":
            _run(["portfolio"])
        elif choice == "6":
            date_s = input("  Deposit date (YYYY-MM-DD, empty = today) › ").strip()
            amount = input("  Amount in euro › ").strip()
            _run(["deposit", "--date", date_s, "--amount", amount])
        elif choice == "7":
            _run(["legs"])
            g = input("\n  Which leg to change? (market / volatility / tdex, empty = none) › ").strip().lower()
            if g in {"market", "volatility", "tdex"}:
                ed = input("  Enable or disable? (e = enable, d = disable) › ").strip().lower()
                if ed.startswith("e"):
                    _run(["legs", "--enable", g])
                elif ed.startswith("d"):
                    _run(["legs", "--disable", g])
                else:
                    print("  Nothing changed.")
        elif choice == "8":
            _run(["check"])
        elif choice == "9":
            _run(["setup"])
        elif choice == "10":
            _run(["calibrate-longvol"])
            f = input("\n  Do you want to apply a frequency? (e.g. 20, empty = no) › ").strip().rstrip("%")
            if f:
                _run(["calibrate-longvol", "--apply", f])
        elif choice == "11":
            _run(["config-ui"])
        elif choice == "12":
            _run(["test-notify"])
        elif choice == "13":
            _run(["dashboard"])
        elif choice == "14":
            print("  What do you want to calibrate?\n"
                  "    1 = ENTRIES  (when to re-enter after a trend-filter exit)\n"
                  "    2 = EXITS    (confirmation thresholds of the trend filter)\n"
                  "    3 = Classic Monte Carlo (classic exit thresholds)")
            c = input("  Choose 1, 2 or 3 › ").strip()
            if c in {"1", "2"}:
                part = "entries" if c == "1" else "exits"
                _run(["calibrate-grid", "--part", part])
                r = input("\n  Write these thresholds to params.yaml (and switch to "
                          "market_mode: trend)? (y/n, empty = n) › ").strip().lower()
                if r in {"y", "yes"}:
                    _run(["calibrate-grid", "--part", part, "--apply", "yes",
                          "--no-browser"])
                input(f"\n{DIM}  Press Enter to go back to the menu…{OFF}")
                continue
            if c != "3":
                print("  Invalid choice.")
                input(f"\n{DIM}  Press Enter to go back to the menu…{OFF}")
                continue
            n = input("  How many Monte Carlo samples? (empty = 500) › ").strip() or "500"
            e = input("  Also calibrate the ENTRY thresholds? (y/n, empty = n) › ").strip().lower()
            extra = ["--entries"] if e in {"y", "yes"} else []
            _run(["calibrate-market", "--montecarlo", n] + extra)
            r = input("\n  Apply a candidate? (rank 1-10, 'best' or "
                      "'recommended', empty = no) › ").strip()
            if r:
                _run(["calibrate-market", "--montecarlo", n, "--apply", r] + extra)
        elif choice == "15":
            print("  Profiles: 1 = fast (~1 min) · 2 = normal (~4 min) · "
                  "3 = deep (~20 min)")
            lvl = {"1": "fast", "2": "normal", "3": "deep"}.get(
                input("  Which profile? (empty = normal) › ").strip(), "normal")
            s = input("  Also try moving the engine parameters? "
                      "(y/n, empty = n) › ").strip().lower()
            extra = ["--sensitivity"] if s in {"y", "yes"} else []
            _run(["robustness", "--level", lvl] + extra)
        elif choice == "16":
            _full_run()
        else:
            print("  Invalid choice.")
        input(f"\n{DIM}  Press Enter to go back to the menu…{OFF}")


if __name__ == "__main__":
    sys.exit(main())
