# Tactical Trading Bot — V1

A daily **signal** bot (BUY / SELL / HOLD) for US equities and volatility. It
combines two models:

- a **bottom-up regime model** built on 40 ETFs, which classifies every day
  into one of four macro quadrants (Disinflation, Reflation, Stagflation,
  Deflation);
- a **VIX model** with three voting legs — volatility risk premium (VRP),
  relative strength (RSI) and the TDEX tail-risk index — whose thresholds
  depend on the current quadrant.

Everything runs locally: prices live in a single DuckDB file, the engines
are incremental (they only process new days) and the backtest replays
history with exactly the same code used live.

> **Disclaimer.** This project produces *signals only*: it never places
> orders. It is not financial advice. Back-tested results do not guarantee
> future returns — read the robustness report before trusting any number.

---

## Quick start (no commands needed)

First time? Follow the step-by-step **[setup guide](SETUP.md)**.

1. Install **Python 3.11+**.
2. Double-click the launcher for your system:
   **`START.bat`** (Windows) · **`start.command`** (macOS) · **`start.sh`** (Linux).
   On the first run it creates a virtual environment and installs the
   dependencies (about a minute). On macOS/Linux, if the launcher does not
   start, make it executable once: `chmod +x start.command start.sh`.
3. A **numbered menu** appears: type the number of an action and press Enter.

| # | Action | What it does |
|---|--------|--------------|
| 1 | Import the history | Loads price history you already have (once) |
| 2 | Update the data | Fills only the missing days (or generates the demo data) |
| 3 | Compute today's signal | Shows today's regime and signal |
| 4 | Backtest | Replays the whole history with the live code, HTML report |
| 5 | Portfolio / trade log | Simulated P&L on your deposits |
| 6 | Add a deposit | Records a deposit in `contributions.csv` |
| 7 | Voting legs | Enable/disable the market · volatility · TDEX legs |
| 8 | Check | "Is everything ready?" — archive and key status |
| 9 | API key setup | Creates the `.env` template for real data and notifications |
| 10 | Calibrate the volatility leg | Long-vol thresholds (majority mode) |
| 11 | Control panel | Edit `params.yaml` from the browser, with preview |
| 12 | Test the notifications | Sends a test message on Telegram / email |
| 13 | Dashboard | Signal, votes and performance in the browser |
| 14 | Calibrate the market leg | Grid calibration of entries / exits |
| 15 | Robustness | Walk-forward, overfitting (PBO, DSR), bootstrap, permutation tests |
| 16 | Full run | 2 → 3 → 4 → 13 in a row |

## Try it without data or keys (demo mode)

In `params.yaml` set:

```yaml
data:
  mode: demo
```

then run **2 → 3 → 4 → 13** (or menu item 16). The bot generates a
**synthetic** history so you can see everything working offline. Demo
numbers are *not* market data.

## Using real data

Set `data: mode: live` (the default) and either:

- run **Update the data** (item 2): the bot downloads the missing days —
  ETFs from Alpaca (needs free API keys) with Stooq as a fallback, and
  `^SPX`, `^VIX` and TDEX from FRED (no key needed);
- or **Import the history** (item 1) from files you already have:
  - a **folder** with one CSV per symbol (`QUAL.csv`, `SPX.csv`, `TDEX.csv`…)
    with columns `Date, Close` (indices) or `Date, Open, High, Low, Close, Volume` (ETFs);
  - an **Excel workbook** with one sheet per symbol (the sheet name is the symbol);
  - a **single CSV** with columns `symbol, date, close[, open, high, low, volume]`.

  `SPX` / `VIX` are mapped to `^SPX` / `^VIX` automatically. Downloading 20
  years of 40 tickers at once is usually rate-limited, so importing the old
  history once and letting the bot fill only the new days is the
  recommended path.

Everything ends up in `data/market.duckdb`. Imported data is never
overwritten by the incremental update. To back up, copy that file.

## API keys

Keys are **never** stored in the code or in `params.yaml`. Run menu item 9
(or `python cli.py setup`) to create a `.env` file, or copy `.env.example`
to `.env`, then fill in what you need:

| Variable | Needed for |
|----------|-----------|
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | ETF prices in live mode (free Alpaca account) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram notifications (token from @BotFather) |
| `FRED_API_KEY` | Optional — the FRED CSV endpoint works without it |
| `EMAIL_SMTP_USER`, `EMAIL_SMTP_PASSWORD` | Optional email notifications (Gmail needs an app password) |

`.env` is listed in `.gitignore`: it will not be committed.

## Parameters

All model numbers (thresholds, moving-average windows, the 40 ETFs per
quadrant, voting method, notifications…) are in **`params.yaml`**, each with
a plain-language explanation. You can edit the file directly or use the
**Control panel** (item 11), which validates the values, shows a preview of
the changes and keeps a backup in `params.yaml.bak`.

## Notifications

The bot has a **Telegram** channel and, optionally, an **email** channel as
redundancy (if one channel fails the other still goes out). Enable them in
the `notifications:` block of `params.yaml` and put the credentials in
`.env`. Item 12 sends a test message on every active channel. In this
version the daily signal is shown in the menu and in the dashboard:
automatic sending of each day's signal is not connected to the menu yet.

## Command line

```bash
python cli.py update-data                 # fill/update the local history
python cli.py import-history --from PATH  # import your own history
python cli.py signals [--recompute]       # today's signal
python cli.py backtest [--to YYYY-MM-DD]  # full backtest + HTML report
python cli.py portfolio                   # consensus portfolio and trade log
python cli.py deposit --date YYYY-MM-DD --amount 500
python cli.py legs [--enable tdex | --disable volatility]
python cli.py calibrate-longvol           # long-vol thresholds (majority mode)
python cli.py calibrate-market [--montecarlo 500 [--apply recommended]]
python cli.py calibrate-grid --part entries|exits [--apply yes]
python cli.py robustness [--level fast|normal|deep] [--compare gate|majority|both]
python cli.py dashboard                   # build and open the dashboard
python cli.py check | setup | test-notify | config-ui
```

Reports and CSVs are written to `output/`.

## Project layout

```
cli.py, menu.py         entry points (command line and text menu)
params.yaml             all the model parameters
contributions.csv       your deposits (date, amount_eur) — example row included
src/data/               DuckDB store, point-in-time view, calendar, import, validation
src/providers/          Alpaca, Stooq, FRED data sources
src/regime/             40-ETF regime model
src/vix/                VIX model: indicators, per-quadrant parameters, signal generator
src/engine/             incremental (stateful) engines and daily pipeline
src/backtest/           backtester, calibrators, grid search, robustness battery, reports
src/portfolio/          consensus model and portfolio simulator
src/config/             params.yaml editor and browser control panel
src/notify/             Telegram and email channels
src/dashboard.py        HTML dashboard (ECharts)
tests/                  test suite (pytest)
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

The suite (180 tests) runs fully offline on synthetic data and includes
golden tests against the original Excel models, live-vs-backtest
equivalence tests and anti-lookahead (point-in-time) checks.
