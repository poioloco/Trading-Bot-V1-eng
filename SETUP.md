# Setup guide

A short, step-by-step guide to get the bot running on real data.

## 1. What you need

- **Python 3.11 or newer** ([python.org](https://www.python.org/downloads/)).
  On Windows, tick **"Add python.exe to PATH"** during installation.
- About 20 minutes for the first setup.
- Optional: a free **Alpaca** account (ETF prices) and a **Telegram** bot
  (notifications). Neither is essential to try the bot: the demo needs no
  keys at all, and on real data the bot uses Stooq for ETF prices if
  Alpaca is not set up. Everything else is free and needs no key.

## 2. First launch

1. Download the repository (green **Code** button → *Download ZIP*) and unzip it.
2. Double-click **`START.bat`** (Windows), **`start.command`** (macOS) or
   **`start.sh`** (Linux). On macOS/Linux, if it does not start, run once:
   `chmod +x start.command start.sh`.
3. The first launch creates a virtual environment and installs the
   dependencies (about a minute). Then the numbered menu appears.

From now on you only ever use this menu: type a number and press Enter.

## 3. Optional: try the demo first

Open `params.yaml`, set `mode: demo` in the `data:` block, then run menu
item **16 (Full run)**. The bot invents a synthetic price history so you can
see every screen offline. Demo prices are fake and change every time you
run *Update the data*. Set `mode: live` again before using real data.

## 4. Get your keys

Run menu item **9 (API key setup)**: it creates a file called `.env` in the
bot folder. Open it with a text editor, paste each key after the `=` sign
and save. Leave empty what you do not use.

| Key | How to get it |
|-----|---------------|
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | Create a free account at alpaca.markets (paper trading is fine) and generate an API key in the dashboard. Without these keys the bot uses Stooq for ETF prices. |
| `TELEGRAM_BOT_TOKEN` | In Telegram, write to **@BotFather**, send `/newbot` and follow the steps: it gives you the token. |
| `TELEGRAM_CHAT_ID` | Send any message to your new bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser (with your token): the number after `"chat":{"id":` is your chat ID. |
| `EMAIL_SMTP_USER`, `EMAIL_SMTP_PASSWORD` | Optional, for email notifications. With Gmail use an **app password** (myaccount.google.com/apppasswords, requires 2-step verification), not your normal password. Also set `email_enabled: true` in `params.yaml`. |
| `FRED_API_KEY` | Not needed. |

`.env` is never uploaded to GitHub (it is in `.gitignore`). Never share it.

## 5. Load the data (first time only)

1. Make sure `params.yaml` has `mode: live` in the `data:` block (the default).
2. Run menu item **2 (Update the data)**. The bot downloads the history of the
   40 ETFs, the S&P 500, the VIX and the TDEX index. The first time this
   takes a few minutes. Afterwards it only adds the missing days.
3. Run menu item **8 (Check)**. It marks what is ready with ✓ and what is
   missing with !. A missing Alpaca or Telegram key is only a warning if
   you do not use them.

The free sources do not cover the full history back to 2005 for every
series. For example, FRED publishes only the last 10 years of the S&P 500.
That is enough for the daily signal. For a backtest over the full history,
import your own files with menu item **1 (Import the history)** before
step 2. The formats are described in the README.

## 6. Daily use

Run menu item **16 (Full run)** once a day. It updates the data, computes
the signal, redoes the backtest and opens the dashboard in your browser.

Run it **after about 10:00 New York time (16:00 in Central Europe)**. FRED
publishes the VIX and S&P 500 close of a day only on the following business
day, around 09:40 New York time. If you run it earlier, the bot computes the
signal on the last complete day and tells you so.

To record money you add to the portfolio simulation, use menu item
**6 (Add a deposit)**.

## 7. Notifications

After filling in the Telegram (or email) keys, run menu item
**12 (Test the notifications)** to check that the message arrives.
In this version the daily signal is shown in the menu and in the dashboard.
Automatic sending of each day's signal is not connected to the menu yet.

## 8. If something goes wrong

| What you see | What to do |
|--------------|------------|
| "Local history EMPTY" | Run menu item 2 (Update the data). |
| "Not yet published by the source: ^SPX, ^VIX" | Normal before ~10:00 New York time. Run it again later. |
| Signal dated the previous day | Same reason: the last day is not published yet. |
| Some symbols failed to download | Run item 2 again later. Missing Alpaca keys → it falls back to Stooq. |
| Telegram test fails | Check token and chat ID, and that you sent at least one message to your bot. |
| You changed `params.yaml` and something broke | The previous version is saved as `params.yaml.bak`. |
| Numbers change at every run | You are in `mode: demo`. Switch to `mode: live`. |

To back up everything, copy `data/market.duckdb` and `params.yaml`.
