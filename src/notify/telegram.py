"""Telegram delivery of SignalReports via the Bot API (sendMessage).

Config: token and chat id are read from environment variables (names in
the .env file). notify_mode:
  - "events" : send only when a BUY/SELL event fires (default)
  - "always" : send every run (heartbeat included)
"""
from __future__ import annotations

import logging
import os

import requests

from ..runner.signal_runner import SignalReport

logger = logging.getLogger(__name__)

API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def format_report(rep: SignalReport) -> str:
    lines = [
        f"📊 *Tactical Bot* — {rep.as_of.date().isoformat()}",
        f"Regime: quad {rep.quad} ({rep.quad_name})",
        "",
    ]
    icons = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⚪ HOLD"}
    for sym, action in rep.asset_signals.items():
        lines.append(f"{icons.get(action, action)}  `{sym}`")
    fired = [k.replace("_", " ").upper() for k, v in rep.events.items() if v]
    if fired:
        lines += ["", "⚡ Events: " + ", ".join(fired)]
    m = rep.metrics
    ctx = [f"SPX {m['spx']:.0f}", f"VIX {m['vix']:.2f}"]
    if m.get("tdex") is not None:
        ctx.append(f"TDEX {m['tdex']:.2f}")
    ctx += [f"VRP {m['vrp_smooth']:.2f}", f"RSI {m['rsi_smooth']:.2f}"]
    if m.get("tdex_smooth") is not None:
        ctx.append(f"TDEXz {m['tdex_smooth']:.2f}")
    if m.get("contango_next") is not None:
        ctx.append(f"ContangoNext {m['contango_next']*100:.2f}%")
    lines += ["", " | ".join(ctx)]
    if rep.notes:
        lines += ["", "_" + "; ".join(rep.notes) + "_"]
    return "\n".join(lines)


class TelegramNotifier:
    def __init__(
        self,
        bot_token_env: str = "TELEGRAM_BOT_TOKEN",
        chat_id_env: str = "TELEGRAM_CHAT_ID",
        notify_mode: str = "events",
        session: requests.Session | None = None,
        timeout: int = 20,
    ) -> None:
        self.bot_token_env = bot_token_env
        self.chat_id_env = chat_id_env
        self.notify_mode = notify_mode
        self._session = session or requests.Session()
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    def should_send(self, rep: SignalReport) -> bool:
        return self.notify_mode == "always" or rep.has_events

    def send_report(self, rep: SignalReport) -> bool:
        """Returns True if a message was sent."""
        if not self.should_send(rep):
            logger.info("No events and notify_mode=events: no message")
            return False
        return self.send_text(format_report(rep))

    def send_text(self, text: str) -> bool:
        token = os.environ.get(self.bot_token_env)
        chat_id = os.environ.get(self.chat_id_env)
        if not token or not chat_id:
            raise RuntimeError(
                f"Missing environment variables: {self.bot_token_env} / "
                f"{self.chat_id_env}"
            )
        resp = self._session.post(
            API_URL.format(token=token),
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        ok = bool(resp.json().get("ok"))
        if not ok:
            logger.error("Telegram answered ok=false: %s", resp.text[:200])
        return ok
