"""Bot notifications: Telegram + email, behind a single interface.

`build_notifier(params)` reads the `notifications:` block of params.yaml
and returns a MultiNotifier with the ENABLED channels. A channel that fails
does not block the others (redundancy: if Telegram is down, the email still
goes out, and vice versa).
"""
from __future__ import annotations

import logging

from ..runner.signal_runner import SignalReport
from .email import EmailNotifier
from .telegram import TelegramNotifier

logger = logging.getLogger(__name__)


class MultiNotifier:
    """Forwards the report to all enabled channels; True if AT LEAST ONE succeeds."""

    def __init__(self, channels: list, notify_mode: str = "events") -> None:
        self.channels = channels
        self.notify_mode = notify_mode

    def should_send(self, rep: SignalReport) -> bool:
        return self.notify_mode == "always" or rep.has_events

    def send_report(self, rep: SignalReport) -> bool:
        if not self.should_send(rep):
            return False
        ok = False
        for ch in self.channels:
            try:
                ok = bool(ch.send_report(rep)) or ok
            except Exception:
                logger.exception("Channel %s failed", type(ch).__name__)
        return ok

    def send_text(self, text: str) -> bool:
        ok = False
        for ch in self.channels:
            try:
                ok = bool(ch.send_text(text)) or ok
            except Exception:
                logger.exception("Channel %s failed", type(ch).__name__)
        return ok


def build_notifier(p: dict) -> MultiNotifier:
    """Builds the channels from the `notifications:` block of params.yaml."""
    n = p.get("notifications", {}) or {}
    mode = "always" if str(n.get("mode", "always")) == "always" else "events"
    channels: list = []
    if n.get("telegram_enabled", True):
        channels.append(TelegramNotifier(notify_mode=mode))
    if n.get("email_enabled", False):
        e = n.get("email", {}) or {}
        channels.append(EmailNotifier(
            smtp_server=str(e.get("smtp_server", "smtp.gmail.com")),
            smtp_port=int(e.get("smtp_port", 587)),
            to_addr=str(e.get("recipient", "") or ""),
            notify_mode=mode,
        ))
    return MultiNotifier(channels, notify_mode=mode)
