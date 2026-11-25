"""EMAIL channel for the bot's signals (redundancy alongside Telegram).

Uses SMTP with STARTTLS, standard library only (no new dependencies).
Credentials live in the `.env` file (never in params.yaml):
  EMAIL_SMTP_USER     = SMTP user (e.g. yourname@gmail.com)
  EMAIL_SMTP_PASSWORD = SMTP password. With Gmail you need an "app
                        password" (myaccount.google.com/apppasswords).
Server, port and recipient live in params.yaml (notifications.email block).
The message text is the same as Telegram's, without the Markdown syntax.
"""
from __future__ import annotations

import logging
import os
import re
import smtplib
from email.message import EmailMessage

from ..runner.signal_runner import SignalReport
from .telegram import format_report

logger = logging.getLogger(__name__)


def _plain(text: str) -> str:
    """Strips Telegram's Markdown formatting (*bold*, `code`, _it_)."""
    return re.sub(r"[*`_]", "", text)


class EmailNotifier:
    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        to_addr: str = "",
        user_env: str = "EMAIL_SMTP_USER",
        password_env: str = "EMAIL_SMTP_PASSWORD",
        notify_mode: str = "events",
        timeout: int = 30,
    ) -> None:
        self.smtp_server = smtp_server
        self.smtp_port = int(smtp_port)
        self.to_addr = to_addr
        self.user_env = user_env
        self.password_env = password_env
        self.notify_mode = notify_mode
        self._timeout = timeout

    # ------------------------------------------------------------------ #
    def should_send(self, rep: SignalReport) -> bool:
        return self.notify_mode == "always" or rep.has_events

    def send_report(self, rep: SignalReport) -> bool:
        if not self.should_send(rep):
            logger.info("No events and notify_mode=events: no email")
            return False
        fired = [k.replace("_", " ").upper() for k, v in rep.events.items() if v]
        subj = (f"Tactical Bot {rep.as_of.date().isoformat()} — "
                + (", ".join(fired) if fired else "no events"))
        return self.send_text(_plain(format_report(rep)), subject=subj)

    def send_text(self, text: str, subject: str = "Tactical Bot") -> bool:
        user = os.environ.get(self.user_env)
        password = os.environ.get(self.password_env)
        if not user or not password:
            raise RuntimeError(
                f"Missing environment variables: {self.user_env} / "
                f"{self.password_env} (see 'setup')"
            )
        msg = EmailMessage()
        msg["From"] = user
        msg["To"] = self.to_addr or user
        msg["Subject"] = subject
        msg.set_content(text)
        with self._connect() as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        logger.info("Email sent to %s", msg["To"])
        return True

    def _connect(self) -> smtplib.SMTP:
        """Separate so it can be mocked in tests."""
        return smtplib.SMTP(self.smtp_server, self.smtp_port,
                            timeout=self._timeout)
