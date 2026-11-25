"""Email channel and MultiNotifier.

Email is REDUNDANCY for Telegram: a failing channel must never block the
other. SMTP is fake (mock): no network in the tests.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.notify import MultiNotifier, build_notifier
from src.notify.email import EmailNotifier, _plain
from src.notify.telegram import TelegramNotifier
from src.runner.signal_runner import SignalReport


def _report(events: bool = True) -> SignalReport:
    return SignalReport(
        as_of=pd.Timestamp("2026-07-03"),
        quad=2, quad_name="Reflation",
        asset_signals={"TQQQ": "BUY", "UVIX": "HOLD"},
        events={"buy_spx": events, "sell_spx": False,
                "buy_vix": False, "sell_vix": False},
        metrics={"spx": 6100.0, "vix": 14.2, "tdex": None,
                 "vrp_smooth": 3.1, "rsi_smooth": 1.2, "hv": 11.0,
                 "tdex_smooth": None},
    )


class FakeSMTP:
    """Captures the message instead of sending it."""
    sent: list = []

    def __init__(self):
        FakeSMTP.sent = []

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def starttls(self): self.tls = True
    def login(self, user, pwd): self.creds = (user, pwd)
    def send_message(self, msg): FakeSMTP.sent.append(msg)


@pytest.fixture
def email_env(monkeypatch):
    monkeypatch.setenv("EMAIL_SMTP_USER", "user@example.com")
    monkeypatch.setenv("EMAIL_SMTP_PASSWORD", "secret")


def test_email_sends_with_event_subject(email_env, monkeypatch):
    n = EmailNotifier(to_addr="dest@example.com", notify_mode="events")
    monkeypatch.setattr(n, "_connect", lambda: FakeSMTP())
    assert n.send_report(_report(events=True)) is True
    (msg,) = FakeSMTP.sent
    assert msg["To"] == "dest@example.com"
    assert "BUY SPX" in msg["Subject"]
    body = msg.get_content()
    assert "TQQQ" in body and "*" not in body        # no Markdown


def test_email_empty_recipient_uses_user(email_env, monkeypatch):
    n = EmailNotifier(notify_mode="always")
    monkeypatch.setattr(n, "_connect", lambda: FakeSMTP())
    n.send_report(_report(events=False))
    assert FakeSMTP.sent[0]["To"] == "user@example.com"


def test_email_respects_events_mode(email_env):
    n = EmailNotifier(notify_mode="events")
    assert n.send_report(_report(events=False)) is False   # nothing sent


def test_email_without_keys_raises(monkeypatch):
    monkeypatch.delenv("EMAIL_SMTP_USER", raising=False)
    monkeypatch.delenv("EMAIL_SMTP_PASSWORD", raising=False)
    with pytest.raises(RuntimeError):
        EmailNotifier(notify_mode="always").send_text("x")


def test_plain_strips_markdown():
    assert _plain("*Bot* — `TQQQ` _note_") == "Bot — TQQQ note"


# ---------------------------------------------------------------------- #
class OkChannel:
    def __init__(self): self.called = False
    def send_report(self, rep): self.called = True; return True
    def send_text(self, t): self.called = True; return True


class KoChannel:
    def send_report(self, rep): raise RuntimeError("channel down")
    def send_text(self, t): raise RuntimeError("channel down")


def test_multinotifier_one_channel_down_does_not_block(caplog):
    ok = OkChannel()
    multi = MultiNotifier([KoChannel(), ok], notify_mode="always")
    assert multi.send_report(_report()) is True     # the other one went out
    assert ok.called


def test_multinotifier_respects_mode():
    ok = OkChannel()
    multi = MultiNotifier([ok], notify_mode="events")
    assert multi.send_report(_report(events=False)) is False
    assert not ok.called


def test_build_notifier_reads_params():
    p = {"notifications": {"telegram_enabled": True, "email_enabled": True,
                           "mode": "events",
                           "email": {"smtp_server": "smtp.example.com",
                                     "smtp_port": 465, "recipient": "a@b.c"}}}
    multi = build_notifier(p)
    kinds = {type(c) for c in multi.channels}
    assert kinds == {TelegramNotifier, EmailNotifier}
    assert multi.notify_mode == "events"
    em = next(c for c in multi.channels if isinstance(c, EmailNotifier))
    assert (em.smtp_server, em.smtp_port, em.to_addr) == ("smtp.example.com", 465, "a@b.c")


def test_build_notifier_telegram_only_default():
    multi = build_notifier({"notifications": {"telegram_enabled": True,
                                              "mode": "always"}})
    assert [type(c) for c in multi.channels] == [TelegramNotifier]
    assert multi.notify_mode == "always"
