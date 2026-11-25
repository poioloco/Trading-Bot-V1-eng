"""APScheduler-based job runner: EOD + hourly intraday signal runs.

Dedup: an event already notified today is not re-sent by later intraday
runs (state kept in a small JSON file next to the bot).
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from ..notify.telegram import TelegramNotifier, format_report
from ..runner.signal_runner import SignalReport, SignalRunner

logger = logging.getLogger(__name__)


class EventDedup:
    def __init__(self, path: str | Path = "state/sent_events.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def filter_new(self, rep: SignalReport) -> dict[str, bool]:
        state = self._load()
        today = date.today().isoformat()
        sent = set(state.get(today, []))
        return {k: v for k, v in rep.events.items() if v and k not in sent}

    def mark_sent(self, rep: SignalReport) -> None:
        state = self._load()
        today = date.today().isoformat()
        sent = set(state.get(today, []))
        sent |= {k for k, v in rep.events.items() if v}
        self.path.write_text(json.dumps({today: sorted(sent)}))


class BotScheduler:
    def __init__(
        self,
        runner: SignalRunner,
        notifier: TelegramNotifier,
        eod_cron: str = "30 22 * * mon-fri",       # 22:30 UTC ~ after the US close
        intraday_cron: str = "30 14-20 * * mon-fri",  # hourly during the US session (UTC)
        intraday_enabled: bool = True,
        dedup: EventDedup | None = None,
        timezone: str = "UTC",
    ) -> None:
        self.runner = runner
        self.notifier = notifier
        self.eod_cron = eod_cron
        self.intraday_cron = intraday_cron
        self.intraday_enabled = intraday_enabled
        self.dedup = dedup or EventDedup()
        self.timezone = timezone
        self.scheduler = BlockingScheduler(timezone=timezone)

    # ------------------------------------------------------------------ #
    def job(self, kind: str) -> None:
        logger.info("Run %s started", kind)
        try:
            rep = self.runner.run()
        except Exception:
            logger.exception("Run %s failed", kind)
            return
        new_events = self.dedup.filter_new(rep)
        if kind == "eod" and self.notifier.notify_mode == "always":
            self.notifier.send_report(rep)
            self.dedup.mark_sent(rep)
        elif new_events:
            self.notifier.send_text(format_report(rep))
            self.dedup.mark_sent(rep)
        else:
            logger.info("Run %s: no new events to notify", kind)

    def start(self) -> None:
        self.scheduler.add_job(
            self.job, CronTrigger.from_crontab(self.eod_cron, timezone=self.timezone),
            args=["eod"], id="eod",
        )
        if self.intraday_enabled:
            self.scheduler.add_job(
                self.job,
                CronTrigger.from_crontab(self.intraday_cron, timezone=self.timezone),
                args=["intraday"], id="intraday",
            )
        logger.info(
            "Scheduler running (eod='%s', intraday='%s', tz=%s)",
            self.eod_cron,
            self.intraday_cron if self.intraday_enabled else "off",
            self.timezone,
        )
        self.scheduler.start()
