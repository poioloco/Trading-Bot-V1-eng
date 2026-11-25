"""NYSE trading calendar with the correct time zone.

Fixes two classic errors at the root:
  1. holidays: uses a real, up-to-date calendar (exchange_calendars library),
     not a static list that goes stale;
  2. off-by-one-day: the "current trading date" is the last US session
     already CLOSED according to New York time, not the UTC or local date.

So an EOD run launched at 22:00 CET (16:00 in New York, after the
bell) is labelled with the right trading date, even across daylight-saving
changes and holidays.
"""
from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

# Regular NYSE closing time (16:00 New York time). Half days close
# earlier (13:00): in those cases we still treat the market as closed
# only after 16:00 NY, so there is no risk of running early.
_CLOSE_HOUR_NY = 16


@lru_cache(maxsize=1)
def _cal():
    # wide range to cover from 2004 (the model starts in 2005) up to a few
    # years into the future, without falling back on weekday heuristics
    return xcals.get_calendar("XNYS", start="2004-01-01", end="2030-12-31")


def is_session(d: date) -> bool:
    """True if `d` is an NYSE trading day."""
    cal = _cal()
    ts = pd.Timestamp(d).normalize()
    if ts < cal.first_session or ts > cal.last_session:
        return ts.weekday() < 5  # outside the calendar range: weekday heuristic
    return cal.is_session(ts)


def last_session_on_or_before(d: date) -> date:
    """Last trading session <= d (today itself if it is a market day)."""
    cal = _cal()
    ts = pd.Timestamp(d).normalize()
    sessions = cal.sessions_in_range(ts - pd.Timedelta(days=10), ts)
    if len(sessions) == 0:
        # weekday fallback outside the calendar range
        while ts.weekday() >= 5:
            ts -= pd.Timedelta(days=1)
        return ts.date()
    return sessions[-1].date()


def previous_session(d: date) -> date:
    """Trading session strictly before d."""
    cal = _cal()
    ts = pd.Timestamp(d).normalize()
    sessions = cal.sessions_in_range(ts - pd.Timedelta(days=10), ts - pd.Timedelta(days=1))
    if len(sessions) == 0:
        ts -= pd.Timedelta(days=1)
        while ts.weekday() >= 5:
            ts -= pd.Timedelta(days=1)
        return ts.date()
    return sessions[-1].date()


def sessions_between(start: date, end: date) -> list[date]:
    """All trading sessions in [start, end]."""
    cal = _cal()
    s = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [ts.date() for ts in s]


def current_trading_date(now_utc: datetime | None = None) -> date:
    """The US trading date to use for an EOD run, given 'now' in UTC.

    Rule: if the close (16:00 NY) has already passed in New York and today
    is a market day, the date is TODAY; otherwise it is the last previous
    session. No more signals stamped with the wrong date.
    """
    if now_utc is None:
        now = pd.Timestamp.now(tz="UTC")
    else:
        ts = pd.Timestamp(now_utc)
        now = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    ny = now.tz_convert("America/New_York")
    today = ny.normalize().date()
    if is_session(today) and ny.hour >= _CLOSE_HOUR_NY:
        # market closed today: the trading date is today
        return today
    # market not closed yet (or holiday/weekend): last completed session
    return previous_session(today) if is_session(today) else last_session_on_or_before(today)


def missing_sessions(have: list[date], start: date, end: date) -> list[date]:
    """Trading sessions in [start, end] not present in `have` (for the
    automatic backfill of the days the bot was switched off)."""
    have_set = {pd.Timestamp(d).date() for d in have}
    return [d for d in sessions_between(start, end) if d not in have_set]
