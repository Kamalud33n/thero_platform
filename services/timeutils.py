"""
Central time handling for thero.
"""
import datetime as _dt
import os
from zoneinfo import ZoneInfo

DISPLAY_TIMEZONE = os.getenv("DISPLAY_TIMEZONE", "Asia/Muscat")


def utcnow() -> _dt.datetime:
    """Naive UTC datetime — use for anything written to a `DateTime`
    column in models.py (those columns are naive, so this matches)."""
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)


def utcnow_iso() -> str:
    """UTC timestamp as an ISO string with an explicit 'Z' suffix, for
    JSON/WebSocket payloads sent to the browser — see module docstring,
    point 1."""
    return utcnow().isoformat() + "Z"


def to_display(dt: _dt.datetime | None) -> _dt.datetime | None:
    """Convert a naive-UTC datetime (as stored in the DB) into
    DISPLAY_TIMEZONE, for anything rendered server-side with no browser
    to localize it (PDF reports). Returns a naive datetime in local
    time so it drops straight into existing strftime()/Jinja formatting."""
    if dt is None:
        return None
    aware_utc = dt.replace(tzinfo=_dt.timezone.utc)
    local = aware_utc.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    return local.replace(tzinfo=None)