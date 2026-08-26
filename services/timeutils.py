"""
Central time handling for thero.

Everything internal — DB columns, room expiry checks, token expiry,
WebSocket "ts" fields, the /api/health timestamp — is UTC. This makes
timestamps identical no matter which region the app server or the
database happen to run in (dev laptop, staging box, cloud prod all
agree), instead of silently riding on whatever OS timezone the process
happens to be started under.

Two edges intentionally do NOT stay in raw UTC:

1. JSON/WebSocket timestamps sent to the browser (utcnow_iso()) carry an
   explicit "Z" suffix, so the browser's own `new Date(...)` parses them
   as UTC and renders them in whichever timezone the *viewer's device*
   is actually in. This is the standard, correct pattern for a web UI —
   better than baking one fixed timezone into the API, since a doctor
   and patient could be looking at the same session from different
   places.

2. Server-rendered PDF reports (services/report_builder.py) have no
   browser to do that conversion, so those explicitly localize to
   DISPLAY_TIMEZONE — the clinic's own timezone (MedNova Care is in
   Muscat, Oman → Gulf Standard Time), configured once via env var. This
   is NOT auto-detected from the network/server IP — that's unreliable
   for a backend (cloud IP != clinic location) and would make report
   timestamps silently drift if the app ever moves regions.

NOTE: MySQL has its own default-value clock, independent of Python —
several columns in models.py default via `func.now()` (a SQL-side
NOW(), not this module). For that to also be UTC, the MySQL server's
own time_zone must be pinned to UTC too — see the `db` service's
`command: --default-time-zone=+00:00` in docker-compose.yml. If you
ever add a new column with a func.now() default, or point this app at
a MySQL instance not covered by that compose file, make sure the DB's
time_zone is UTC or these two clocks will drift apart again.
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
