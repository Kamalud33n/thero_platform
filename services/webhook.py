"""
Outbound webhooks: thero -> Laravel (MedNova Care)
"""
import asyncio
import datetime
import logging
import os
from typing import Any, Dict, Optional

import httpx

from auth import (
    results_webhook_headers,
    schedule_webhook_headers,
    RESULTS_WEBHOOK_SECRET,
    SCHEDULE_WEBHOOK_SECRET,
)

logger = logging.getLogger("thero.webhook")

MEDNOVA_RESULTS_WEBHOOK_URL  = os.getenv("MEDNOVA_RESULTS_WEBHOOK_URL")
MEDNOVA_SCHEDULE_WEBHOOK_URL = os.getenv("MEDNOVA_SCHEDULE_WEBHOOK_URL")

# Used to turn a relative join_url (e.g. "/room/ROOM-XXXX?token=...") into
# an absolute link before it's sent to Laravel, which is on a different
# domain — see .env's comment on this var. No trailing slash expected.
THERO_PUBLIC_BASE_URL = os.getenv("THERO_PUBLIC_BASE_URL", "").rstrip("/")

REQUEST_TIMEOUT_SECONDS = 10.0

# Retry delays for both directions — 6 attempts total (1 initial + 5
# retries), cumulative sleep ≈ 5.75 min, plus up to 6×10s of request time
# in the worst case (every attempt timing out) ≈ 6 min worst case overall,
# matching the "~6-minute worst case retry schedule" both call sites'
# comments reference. Exponential-ish, capped, so a persistently-down
# Laravel endpoint doesn't hammer them.
RETRY_DELAYS_SECONDS = [5, 15, 30, 60, 120, 120]


async def _post_with_retries(url: str, payload: Dict[str, Any], headers: Dict[str, str], label: str) -> None:
    """
    Shared delivery loop for both webhook directions. Never raises —
    callers fire this via asyncio.create_task and have nothing useful to
    do with an exception (the response that triggered the webhook has
    already gone back to the client). All failure information goes to the
    logger instead.
    """
    attempt = 0
    delays = RETRY_DELAYS_SECONDS
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        while True:
            attempt += 1
            try:
                resp = await client.post(url, json=payload, headers=headers)
                if 200 <= resp.status_code < 300:
                    logger.info("%s webhook delivered (attempt %d, status %d)", label, attempt, resp.status_code)
                    return
                # 4xx from Laravel means our payload/auth is wrong, not a
                # transient failure — retrying the exact same request
                # won't help, so don't burn through the retry budget.
                if 400 <= resp.status_code < 500:
                    logger.error(
                        "%s webhook rejected (attempt %d, status %d) — not retrying, "
                        "this looks like a payload/auth problem: %s",
                        label, attempt, resp.status_code, resp.text[:500],
                    )
                    return
                logger.warning("%s webhook failed (attempt %d, status %d)", label, attempt, resp.status_code)
            except httpx.HTTPError as exc:
                logger.warning("%s webhook attempt %d raised %s: %s", label, attempt, type(exc).__name__, exc)

            if attempt > len(delays):
                logger.error("%s webhook gave up after %d attempts", label, attempt)
                return
            await asyncio.sleep(delays[attempt - 1])


def _iso(dt: Optional[datetime.datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


# ─── Direction 1: session results (item 27) ────────────────────────────────

def build_session_result_payload(
    sess,
    target_rom: Optional[float] = None,
    target_reps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Snapshots every scalar column this webhook needs off a SessionModel
    ORM instance into a plain dict.

    MUST be called while `sess` is still attached to a live session (i.e.
    before db.commit() / before the enclosing `with get_db()` block
    exits) — SQLAlchemy expires an instance's attributes on commit by
    default, and once the session/context is gone, touching any attribute
    on a detached instance raises DetachedInstanceError. Reading everything
    into a plain dict here, synchronously, sidesteps that entirely: the
    dict has no ties back to the ORM/session and is safe to hand to
    asyncio.create_task() after the caller's `with get_db()` block ends.

    consultation_id is read straight off SessionModel (not TelehealthRoom)
    — bridge mode copies it onto the SessionModel row at creation time
    (routers/bridge.py) specifically so this function doesn't need to join
    back to the room to find it. NULL for every non-bridge session.

    target_rom / target_reps are NOT SessionModel columns — they're the
    doctor-set targets from the bridge request (TelehealthRoom.target_rom /
    .target_reps), which only the caller has (SessionModel only stores what
    was actually achieved). Passed in explicitly rather than looked up here
    so this function doesn't need a TelehealthRoom lookup of its own; both
    default to None for every non-bridge caller, which has no such room.

    Field names below (measured_rom, reps_completed, completed_at) match
    Tasks_Kamal_Python.md's Task 2 example payload — renamed from this
    function's earlier average_rom/completed_reps/end_time column names,
    which Laravel was never wired to expect.
    """
    return {
        "session_id":          sess.id,
        "patient_id":          sess.patient_id,
        "consultation_id":     sess.consultation_id,
        "exercise_type":       sess.exercise_type,
        "affected_side":       sess.affected_side,
        "status":              sess.status,
        "end_reason":          sess.end_reason,
        "start_time":          _iso(sess.start_time),
        "completed_at":        _iso(sess.end_time),
        "duration_seconds":    sess.duration_seconds,
        "total_reps":          sess.total_reps,
        "reps_completed":      sess.completed_reps,
        "target_reps":         target_reps,
        "accuracy_percentage": sess.accuracy_percentage,
        "measured_rom":        sess.average_rom,
        "target_rom":          target_rom,
        "incorrect_movements": sess.incorrect_movements,
        "stability_score":     sess.stability_score,
        "balance_score":       sess.balance_score,
        "movement_smoothness": sess.movement_smoothness,
        "fatigue_estimation":  sess.fatigue_estimation,
        "recovery_score":      sess.recovery_score,
    }


async def send_session_result_webhook(payload: Dict[str, Any]) -> None:
    """Fire-and-forget — see module docstring. No-ops (logged once) if
    MEDNOVA_RESULTS_WEBHOOK_URL / THERO_RESULTS_WEBHOOK_SECRET aren't
    configured yet, which is the current state as of this writing (Nada
    hasn't sent either)."""
    if not MEDNOVA_RESULTS_WEBHOOK_URL or not RESULTS_WEBHOOK_SECRET:
        logger.info(
            "Session-results webhook skipped for session %s — "
            "MEDNOVA_RESULTS_WEBHOOK_URL/THERO_RESULTS_WEBHOOK_SECRET not configured yet",
            payload.get("session_id"),
        )
        return
    await _post_with_retries(
        MEDNOVA_RESULTS_WEBHOOK_URL, payload, results_webhook_headers(), label="session-results",
    )


# ─── Direction 2: session scheduled (Remote / Self Training only) ─────────

def build_session_scheduled_payload(room, join_url_path: str) -> Dict[str, Any]:
    """
    Built from a TelehealthRoom right after it's created by
    telehealth.create_room() — same "read scalars into a plain dict before
    the DB session can expire them" reasoning as
    build_session_result_payload() above.

    join_url_path is the relative path (e.g. "/join/{room_id}?token=...")
    the caller already built for the patient; this turns it absolute using
    THERO_PUBLIC_BASE_URL since Laravel is on a different domain and can't
    do anything with a relative path. If THERO_PUBLIC_BASE_URL isn't set,
    the relative path is sent as-is rather than raising — the webhook as a
    whole is still a no-op until the URL/secret are configured anyway (see
    send_schedule_webhook), so this only matters once someone's actually
    wiring the base URL up too.

    Not called for mode="bridge" rooms — routers/bridge.py returns
    therapist_url/patient_url directly in its own HTTP response, since
    Laravel is the caller that requested the room in the first place and
    already knows about it.
    """
    join_url = join_url_path
    if THERO_PUBLIC_BASE_URL and join_url_path.startswith("/"):
        join_url = f"{THERO_PUBLIC_BASE_URL}{join_url_path}"

    return {
        "room_id":       room.id,
        "patient_id":    room.patient_id,
        "mode":          room.mode,
        "exercise_type": room.exercise_type,
        "scheduled_at":  _iso(room.scheduled_at),
        "expires_at":    _iso(room.expires_at),
        "join_url":      join_url,
    }


async def send_session_scheduled_webhook(payload: Dict[str, Any]) -> None:
    """Fire-and-forget — see module docstring. No-ops (logged once) if
    MEDNOVA_SCHEDULE_WEBHOOK_URL / MEDNOVA_SCHEDULE_WEBHOOK_SECRET aren't
    configured yet (Laravel's receiving endpoint doesn't exist yet as of
    this writing — "inime than build pannuvanga")."""
    if not MEDNOVA_SCHEDULE_WEBHOOK_URL or not SCHEDULE_WEBHOOK_SECRET:
        logger.info(
            "Session-scheduled webhook skipped for room %s — "
            "MEDNOVA_SCHEDULE_WEBHOOK_URL/MEDNOVA_SCHEDULE_WEBHOOK_SECRET not configured yet",
            payload.get("room_id"),
        )
        return
    await _post_with_retries(
        MEDNOVA_SCHEDULE_WEBHOOK_URL, payload, schedule_webhook_headers(), label="session-scheduled",
    )