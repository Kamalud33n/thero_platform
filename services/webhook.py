"""
outbound webhooks: thero -> Laravel / MedNova Care.

Two independent directions live in this module, each with its own secret
and URL (see auth.py module docstring for why they're deliberately not
shared):
  - session-results  (AUTH: auth.results_webhook_headers())  — existing,
    fired when a session finishes.
  - session-scheduled (AUTH: auth.schedule_webhook_headers()) — fired from
    telehealth.create_room() when a Remote/Self Training room is
    scheduled, so Laravel can route the join_url to the patient's MedNova
    account. Laravel's receiving endpoint doesn't exist yet as of this
    writing — get the real URL + secret from them once it does.

AUTH: uses auth.results_webhook_headers() — see that function's docstring
for an open question about whether the new secret is a plain shared
secret or part of an RSA signing scheme. This module assumes the former.
The same assumption applies to schedule_webhook_headers() below.
"""
import asyncio
import logging
import os
from typing import Any, Dict, Optional

import httpx

from auth import results_webhook_headers, schedule_webhook_headers

logger = logging.getLogger("thero.webhook")

WEBHOOK_URL = os.getenv("MEDNOVA_RESULTS_WEBHOOK_URL")

# Session-scheduled direction — own URL, separate from the results one
# above (Laravel endpoint not built yet as of this writing).
SCHEDULE_WEBHOOK_URL = os.getenv("MEDNOVA_SCHEDULE_WEBHOOK_URL")

# Nada: "roughly 1s, 4s, 15s, 60s, 5min" — see ASSUMPTION FLAGGED above
# for exactly how these five numbers map onto five attempts.
_RETRY_DELAYS_SECONDS = [1, 4, 15, 60, 300]
_ATTEMPT_TIMEOUT_SECONDS = 10.0


def _alert_delivery_failed(session_id: str, patient_id: str, last_error: str) -> None:
    """
    All 5 attempts exhausted. Per Nada this means a patient's clinical
    record failed to reach MedNova and someone needs to know NOW, not
    just eventually find it in a log — hence CRITICAL, not just an error
    log line.

    This is currently a LOUD LOCAL LOG ONLY. No Slack/email/PagerDuty (or
    any outbound alert channel) exists anywhere in this codebase, so
    there's nothing to wire this into yet — needs an ops decision on
    where alerts should land before this can do more than log. Treat this
    function as the single integration point to extend once that's
    decided, rather than scattering alert calls elsewhere.
    """
    logger.critical(
        "SESSION RESULTS WEBHOOK DELIVERY FAILED after %d attempts — "
        "session_id=%s patient_id=%s last_error=%s — clinical record NOT "
        "delivered to MedNova, manual follow-up required",
        len(_RETRY_DELAYS_SECONDS), session_id, patient_id, last_error,
    )


def build_session_result_payload(session) -> Dict[str, Any]:
    """
    Builds the webhook body from a SessionModel row.

    IMPORTANT: call this while the `with get_db() as db:` block that
    loaded `session` is still open. This reads attributes off a live
    SQLAlchemy ORM object — calling it after the session/context has
    closed will raise DetachedInstanceError on lazy-loaded columns. Build
    the payload dict first, THEN spawn the webhook task after (or even
    outside) the `with` block — see call sites in routers/sessions.py,
    routers/ws.py, telehealth.py.
    """
    return {
        "session_id":          session.id,
        "patient_id":          session.patient_id,
        "exercise_type":       session.exercise_type,
        "affected_side":       session.affected_side,
        "status":              session.status,
        "end_reason":          session.end_reason,
        "start_time":          session.start_time.isoformat() if session.start_time else None,
        "end_time":            session.end_time.isoformat() if session.end_time else None,
        "duration_seconds":    session.duration_seconds,
        "total_reps":          session.total_reps,
        "completed_reps":      session.completed_reps,
        "accuracy_percentage": session.accuracy_percentage,
        "average_rom":         session.average_rom,
        "incorrect_movements": session.incorrect_movements,
        "stability_score":     session.stability_score,
        "balance_score":       session.balance_score,
        "movement_smoothness": session.movement_smoothness,
        "fatigue_estimation":  session.fatigue_estimation,
        "recovery_score":      session.recovery_score,
    }


async def send_session_result_webhook(payload: Dict[str, Any]) -> None:
    """
    Runs the full 5-attempt retry/backoff schedule internally — worst
    case this takes ~(1+4+15+60+300)s ≈ 6.3 minutes to give up. Callers
    MUST NOT `await` this inline from a request/WS handler that a
    therapist or patient is waiting on; fire it via
    `asyncio.create_task(send_session_result_webhook(payload))` instead so
    the retry schedule runs in the background and never blocks the
    HTTP/WS response. See call sites in routers/sessions.py,
    routers/ws.py, telehealth.py for the pattern.

    No external task queue (Celery/RQ/etc.) exists in this codebase, so
    this keeps the retry loop in-process for now — the simplest thing
    that satisfies Nada's 5-attempt guarantee without adding new
    infrastructure. Known limitation: an in-flight retry is lost if the
    server restarts mid-backoff (e.g. mid-way through the 5-minute final
    wait). Not addressed here — revisit with a real task queue if that
    ever actually costs a webhook delivery in practice.
    """
    session_id = payload.get("session_id", "?")
    patient_id = payload.get("patient_id", "?")

    if not WEBHOOK_URL:
        logger.error(
            "MEDNOVA_RESULTS_WEBHOOK_URL is not set — cannot deliver "
            "session result webhook for session_id=%s", session_id,
        )
        return

    try:
        headers = results_webhook_headers()
    except RuntimeError as exc:
        # Config problem, not a transient delivery failure — still an
        # immediate ops-visible failure, so log it the same way a fully
        # exhausted retry loop would.
        _alert_delivery_failed(session_id, patient_id, str(exc))
        return

    last_error: Optional[str] = None

    for attempt, delay in enumerate(_RETRY_DELAYS_SECONDS, start=1):
        await asyncio.sleep(delay)
        try:
            async with httpx.AsyncClient(timeout=_ATTEMPT_TIMEOUT_SECONDS) as client:
                resp = await client.post(WEBHOOK_URL, json=payload, headers=headers)
            if resp.status_code == 200:
                return  # delivered — a repeat delivery also gets 200 (Laravel-side dedup)
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        logger.warning(
            "Session results webhook attempt %d/%d failed for "
            "session_id=%s: %s",
            attempt, len(_RETRY_DELAYS_SECONDS), session_id, last_error,
        )

    _alert_delivery_failed(session_id, patient_id, last_error or "unknown error")


def _alert_schedule_delivery_failed(room_id: str, external_id: str, last_error: str) -> None:
    """
    Same "loud local log only, no alert channel wired up yet" posture as
    _alert_delivery_failed() above — see that function's docstring. A
    failed session-scheduled delivery means the patient never gets routed
    to their join link, so this is just as CRITICAL as a lost clinical
    record.
    """
    logger.critical(
        "SESSION SCHEDULED WEBHOOK DELIVERY FAILED after %d attempts — "
        "room_id=%s external_id=%s last_error=%s — join link NOT "
        "delivered to MedNova, manual follow-up required",
        len(_RETRY_DELAYS_SECONDS), room_id, external_id, last_error,
    )


def build_session_scheduled_payload(room, patient, join_url: str) -> Dict[str, Any]:
    """
    Builds the webhook body from a TelehealthRoom row + its Patient, for
    BOTH modes (remote and self_training) — Laravel needs to route the
    patient to the link either way (confirmed 2026-08-25: fires for both
    modes, own secret separate from the results webhook).

    Same live-object caveat as build_session_result_payload() above: call
    this while the `with get_db() as db:` block that loaded `room` and
    `patient` is still open, then spawn the webhook task after (or outside)
    the `with` block. See telehealth.create_room() for the call site.

    external_id is patient.external_id — the Laravel customer_id this
    patient was synced in under (routers/integration.py sync_patient) —
    NOT thero's own patient_id, since Laravel has no use for thero's
    internal id and needs its own id to know which logged-in patient
    account to route the link to.
    """
    return {
        "room_id":       room.id,
        "external_id":   patient.external_id,
        "mode":          room.mode,
        "join_url":      join_url,
        "exercise_type": room.exercise_type,
        "scheduled_at":  room.scheduled_at.isoformat() if room.scheduled_at else None,
        "expires_at":    room.expires_at.isoformat() if room.expires_at else None,
    }


async def send_session_scheduled_webhook(payload: Dict[str, Any]) -> None:
    """
    Same 5-attempt retry/backoff schedule and in-process reasoning as
    send_session_result_webhook() above (see that function's docstring
    for the full rationale — worst case ~6.3 minutes, no external task
    queue, an in-flight retry is lost on server restart). Callers MUST
    fire this via `asyncio.create_task(...)`, never awaited inline — see
    telehealth.create_room() for the call site.

    If patient.external_id is empty (a patient created directly in thero,
    never synced from MedNova — e.g. local testing) there is no MedNova
    account to route to, so this is skipped with a warning rather than
    sent to Laravel with a blank external_id, which Laravel couldn't
    match to any patient anyway.
    """
    room_id = payload.get("room_id", "?")
    external_id = payload.get("external_id")

    if not external_id:
        logger.warning(
            "Skipping session-scheduled webhook for room_id=%s — patient "
            "has no external_id (not synced from MedNova)", room_id,
        )
        return

    if not SCHEDULE_WEBHOOK_URL:
        logger.error(
            "MEDNOVA_SCHEDULE_WEBHOOK_URL is not set — cannot deliver "
            "session scheduled webhook for room_id=%s", room_id,
        )
        return

    try:
        headers = schedule_webhook_headers()
    except RuntimeError as exc:
        _alert_schedule_delivery_failed(room_id, external_id, str(exc))
        return

    last_error: Optional[str] = None

    for attempt, delay in enumerate(_RETRY_DELAYS_SECONDS, start=1):
        await asyncio.sleep(delay)
        try:
            async with httpx.AsyncClient(timeout=_ATTEMPT_TIMEOUT_SECONDS) as client:
                resp = await client.post(SCHEDULE_WEBHOOK_URL, json=payload, headers=headers)
            if resp.status_code == 200:
                return  # delivered — a repeat delivery also gets 200 (Laravel-side dedup, once they build it)
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        logger.warning(
            "Session scheduled webhook attempt %d/%d failed for "
            "room_id=%s: %s",
            attempt, len(_RETRY_DELAYS_SECONDS), room_id, last_error,
        )

    _alert_schedule_delivery_failed(room_id, external_id, last_error or "unknown error")