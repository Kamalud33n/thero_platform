import logging
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger("thero.ws")


# ── WS message versioning (item 26) ─────────────────────────────────────
# Every server -> client WebSocket message (/ws/pose, /ws/signal/{room_id},
# /ws/self-training/{room_id}) now carries a top-level "v" field so the
# frontend (and any future consumer) can tell which message shape it's
# looking at and branch/upgrade safely instead of guessing from field
# presence. Bump WS_MESSAGE_VERSION whenever an existing message TYPE's
# shape changes in a backward-incompatible way (field renamed/removed,
# meaning changed) — adding a new optional field or a brand new message
# type does not require a bump.
#
# Only the message *shape/contract* is versioned this way, not each
# individual message type separately — simpler for both client and
# server to reason about with just one message family in play right now.
# If /ws/pose and /ws/signal ever diverge in shape, split this into
# per-channel constants then.
WS_MESSAGE_VERSION = 1


def stamp(message: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the current WS protocol version to an outgoing server->client
    message. Call this on every dict passed to `.send_json()` /
    `ws_mgr.send()` / `ws_mgr.broadcast()` right before it goes out —
    mutates and returns the same dict for convenient inline use:
        await websocket.send_json(stamp({"type": "connected", ...}))
    """
    message["v"] = WS_MESSAGE_VERSION
    return message


def client_message_version(msg: Dict[str, Any]) -> int:
    """Best-effort read of the client's own declared version on an inbound
    message. Older/unversioned clients won't send "v" at all — treated as
    version 1 (today's only shape) rather than rejected, so this stays
    backward compatible with any frontend that hasn't been updated yet.
    """
    try:
        return int(msg.get("v", 1))
    except (TypeError, ValueError):
        return 1


# ── Item 26 enforcement (per Nada, 2026-08-23) ──────────────────────────
# "Format: a plain integer v: 1 is enough. Semantic versioning solves a
# problem we don't have — there's one producer and one consumer here. On
# mismatch: neither strict rejection nor silent ignoring. If the client
# version is older than the engine supports, accept it and log a warning.
# If it's newer — meaning the client is ahead of a deployment — close
# with a specific code and a clear message so our frontend can tell the
# patient to refresh. The reason not to be strict: a patient mid-session
# shouldn't have their session killed because we deployed. The reason not
# to ignore silently: we'd never find out we have stale clients in the
# wild."
#
# Custom WS close code for the "client is newer than us" case. Distinct
# from the existing 4000/4001/4002/4003/4009 close codes used elsewhere
# in routers/ws.py and telehealth.py — pick a code that isn't already in
# use before wiring this into a fourth channel.
WS_VERSION_MISMATCH_CLOSE_CODE = 4010

# Short machine-readable reason string, separate from the human-readable
# `message` field — lets the frontend match on `reason` reliably even if
# the wording of `message` changes later.
WS_VERSION_MISMATCH_REASON = "protocol_version_ahead"


def check_client_protocol_version(msg: Dict[str, Any]) -> Optional[int]:
    """
    Call this once per inbound client message, right after
    `receive_json()`, in every server->client WS loop (/ws/pose,
    /ws/signal/{room_id}, /ws/self-training/{room_id}).

    Returns:
        None                          — versions match, or client is OLDER
                                         than the server (accepted; a
                                         warning is logged here, caller
                                         doesn't need to do anything else).
        WS_VERSION_MISMATCH_CLOSE_CODE — client is NEWER than the server.
                                         Caller must send
                                         build_version_mismatch_message()
                                         to the client, then close the
                                         socket with this code, then stop
                                         processing this connection.

    Deliberately does NOT close the socket itself — different call sites
    close differently (some via ws_mgr.send + websocket.close(), some via
    a bare websocket.close()), so this just reports the verdict and lets
    each call site handle it in its own existing style.
    """
    client_v = client_message_version(msg)
    if client_v < WS_MESSAGE_VERSION:
        logger.warning(
            "Stale WS client — client protocol version %d < server %d "
            "(msg type=%r). Accepting per item 26 policy: never kill a "
            "patient's mid-session connection over a version gap.",
            client_v, WS_MESSAGE_VERSION, msg.get("type"),
        )
        return None
    if client_v > WS_MESSAGE_VERSION:
        return WS_VERSION_MISMATCH_CLOSE_CODE
    return None


def build_version_mismatch_message(client_v: int) -> Dict[str, Any]:
    """
    The "clear message" Nada asked for, sent to the client right before
    closing with WS_VERSION_MISMATCH_CLOSE_CODE — gives the frontend both
    a machine-readable `reason` to branch on and a human-readable
    `message` to show the patient directly if it just displays it as-is.
    Already carries `v` via stamp() so it matches every other
    server->client message shape.
    """
    return stamp({
        "type":            "protocol_version_mismatch",
        "reason":          WS_VERSION_MISMATCH_REASON,
        "message":         "This session needs a newer app version than "
                            "this server currently supports. Please "
                            "refresh the page and try again.",
        "server_version":  WS_MESSAGE_VERSION,
        "client_version":  client_v,
    })


def derive_session_summary_stats(payload: Dict[str, Any]) -> Tuple[float, float, int]:
    """
    Derive (accuracy_percentage, average_rom, incorrect_movements) from the
    itemized `exercise_results` / `joint_angles` lists already present in a
    session-save payload, instead of trusting the separate top-level
    summary fields the client also sends.

    Why: POST /api/sessions and the self-training save endpoint both
    accept a top-level `accuracy_percentage` / `average_rom` /
    `incorrect_movements` AND a granular `exercise_results` /
    `joint_angles` breakdown in the same payload — but nothing checked
    that the two agreed. A client could send `exercise_results` showing
    genuinely poor form (low per-rep `accuracy`, `is_correct: false`
    angles) while the top-level summary still claimed a high score, and
    the server would persist whichever number the client felt like
    sending. Recomputing the summary from the itemized data server-side
    closes that gap — the per-rep/per-frame breakdown is harder to fake
    consistently than a single top-line number.

    Falls back to the client's top-level summary values when the
    itemized lists are empty (e.g. very short sessions, or a save path
    that doesn't send per-rep detail) so this never turns a legitimate
    save into a hard failure.
    """
    exercise_results: List[Dict[str, Any]] = payload.get("exercise_results") or []
    joint_angles: List[Dict[str, Any]] = payload.get("joint_angles") or []

    if exercise_results:
        accuracies = [er.get("accuracy", 0.0) or 0.0 for er in exercise_results]
        roms       = [er.get("rom_achieved", 0.0) or 0.0 for er in exercise_results]
        accuracy_percentage = sum(accuracies) / len(accuracies)
        average_rom          = sum(roms) / len(roms)
    else:
        accuracy_percentage = payload.get("accuracy_percentage", 0.0) or 0.0
        average_rom          = payload.get("average_rom", 0.0) or 0.0

    if joint_angles:
        incorrect_movements = sum(1 for ja in joint_angles if not ja.get("is_correct", True))
    else:
        incorrect_movements = payload.get("incorrect_movements", 0) or 0

    return accuracy_percentage, average_rom, incorrect_movements


# ── Analytics / report scoring (used by routers/analytics.py and
# services/report_builder.py) ───────────────────────────────────────────

def calculate_recovery_score(sessions: List[Any]) -> float:
    """Single 0-100 composite score for a bucket of sessions (a day's
    sessions in analytics.py, or a patient's full history in a report).

    Weighted blend of the signals that actually indicate recovery
    progress — accuracy and ROM matter most, stability/balance/smoothness
    are secondary quality signals, and a penalty is applied for
    incorrect-movement rate so a fast-but-sloppy session doesn't outscore
    a slower, correct one. Returns 0.0 for an empty bucket rather than
    raising, since callers (e.g. analytics.py's per-day buckets) routinely
    pass empty lists for days with no sessions.
    """
    n = len(sessions)
    if n == 0:
        return 0.0

    def _avg(attr: str) -> float:
        return sum(getattr(s, attr, 0.0) or 0.0 for s in sessions) / n

    accuracy   = _avg("accuracy_percentage")
    rom        = _avg("average_rom")
    stability  = _avg("stability_score")
    balance    = _avg("balance_score")
    smoothness = _avg("movement_smoothness")

    total_reps = sum((getattr(s, "completed_reps", 0) or 0) for s in sessions)
    incorrect  = sum((getattr(s, "incorrect_movements", 0) or 0) for s in sessions)
    error_rate = (incorrect / total_reps) if total_reps else 0.0
    penalty    = min(error_rate * 100, 25.0)  # cap so a rough session can't go negative

    score = (
        accuracy   * 0.35 +
        rom        * 0.25 +
        stability  * 0.15 +
        balance    * 0.15 +
        smoothness * 0.10
    ) - penalty

    return round(max(0.0, min(100.0, score)), 1)


def calculate_improvement(sessions: List[Any]) -> float:
    """% change in accuracy between the first half and second half of a
    chronologically-ordered session list — the "is this patient trending
    better" number shown on PDF reports. Positive = improving.
    Needs at least 2 sessions to mean anything; returns 0.0 otherwise.
    """
    n = len(sessions)
    if n < 2:
        return 0.0

    ordered = sorted(sessions, key=lambda s: getattr(s, "start_time", None) or 0)
    mid = n // 2
    first_half, second_half = ordered[:mid] or ordered[:1], ordered[mid:]

    def _avg_acc(bucket) -> float:
        return sum((getattr(s, "accuracy_percentage", 0.0) or 0.0) for s in bucket) / len(bucket)

    before, after = _avg_acc(first_half), _avg_acc(second_half)
    if before == 0:
        return 100.0 if after > 0 else 0.0
    return round(((after - before) / before) * 100, 1)


def session_summary(session: Any) -> Dict[str, Any]:
    """Compact dict form of a SessionModel row for list views (analytics
    "today/yesterday" session lists) — avoids leaking the full ORM object
    or every column into the JSON response."""
    return {
        "id":                   session.id,
        "patient_id":           session.patient_id,
        "exercise_type":        session.exercise_type,
        "start_time":           session.start_time.isoformat() if session.start_time else None,
        "duration_seconds":     session.duration_seconds,
        "completed_reps":       session.completed_reps,
        "total_reps":           session.total_reps,
        "accuracy_percentage":  session.accuracy_percentage,
        "average_rom":          session.average_rom,
        "stability_score":      session.stability_score,
        "incorrect_movements":  session.incorrect_movements,
        "status":               session.status,
        "end_reason":           session.end_reason,
    }