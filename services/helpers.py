import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

from models import SessionModel, JointAngle, ExerciseResult, History, VALID_END_REASONS
from services.webhook import build_session_result_payload


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


# ── Shared session-save core (routers/sessions.py + telehealth.py) ─────
# Both POST /api/sessions (JWT-authenticated, routers/sessions.py) and
# POST /api/telehealth/bridge-save-session/{room_id} (bridge-token
# authenticated, telehealth.py) accept the exact same payload shape
# (exercise_type, affected_side, exercise_results[], joint_angles[],
# summary metrics, ...) and need to do the exact same INSERT-or-UPDATE
# work against SessionModel. The two endpoints differ only in HOW the
# caller is authenticated and how `pid` (the patient this session
# belongs to) is established — that part stays in each router/endpoint,
# never here. This function assumes `pid` has ALREADY been validated by
# the caller and just does the save.

def _resolve_end_reason(payload: Dict[str, Any]) -> str:
    """
    Item 24: finalize `end_reason` against the 7-value VALID_END_REASONS
    set (models.py) — completed / stopped_by_patient / pain /
    technical_error / disconnected / timeout / stopped_by_therapist.

    The client (patient/therapist page) sends whichever button the person
    actually pressed to end the session. We validate against
    VALID_END_REASONS instead of trusting an arbitrary string, since this
    is what reports/analytics will eventually group sessions by. Anything
    missing or not in the allowed set falls back to "completed" — the
    pre-existing behaviour for every client that finishes a session
    normally and doesn't send this field at all (keeps this change
    backward compatible with any caller that hasn't been updated yet).

    NOTE: the old value "stopped" (pre-item-24 clarification) is no
    longer valid and will fall back to "completed" here rather than
    "stopped_by_patient" — we don't guess which of patient/therapist it
    was for a legacy/unmigrated caller. Frontend callers still sending
    "stopped" need updating to send "stopped_by_patient" explicitly
    (pending patient.html/session.html wiring — not done yet).
    """
    reason = payload.get("end_reason")
    if reason in VALID_END_REASONS:
        return reason
    return "completed"


def _resolve_rep_counts(payload: Dict[str, Any], exercise_results_payload: list) -> Tuple[int, int]:
    """
    🟡 Semantics NOT yet confirmed with Nada — see inline notes. This is a
    safety-net derivation, not a verified business rule. Flagging clearly
    so it isn't mistaken for a confirmed spec.

    Previously total_reps/completed_reps were trusted verbatim from the
    top-level payload with zero relationship check — a buggy/malicious
    client could send completed_reps > total_reps, or numbers with no
    connection to what actually happened in the session.

    exercise_results[] (one row per rep, each with `is_completed`) is
    already saved to the ExerciseResult table below — so when it's
    present, it's real per-rep ground truth and is a stronger signal
    than the two raw top-level ints:

        total_reps     = number of rep records the client reported at all
        completed_reps = number of those rows where is_completed == True

    ASSUMPTION (needs confirmation): "completed" = met the rep's quality/
    ROM bar, not just "attempted". If exercise_results[] isn't guaranteed
    to be populated for every session type (e.g. self_training / telehealth
    mode — unconfirmed), we fall back to the raw payload values, clamped
    so completed_reps can never exceed total_reps.
    """
    if exercise_results_payload:
        total_reps = len(exercise_results_payload)
        completed_reps = sum(
            1 for er in exercise_results_payload if er.get("is_completed")
        )
        return total_reps, completed_reps

    # Fallback: no per-rep breakdown sent — trust the top-level ints but
    # clamp so completed can never exceed total (the "safety clamp" that
    # existed in spirit on the frontend but never enforced server-side).
    total_reps = payload.get("total_reps", 0) or 0
    completed_reps = payload.get("completed_reps", 0) or 0
    completed_reps = max(0, min(completed_reps, total_reps))
    return total_reps, completed_reps


def save_session_core(
    db,
    pid: str,
    payload: Dict[str, Any],
    consultation_id: Optional[str] = None,
    room_id: Optional[str] = None,
) -> Tuple[SessionModel, Dict[str, Any]]:
    """
    Shared INSERT-or-UPDATE body of a full session-save payload — extracted
    from what used to be routers/sessions.py's save_session() so that
    telehealth.py's bridge-token-authenticated
    POST /api/telehealth/bridge-save-session/{room_id} can reuse it instead
    of re-implementing the same exercise_results/joint_angles/summary-field
    handling a second time.

    Caller's responsibility (NOT done here):
      - authenticating the request and resolving `pid` (JWT + assert_owns_patient
        for routers/sessions.py; room.token match + room.patient_id for the
        bridge endpoint — the bridge caller's body is never trusted for
        patient identity, only the room row is)
      - db.commit() (so the caller can add its own rows — e.g. the bridge
        endpoint's room bookkeeping — in the same transaction if it ever
        needs to)
      - firing send_session_result_webhook(webhook_payload) as a background
        task AFTER the `with get_db()` block exits (must not delay the
        HTTP response — see build_session_result_payload() docstring)

    `consultation_id` / `room_id` are None for the plain JWT path (no
    telehealth room involved) and are passed through by the bridge
    endpoint so the webhook payload can carry them, same as
    save_self_training_session already does with room_id.

    Returns (sess, webhook_payload) — webhook_payload is already a plain
    dict (see build_session_result_payload docstring for why this must be
    read before commit/context exit).
    """
    def _dt(key):
        v = payload.get(key)
        if not v:
            return None
        # JS toISOString() emits a trailing 'Z', which Python 3.10's
        # fromisoformat() can't parse directly (only 3.11+ supports it).
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(v)

    exercise_results_payload = payload.get("exercise_results", [])
    total_reps, completed_reps = _resolve_rep_counts(payload, exercise_results_payload)
    end_reason = _resolve_end_reason(payload)

    # If this finishing POST references a session_id that POST
    # /api/sessions/start already created for this same patient and
    # that's still "in_progress", UPDATE that row instead of inserting
    # a second one — previously this endpoint always did a fresh
    # INSERT, so any client that adopted the start/finish flow would
    # end up with a duplicate row (an empty in_progress one + the real
    # finished one). Legacy callers that never call /start (no
    # session_id in the payload, or the id doesn't match an
    # in_progress row of theirs) fall through to the old insert
    # behaviour unchanged.
    # Matches "in_progress" (the normal case) AND "abandoned" — the
    # latter covers the race where the patient's /ws/pose socket
    # disconnects and _abandon_if_unfinished() (routers/ws.py) already
    # flipped this row to "abandoned" BEFORE this finishing POST
    # arrives (e.g. tab closes right as the therapist's page submits
    # the summary). Without "abandoned" here, that race caused this
    # query to miss the row entirely and fall through to the INSERT
    # branch below — creating a duplicate row for the same session_id
    # while the original stayed stuck at status="abandoned" with
    # whatever partial data the live pipeline had captured. The
    # finishing POST is the explicit, authoritative "this session is
    # over" signal from the therapist/patient client, so it should
    # always win and update the one true row regardless of which of
    # the two arrived first — never create a second row for a
    # session_id that already exists.
    existing_id = payload.get("session_id")
    sess = None
    if existing_id:
        sess = (
            db.query(SessionModel)
            .filter(SessionModel.id == existing_id,
                    SessionModel.patient_id == pid,
                    SessionModel.status.in_(("in_progress", "abandoned")))
            .first()
        )

    if sess is not None:
        sess.exercise_type       = payload.get("exercise_type", sess.exercise_type)
        sess.affected_side       = payload.get("affected_side", sess.affected_side)
        sess.end_time            = _dt("end_time") or datetime.datetime.now()
        sess.duration_seconds    = payload.get("duration_seconds", 0)
        sess.total_reps          = total_reps
        sess.completed_reps      = completed_reps
        sess.accuracy_percentage = payload.get("accuracy_percentage", 0.0)
        sess.average_rom         = payload.get("average_rom", 0.0)
        sess.incorrect_movements = payload.get("incorrect_movements", 0)
        sess.stability_score     = payload.get("stability_score", 0.0)
        sess.balance_score       = payload.get("balance_score", 0.0)
        sess.movement_smoothness = payload.get("movement_smoothness", 0.0)
        sess.fatigue_estimation  = payload.get("fatigue_estimation", 0.0)
        sess.recovery_score      = payload.get("recovery_score", 0.0)
        sess.session_data        = payload.get("session_data", {})
        sess.status              = "completed"
        sess.end_reason          = end_reason
    else:
        sess = SessionModel(
            patient_id          = pid,
            consultation_id     = consultation_id,
            exercise_type       = payload.get("exercise_type", "General Exercise"),
            affected_side       = payload.get("affected_side", "both"),
            start_time          = _dt("start_time") or datetime.datetime.now(),
            end_time            = _dt("end_time"),
            duration_seconds    = payload.get("duration_seconds", 0),
            total_reps          = total_reps,
            completed_reps      = completed_reps,
            accuracy_percentage = payload.get("accuracy_percentage", 0.0),
            average_rom         = payload.get("average_rom", 0.0),
            incorrect_movements = payload.get("incorrect_movements", 0),
            stability_score     = payload.get("stability_score", 0.0),
            balance_score       = payload.get("balance_score", 0.0),
            movement_smoothness = payload.get("movement_smoothness", 0.0),
            fatigue_estimation  = payload.get("fatigue_estimation", 0.0),
            recovery_score      = payload.get("recovery_score", 0.0),
            session_data        = payload.get("session_data", {}),
            status              = "completed",
            end_reason          = end_reason,
        )
        db.add(sess)
    db.flush()

    for ja in payload.get("joint_angles", []):
        db.add(JointAngle(
            session_id   = sess.id,
            joint_name   = ja.get("joint_name", "Unknown"),
            angle_value  = ja.get("angle_value", 0.0),
            target_angle = ja.get("target_angle"),
            deviation    = ja.get("deviation"),
            is_correct   = ja.get("is_correct", True),
        ))

    for er in exercise_results_payload:
        db.add(ExerciseResult(
            session_id         = sess.id,
            exercise_name      = er.get("exercise_name", "Unknown"),
            repetition_number  = er.get("repetition_number", 0),
            accuracy           = er.get("accuracy", 0.0),
            rom_achieved       = er.get("rom_achieved", 0.0),
            speed              = er.get("speed", 0.0),
            hold_duration      = er.get("hold_duration", 0.0),
            compensation_score = er.get("compensation_score", 0.0),
            is_completed       = er.get("is_completed", False),
            feedback           = er.get("feedback", ""),
        ))

    db.add(History(
        patient_id = pid,
        action     = "Session Saved",
        details    = f"Session {sess.id} — {sess.completed_reps}/{sess.total_reps} reps, ended: {sess.end_reason}",
    ))

    # Item 27: webhook payload must be built BEFORE commit/context
    # exit — see build_session_result_payload() docstring for why
    # (DetachedInstanceError otherwise).
    webhook_payload = build_session_result_payload(sess, room_id=room_id)

    return sess, webhook_payload


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