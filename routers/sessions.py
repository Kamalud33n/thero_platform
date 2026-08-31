import asyncio
import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from auth import (
    CurrentTherapist,
    get_current_therapist,
    issue_patient_session_token,
    issue_doctor_session_token,
)
from database import get_db
from models import SessionModel, JointAngle, ExerciseResult, History, VALID_END_REASONS
from repositories.patient_repo import assert_owns_patient
from services.webhook import build_session_result_payload, send_session_result_webhook

router = APIRouter()


@router.post("/api/sessions/start")
async def start_session(
    payload: Dict[str, Any],
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    """
    Creates the SessionModel row UP FRONT, before any patient ever
    connects to /ws/pose — this is what "session creation without a
    patient DB row" was missing: previously /ws/pose accepted any
    client-chosen session_id string with zero requirement that a
    Patient or SessionModel row existed for it at all.

    assert_owns_patient() below both (a) confirms the patient row is
    real and (b) confirms it belongs to THIS therapist, before a
    session_id or patient token is ever minted for it.

    Returns a single-use, short-lived patient session token
    (see auth.issue_patient_session_token) that the therapist's page
    hands to the patient to open /ws/pose with.
    """
    with get_db() as db:
        pid = payload.get("patient_id")
        assert_owns_patient(db, pid, therapist)

        sess = SessionModel(
            patient_id    = pid,
            exercise_type = payload.get("exercise_type", "General Exercise"),
            affected_side = payload.get("affected_side", "both"),
            start_time    = datetime.datetime.now(),
            status        = "in_progress",
        )
        db.add(sess)
        db.commit()
        db.refresh(sess)

        token = issue_patient_session_token(session_id=sess.id, patient_id=pid)

        return JSONResponse({
            "success":    True,
            "session_id": sess.id,
            "token":      token,
            "expires_in": 900,
        })


@router.post("/api/sessions/{session_id}/watch-token")
async def get_doctor_watch_token(
    session_id: str,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    """
    Mints a short-lived, signed doctor session token (auth.issue_doctor_session_token)
    for THIS therapist to watch an in-progress session over /ws/pose?role=doctor.

    Closes the gap noted in services/camera_ws.py's RoomManager docstring —
    doctor role was previously trusted from the raw query string with no
    signed claim behind it. Re-confirms ownership here (via the session's
    patient_id) rather than trusting that the therapist who started the
    session is necessarily the one asking to watch it now — same
    assert_owns_patient() check every other therapist-scoped route uses.
    """
    with get_db() as db:
        sess = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if sess is None:
            return JSONResponse({"success": False, "message": "Session not found"}, status_code=404)

        # 404, not 403, for the same reason get_owned_patient() does it:
        # existence and ownership must be indistinguishable from the caller.
        assert_owns_patient(db, sess.patient_id, therapist)

        token = issue_doctor_session_token(session_id=session_id, therapist=therapist)
        return JSONResponse({
            "success":    True,
            "session_id": session_id,
            "token":      token,
            "expires_in": 3600,
        })


@router.get("/api/sessions/{patient_id}")
async def get_sessions(
    patient_id: str,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    with get_db() as db:
        assert_owns_patient(db, patient_id, therapist)
        sessions = (
            db.query(SessionModel)
            .filter(SessionModel.patient_id == patient_id)
            # Item 27 (Nada, 2026-08-23): a session that opened and
            # captured ZERO frames before ending — WS dropped almost
            # instantly, nothing was ever tracked — still gets reported
            # to the results webhook as abandoned/0 reps (see
            # routers/ws.py._abandon_if_unfinished + services/webhook.py),
            # but must NOT show up in the patient's session history here.
            # Excluded by shape (status="abandoned" AND total_reps==0)
            # rather than a dedicated column, since that combination
            # doesn't occur any other way — a completed/stopped session
            # always has status != "abandoned", and a real abandoned
            # session that captured at least one rep still belongs in
            # history per Nada's spec.
            .filter(~((SessionModel.status == "abandoned") & (SessionModel.total_reps == 0)))
            .order_by(SessionModel.start_time.desc())
            .all()
        )
        out = []
        for s in sessions:
            out.append({
                "session_id":          s.id,
                "patient_id":          s.patient_id,
                "exercise_type":       s.exercise_type,
                "affected_side":       s.affected_side,
                "start_time":          s.start_time.isoformat(),
                "end_time":            s.end_time.isoformat() if s.end_time else None,
                "status":              s.status,
                "end_reason":          s.end_reason,  # item 24 — None treated as "completed" by convention, see models.py
                "duration_seconds":    s.duration_seconds,
                "total_reps":          s.total_reps,
                "completed_reps":      s.completed_reps,
                "accuracy_percentage": s.accuracy_percentage,
                "average_rom":         s.average_rom,
                "incorrect_movements": s.incorrect_movements,
                "stability_score":     s.stability_score,
                "balance_score":       s.balance_score,
                "movement_smoothness": s.movement_smoothness,
                "fatigue_estimation":  s.fatigue_estimation,
                "recovery_score":      s.recovery_score,
                "joint_angles": [
                    {"joint_name": ja.joint_name, "angle": ja.angle_value,
                     "target": ja.target_angle, "is_correct": ja.is_correct}
                    for ja in s.joint_angles[:20]
                ],
            })
        return JSONResponse(out)


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


def _resolve_rep_counts(payload: Dict[str, Any], exercise_results_payload: list) -> tuple[int, int]:
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


@router.post("/api/sessions")
async def save_session(
    payload: Dict[str, Any],
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    with get_db() as db:
        pid = payload.get("patient_id")
        assert_owns_patient(db, pid, therapist)

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
        webhook_payload = build_session_result_payload(sess)
        db.commit()

    # Fired OUTSIDE the `with get_db()` block, on purpose: this session
    # finished successfully from the therapist's point of view the
    # moment db.commit() above returns — the webhook's ~6-minute worst
    # case retry schedule (see services/webhook.py) must never delay
    # this response back to the client.
    asyncio.create_task(send_session_result_webhook(webhook_payload))

    return JSONResponse({"success": True, "message": "Session saved", "session_id": sess.id})