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
from models import SessionModel
from repositories.patient_repo import assert_owns_patient
from services.helpers import save_session_core
from services.webhook import send_session_result_webhook

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


@router.post("/api/sessions")
async def save_session(
    payload: Dict[str, Any],
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    """
    JWT-authenticated session save. The actual INSERT-or-UPDATE work is
    shared with the bridge-token-authenticated
    POST /api/telehealth/bridge-save-session/{room_id} (telehealth.py) via
    services.helpers.save_session_core — this endpoint's own job is just
    (a) authenticating the therapist and (b) proving they own `patient_id`
    before any of that shared logic runs.
    """
    with get_db() as db:
        pid = payload.get("patient_id")
        assert_owns_patient(db, pid, therapist)

        sess, webhook_payload = save_session_core(db, pid, payload)
        db.commit()

    # Fired OUTSIDE the `with get_db()` block, on purpose: this session
    # finished successfully from the therapist's point of view the
    # moment db.commit() above returns — the webhook's ~6-minute worst
    # case retry schedule (see services/webhook.py) must never delay
    # this response back to the client.
    asyncio.create_task(send_session_result_webhook(webhook_payload))

    return JSONResponse({"success": True, "message": "Session saved", "session_id": sess.id})