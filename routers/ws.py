import asyncio
import base64
import datetime
import time as _time

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from database import get_db
from models import SessionModel
from services.camera_ws import ws_mgr
from services.webhook import build_session_result_payload, send_session_result_webhook
from services.timeutils import utcnow, utcnow_iso
from services.helpers import (
    client_message_version,
    check_client_protocol_version,
    build_version_mismatch_message,
)
from auth import (
    check_ws_origin,
    decode_patient_session_token,
    consume_patient_session_jti,
    decode_doctor_session_token,
)

router = APIRouter()

# Client → server message shapes (all JSON, one per WS text frame):
#   {"type": "frame", "data": "<base64 JPEG>"}   — a webcam frame to process
#   {"type": "set_exercise", "exercise_type": "...", "target_rom": 90, "affected_side": "left"}
#   {"type": "reset_session"}
#   (patient role only — see below)
#
# Server → client:
#   {"type": "connected", "connection_id": "...", "role": "patient"|"doctor"}  on connect
#   {"type": "pose_data", "frame": "<base64 JPEG>", "pose_data": {...}, "ts": "..."}
#     — broadcast to every connection in the room (patient's own client +
#       any doctor watching), not just echoed back to the sender.
#   {"type": "backpressure", "recommended_fps": <float>, "reason": "server_congested"}
#     — sent ONLY to the patient/producer connection, at most once every
#       few seconds, when this connection can't sustain TARGET_FPS or the
#       browser is sending faster than the server can drain. Real
#       congestion feedback, not just server-side frame-skip — a client
#       that wants to honor it should reduce its own capture/send rate to
#       recommended_fps. Optional to act on; the server already protects
#       itself either way via the adaptive skip ratio.
#
# Connect as:
#   Patient: /ws/pose?session_id=<id>&role=patient&token=<patient session token>
#     session_id + token together are now REQUIRED and verified — see
#     _authorize_patient_connection() below. token is minted by
#     POST /api/sessions/start (routers/sessions.py) and is single-use
#     (jti replay protection, see auth.consume_patient_session_jti).
#   Doctor: /ws/pose?session_id=<id>&role=doctor&token=<doctor watch token>
#     session_id + token together are now REQUIRED and verified — see
#     _authorize_doctor_connection() below. token is minted by
#     POST /api/sessions/{session_id}/watch-token (routers/sessions.py),
#     which re-checks the calling therapist owns the session's patient
#     before issuing anything. Unlike the patient token, this one is NOT
#     single-use (a doctor may legitimately reconnect within the TTL) —
#     see auth.py's "Doctor session-scoped token" section.


async def _authorize_patient_connection(websocket: WebSocket, session_id: str) -> bool:
    """
    Verifies the patient's session token (signature, exp, iss, aud, iat —
    see auth.decode_patient_session_token) AND that it hasn't already been
    redeemed (jti replay protection — auth.consume_patient_session_jti),
    AND that session_id actually corresponds to a real, still-open
    SessionModel row for the patient named in the token.

    Returns True and leaves the connection open for the caller to accept
    on success. On any failure, closes the socket with a specific code and
    returns False — caller must not proceed.
    """
    token = websocket.query_params.get("token")
    try:
        claims = decode_patient_session_token(token)
    except Exception:
        # decode_patient_session_token raises HTTPException, which has no
        # meaning on a WebSocket — translate to a close code instead.
        await websocket.close(code=4001)  # invalid/expired/malformed token
        return False

    if claims.session_id != session_id:
        # Token was issued for a different session_id than the one this
        # connection is trying to join — reject rather than silently
        # trusting whichever one the query string happens to say.
        await websocket.close(code=4001)
        return False

    with get_db() as db:
        try:
            consume_patient_session_jti(db, claims)
        except Exception:
            # Already used — replay attempt.
            await websocket.close(code=4008)  # custom code: token already used
            return False

        sess = (
            db.query(SessionModel)
            .filter(SessionModel.id == session_id,
                    SessionModel.patient_id == claims.patient_id,
                    SessionModel.status == "in_progress")
            .first()
        )
        if sess is None:
            # Either the row doesn't exist, belongs to a different patient
            # than the token claims, or has already been finished/abandoned
            # — none of those should let a new connection start scoring
            # against it.
            await websocket.close(code=4001)
            return False

        # Commit the jti "used" row now, before accept() — closes the race
        # where two connections present the same (captured) token
        # concurrently and both pass the pre-commit SELECT above.
        db.commit()

    return True


async def _authorize_doctor_connection(websocket: WebSocket, session_id: str) -> bool:
    """
    Verifies the doctor's watch token (signature, exp, iss, aud, iat, role
    claim — see auth.decode_doctor_session_token) AND that it was issued
    for this exact session_id. No jti/replay check (doctor tokens are
    reusable within their TTL — see auth.py note).

    Returns True and leaves the connection open for the caller to accept
    on success. On any failure, closes the socket with a specific code and
    returns False — caller must not proceed.
    """
    token = websocket.query_params.get("token")
    try:
        claims = decode_doctor_session_token(token)
    except Exception:
        await websocket.close(code=4001)  # invalid/expired/malformed token
        return False

    if claims.session_id != session_id:
        # Token was minted for a different session_id than this connection
        # is trying to join — reject rather than trusting the query string.
        await websocket.close(code=4001)
        return False

    return True


async def _abandon_if_unfinished(session_id: str, camera) -> None:
    """
    Mid-session disconnect handling for the PATIENT connection: if the
    session_id's SessionModel row is still "in_progress" when the socket
    drops (crash, network loss, tab closed — anything that isn't the
    normal finishing POST /api/sessions arriving first), auto-save
    whatever the live pipeline captured instead of silently losing the
    session data forever.

    Best-effort by nature: the live pipeline only ever tracked ONE rep
    counter (see services/metrics.py), not a separate attempted-vs-completed
    breakdown, so both total_reps and completed_reps are set to that same
    live count here. accuracy_percentage is the live rolling accuracy
    (services/metrics.SessionMetrics.get_accuracy) rather than the richer
    per-rep client computation the finishing POST would normally send.
    Marked status="abandoned" (not "completed") specifically so reports/
    analytics can tell a genuinely finished session apart from a dropped
    one if that distinction ever matters downstream.
    """
    if camera is None:
        return
    with get_db() as db:
        sess = (
            db.query(SessionModel)
            .filter(SessionModel.id == session_id, SessionModel.status == "in_progress")
            .first()
        )
        if sess is None:
            return  # already finished normally, or never existed (doctor-only room, etc.)

        m = camera.metrics
        reps = m.get_rep_count()
        sess.end_time            = utcnow()
        if sess.start_time:
            sess.duration_seconds = int((sess.end_time - sess.start_time).total_seconds())
        sess.total_reps          = reps
        sess.completed_reps      = reps
        sess.accuracy_percentage = m.get_accuracy()
        sess.stability_score     = m.get_stability()
        sess.balance_score       = m.get_balance()
        sess.movement_smoothness = m.get_smoothness()
        sess.fatigue_estimation  = m.get_current_fatigue()
        sess.status              = "abandoned"
        # Item 24 (updated per Nada's case-monitoring spec, 2026-08-23):
        # "disconnected" is now its own explicit VALID_END_REASONS value
        # for exactly this situation — WS dropped and never reconnected,
        # with no client-reported reason behind it. Previously left NULL
        # here since none of the old 3 values ("completed"/"stopped"/
        # "pain") fit a dropped socket; "disconnected" now does.
        sess.end_reason           = "disconnected"
        # Item 27: webhook payload must be built BEFORE commit/context
        # exit — see build_session_result_payload() docstring for why.
        # Fires for EVERY disconnect per Nada's "in all cases, without
        # exception" policy — including reps==0 (session opened, captured
        # nothing, dropped immediately). That zero-frame case is still
        # sent to the webhook as abandoned/0 reps exactly as spec'd; it's
        # only excluded from the patient's *history listing* on our side
        # (see routers/sessions.py.get_sessions), not from the webhook.
        webhook_payload = build_session_result_payload(sess)
        db.commit()

    # Outside the `with get_db()` block on purpose — same reasoning as
    # routers/sessions.py.save_session: the ~6-minute worst-case retry
    # schedule must never delay this disconnect handler from returning.
    asyncio.create_task(send_session_result_webhook(webhook_payload))


@router.websocket("/ws/pose")
async def ws_pose(websocket: WebSocket):
    if not check_ws_origin(websocket):
        await websocket.close(code=4009)  # custom code: origin not allowed
        return

    session_id = websocket.query_params.get("session_id")
    role = websocket.query_params.get("role", "patient")
    if not session_id or role not in ("patient", "doctor"):
        # 4000: bad request — missing/invalid session_id or role. Distinct
        # from 4009 (origin) so client-side logging can tell them apart.
        await websocket.close(code=4000)
        return

    if role == "patient":
        authorized = await _authorize_patient_connection(websocket, session_id)
        if not authorized:
            return  # socket already closed with the appropriate code
    elif role == "doctor":
        authorized = await _authorize_doctor_connection(websocket, session_id)
        if not authorized:
            return  # socket already closed with the appropriate code

    connection_id = await ws_mgr.connect(websocket, session_id, role)
    conn = ws_mgr.get_connection(websocket)
    camera = conn.camera  # None for doctor — see RoomConnection in camera_ws.py

    try:
        if camera:
            camera.start()
        await ws_mgr.send(websocket, {
            "type": "connected",
            "connection_id": connection_id,
            "role": role,
        })

        # Cloud refactor (Phase B): the server no longer polls a local
        # device in a loop — the browser pushes frames, so this loop is
        # receive-driven. Frame flow direction flips (browser → server →
        # browser) but it's still the one /ws/pose channel from Phase A,
        # still one CameraManager + SessionMetrics per connection.
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            # Item 26: check BEFORE anything else in the loop — an
            # ahead-of-deployment client shouldn't get partway through
            # frame processing before we notice and close.
            mismatch_code = check_client_protocol_version(msg)
            if mismatch_code is not None:
                await ws_mgr.send(websocket, build_version_mismatch_message(
                    client_v=client_message_version(msg),
                ))
                await websocket.close(code=mismatch_code)
                return

            # Doctor connections are read-only — they only ever receive
            # broadcasts, never drive the session. RoomManager already
            # gives a doctor connection no CameraManager (so there's
            # nothing to process frames with even if this check were
            # missing), but we still ignore the message explicitly here
            # instead of falling through to a None-camera crash.
            if role == "doctor":
                continue

            if msg_type == "set_exercise":
                camera.metrics.set_exercise_state(
                    exercise_type=msg.get("exercise_type"),
                    target_rom=msg.get("target_rom"),
                )
                camera.metrics.set_affected_side(msg.get("affected_side"))
                continue

            if msg_type == "reset_session":
                camera.metrics.reset()
                continue

            if msg_type != "frame":
                continue  # ignore anything we don't recognize

            # Track inbound rate BEFORE the skip decision — this is what
            # lets us detect "browser sending faster than we asked" even
            # on frames we're about to discard (real backpressure needs
            # to see the true arrival rate, not just the processed rate).
            camera.note_arrival()

            # 1. Frame skip — adaptive (see CameraManager.should_process):
            #    starts at 1-in-2, grows under sustained load, eases back
            #    down once load clears.
            if not camera.should_process():
                continue

            # 2. FPS throttle — don't send annotated frames back faster
            #    than TARGET_FPS, even if the browser pushes faster
            if not camera.fps_throttle():
                continue

            frame_start = _time.monotonic()

            # 3. Decode the browser-sent JPEG (base64 → bytes → cv2 frame)
            #    — this is the browser-push equivalent of the old
            #    `self.cap.read()`
            try:
                raw = base64.b64decode(msg.get("data", ""))
            except Exception:
                continue
            frame = camera.decode_frame(raw)
            if frame is None:
                continue

            # 4. Pose/hand processing — this connection's own CameraManager,
            #    so exercise type / rep count / scores are this session's own
            annotated, pose_data = camera.process_frame(frame)
            camera.note_processed(_time.monotonic() - frame_start)
            if annotated is None:
                continue

            # 5. Re-encode the annotated frame at lower quality for the
            #    return trip
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
            success, buf  = cv2.imencode(".jpg", annotated, encode_params)
            if not success:
                continue

            # 6. Broadcast annotated frame + slim pose_data to everyone in
            #    this session's room — the patient's own client (renders
            #    its own overlay) and any doctor connection watching this
            #    session_id. Previously this only echoed back to the
            #    sender, which meant a doctor connection had no way to see
            #    a patient's live pose_data at all.
            await ws_mgr.broadcast(session_id, {
                "type":      "pose_data",
                "frame":     base64.b64encode(buf).decode(),
                "pose_data": pose_data,
                "ts":        utcnow_iso(),
            })

            # Real backpressure signal (item: "frame rate limiting is just
            # frame-skip/FPS-throttle, not real backpressure"): when this
            # connection is genuinely congested — processing can't keep up
            # with TARGET_FPS, or the browser is sending faster than the
            # current adaptive skip ratio expects — tell the sender what
            # rate we can actually sustain right now. Rate-limited to at
            # most once every few seconds (should_send_backpressure_signal)
            # so this never floods the channel. Sent only to the patient
            # connection that's actually producing frames, not broadcast.
            if camera.should_send_backpressure_signal():
                await ws_mgr.send(websocket, {
                    "type":               "backpressure",
                    "recommended_fps":    camera.recommended_client_fps(),
                    "reason":             "server_congested",
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if role == "patient":
            # Mid-session disconnect: save whatever we captured instead of
            # losing it silently if the finishing POST /api/sessions never
            # arrives (crash, network loss, tab closed, phone locked...).
            await _abandon_if_unfinished(session_id, camera)
        ws_mgr.disconnect(websocket)