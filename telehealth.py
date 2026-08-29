import os
import secrets
import datetime
import asyncio
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from auth import CurrentTherapist, get_current_therapist, check_ws_origin
from config import templates
from database import get_db
from models import Patient, TelehealthRoom, SessionModel, JointAngle, ExerciseResult, History
from repositories.patient_repo import get_owned_patient
from repositories.room_repo import get_owned_room
from services.camera_ws import CameraManager
from services.timeutils import utcnow, utcnow_iso
from services.helpers import (
    derive_session_summary_stats,
    stamp as _stamp,
    client_message_version,
    check_client_protocol_version,
    build_version_mismatch_message,
)
from services.webhook import (
    build_session_result_payload, send_session_result_webhook,
    build_session_scheduled_payload, send_session_scheduled_webhook,
)
from models import VALID_END_REASONS

router = APIRouter()

# Every room link — Remote or Self Training — dies 2h after its scheduled
# time, regardless of whether it was ever opened. Matches the "scheduled
# time la irunthu 2hr la expire aaganum" requirement for both modes.
ROOM_LINK_VALID_HOURS = 2
VALID_MODES = ("remote", "self_training")

# How long a Remote room stays alive after the DOCTOR's socket drops
# (browser closed by mistake, black screen from a Zoom camera conflict,
# tab reload, network blip, etc.) before we give up and finalize/close
# it for real. The patient side (templates/patient.html, 'peer_left'
# handler) already just shows "Waiting for doctor to join..." and keeps
# running/streaming during this window — it does NOT end the call on its
# own. This grace period is what lets the doctor reopen the room (via
# "Join Room" again in the Remote Sessions list) and resume the SAME
# live session, instead of every doctor-side hiccup permanently killing
# the room and losing whatever the patient was mid-exercise on.
# A PATIENT disconnect is treated differently (still ends the room
# immediately, below) since patient.html has no reconnect flow at all —
# unlike the doctor, an unreported patient drop really does mean nobody's
# there to keep exercising.
DOCTOR_RECONNECT_GRACE_SECONDS = 180

# Needed to turn the relative join_url ("/join/{room_id}?token=...") into
# an absolute link before it's sent to Laravel in the session-scheduled
# webhook (services/webhook.send_session_scheduled_webhook) — a relative
# path means nothing outside thero's own frontend, and Laravel is on a
# different domain entirely. The API response to the therapist's own
# browser (same origin) keeps using the relative path unchanged, since
# that's always worked fine there.
THERO_PUBLIC_BASE_URL = os.getenv("THERO_PUBLIC_BASE_URL", "").rstrip("/")


def _room_expired(room: TelehealthRoom) -> bool:
    return utcnow() > room.expires_at

#In-memory room registry — doctor + patient pose-tracking "live room".
# Repurposed for the MedNova integration: this used to relay raw WebRTC
# signaling messages between the two sides. Now the patient's camera
# frames are processed server-side (through CameraManager.process_frame,
# the same pipeline /ws/pose uses) and the annotated frame + pose_data is
# broadcast to BOTH sides, while doctor control messages (set_exercise,
# end_session) are relayed to the patient only.
class RoomManager:

    def __init__(self):
        self.rooms: Dict[str, Dict[str, Optional[WebSocket]]] = {}
        # One CameraManager per room — holds this room's own pose pipeline
        # + SessionMetrics, same isolation model as /ws/pose's per-connection
        # CameraManager (services/camera_ws.py).
        self.cameras: Dict[str, CameraManager] = {}
        # Item 5: patient's stop-reason sheet (leaveCallWithReason() in
        # patient.html) sends a 'patient_leaving' signal message carrying
        # WHY (pain/stopped_by_patient/technical_error/completed) an
        # instant before its socket actually closes. Stashed here so
        # _finalize_remote_session (called from the disconnect handler)
        # can use the patient's own reason instead of falling back to a
        # generic "disconnected" — popped the moment it's consumed.
        self.pending_end_reason: Dict[str, str] = {}
        # Doctor-reconnect grace period: room_id -> the pending asyncio
        # Task that will finalize+close the room if the doctor doesn't
        # come back in time. Cancelled the moment the doctor reconnects
        # (see register() below) or the room ends for any other reason.
        self.doctor_grace_tasks: Dict[str, asyncio.Task] = {}

    def cancel_doctor_grace(self, room_id: str):
        task = self.doctor_grace_tasks.pop(room_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def register(self, room_id: str, role: str, ws: WebSocket):
        self.rooms.setdefault(room_id, {"doctor": None, "patient": None})

        # Doctor reconnecting (whether inside or after the grace window
        # technically raced past it) — cancel any pending auto-close so
        # it doesn't fire and yank the rug out from under the session
        # that's now live again.
        if role == "doctor":
            self.cancel_doctor_grace(room_id)

        # Item 3: multi-device/multi-tab guard. Previously this just did
        # `self.rooms[room_id][role] = ws` unconditionally — if a doctor
        # or patient opened the same room in a second tab/device, the
        # FIRST connection got silently orphaned here with zero warning:
        # its socket stayed technically open but nobody was listening to
        # it anymore, so it just went dead the next time it tried to
        # send/receive. Now we explicitly warn + close the old one with a
        # distinct code (4010) so that tab can show the patient/doctor a
        # real "you're connected elsewhere now" message instead of a
        # silent hang.
        existing = self.rooms[room_id][role]
        if existing is not None and existing is not ws:
            try:
                await existing.send_json(_stamp({
                    "type": "kicked",
                    "reason": "connected_elsewhere",
                }))
            except Exception:
                pass
            try:
                await existing.close(code=4010)  # custom: replaced by a newer connection
            except Exception:
                pass

        self.rooms[room_id][role] = ws
        if room_id not in self.cameras:
            self.cameras[room_id] = CameraManager()
            self.cameras[room_id].start()

    def unregister(self, room_id: str, role: str, ws: WebSocket):
        room = self.rooms.get(room_id)
        if room and room.get(role) is ws:
            room[role] = None
        if room and not room["doctor"] and not room["patient"]:
            self.rooms.pop(room_id, None)
            cam = self.cameras.pop(room_id, None)
            if cam is not None:
                cam.stop()

    def get_camera(self, room_id: str) -> Optional[CameraManager]:
        return self.cameras.get(room_id)

    async def relay(self, room_id: str, from_role: str, data: dict):
        """Doctor-sent control messages (set_exercise, end_session) go to
        the patient only — patients never need to relay anything to the
        doctor this way (their frames go through process_and_broadcast)."""
        room = self.rooms.get(room_id)
        if not room:
            return
        target_ws = room.get("patient" if from_role == "doctor" else "doctor")
        if target_ws is None:
            return
        try:
            await target_ws.send_json(_stamp(data))  # item 26
        except Exception:
            pass

    async def broadcast(self, room_id: str, data: dict):
        room = self.rooms.get(room_id)
        if not room:
            return
        _stamp(data)  # item 26 — stamp once, same dict sent to both sides
        for ws in (room.get("doctor"), room.get("patient")):
            if ws is not None:
                try:
                    await ws.send_json(data)
                except Exception:
                    pass

    async def process_and_broadcast(self, room_id: str, raw_frame_b64: str):
        """Patient's raw camera frame -> CameraManager.process_frame() ->
        annotated frame + pose_data -> broadcast to doctor AND patient, so
        both sides see the same live skeleton/metrics overlay."""
        import base64
        import cv2

        camera = self.cameras.get(room_id)
        if camera is None:
            return
        if not camera.should_process() or not camera.fps_throttle():
            return
        try:
            raw = base64.b64decode(raw_frame_b64)
        except Exception:
            return
        frame = camera.decode_frame(raw)
        if frame is None:
            return
        annotated, pose_data = camera.process_frame(frame)
        if annotated is None:
            return
        success, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not success:
            return
        await self.broadcast(room_id, {
            "type":      "pose_data",
            "frame":     base64.b64encode(buf).decode(),
            "pose_data": pose_data,
            "ts":        utcnow_iso(),
        })


room_mgr = RoomManager()


#REST: room lifecycle
@router.post("/api/telehealth/create-room")
async def create_room(
    payload: Dict[str, Any],
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    """
    Doctor schedules a session — either mode:
      mode = "remote"          doctor joins live too, at scheduled_at
      mode = "self_training"   patient does it alone, any time up to expiry

    Both are appointment-based now: scheduled_at is required, and the
    resulting link is only valid for ROOM_LINK_VALID_HOURS after it.
    """
    patient_id    = payload.get("patient_id")
    exercise_type = payload.get("exercise_type")
    affected_side = payload.get("affected_side")  # "left" | "right" | "both" — validated/normalized below
    mode          = payload.get("mode", "remote")
    scheduled_raw = payload.get("scheduled_at")
    target_rom_raw = payload.get("target_rom")

    if mode not in VALID_MODES:
        raise HTTPException(400, f"mode must be one of {VALID_MODES}")
    if not scheduled_raw:
        raise HTTPException(400, "scheduled_at is required (ISO datetime)")

    # Self Training has no live doctor socket to send a "set_exercise"
    # message later (unlike Remote — see ws_signal's set_exercise
    # handling), so target_rom MUST be set by the therapist right here at
    # scheduling time. Without this it was silently falling back to a
    # flat 90.0 for every exercise type. Remote mode keeps it optional —
    # the doctor can still set it live once both sides connect.
    target_rom = None
    if target_rom_raw is not None:
        try:
            target_rom = float(target_rom_raw)
        except (TypeError, ValueError):
            raise HTTPException(400, "target_rom must be a number")
        if target_rom <= 0:
            raise HTTPException(400, "target_rom must be greater than 0")
    if mode == "self_training" and target_rom is None:
        raise HTTPException(
            400,
            "target_rom is required when mode is self_training — there's no "
            "live doctor connection to set it later, so it must be provided "
            "when scheduling the session",
        )
    try:
        v = scheduled_raw[:-1] + "+00:00" if scheduled_raw.endswith("Z") else scheduled_raw
        scheduled_at = datetime.datetime.fromisoformat(v)
    except ValueError:
        raise HTTPException(400, "scheduled_at must be a valid ISO datetime")

    scheduled_webhook_payload = None
    relative_join_url = None

    with get_db() as db:
        # Patient must exist AND already belong to this therapist — for
        # MedNova-synced patients that's the row /integration/sync-patient
        # created, so no separate patient-id system is needed here.
        patient = get_owned_patient(db, patient_id, therapist)

        room = TelehealthRoom(
            token         = secrets.token_urlsafe(24),
            patient_id    = patient_id,
            mednova_consultant_id = therapist.mednova_consultant_id,
            exercise_type = exercise_type,
            target_rom    = target_rom,
            affected_side = (affected_side or "both").strip().lower()
                            if (affected_side or "").strip().lower() in ("left", "right", "both")
                            else "both",
            mode          = mode,
            status        = "pending",
            scheduled_at  = scheduled_at,
            expires_at    = scheduled_at + datetime.timedelta(hours=ROOM_LINK_VALID_HOURS),
        )
        db.add(room)
        db.commit()
        db.refresh(room)

        relative_join_url = f"/join/{room.id}?token={room.token}"

        # Session-scheduled webhook (thero -> Laravel): fires for BOTH
        # modes (remote and self_training) so Laravel can route the
        # patient to their join link — see services/webhook.py module
        # docstring. Build the payload here, while `room`/`patient` are
        # still attached to this open db session (same
        # DetachedInstanceError caveat as build_session_result_payload).
        # Laravel's receiving endpoint doesn't exist yet as of this
        # writing, so this will just log-and-skip until
        # MEDNOVA_SCHEDULE_WEBHOOK_URL is set — safe to wire in now.
        if THERO_PUBLIC_BASE_URL:
            absolute_join_url = f"{THERO_PUBLIC_BASE_URL}{relative_join_url}"
            scheduled_webhook_payload = build_session_scheduled_payload(
                room, patient, absolute_join_url,
            )
        else:
            logging.getLogger("thero.webhook").warning(
                "THERO_PUBLIC_BASE_URL is not set — cannot build an "
                "absolute join_url for the session-scheduled webhook "
                "(room_id=%s); skipping delivery to Laravel", room.id,
            )

        response_body = {
            "room_id":      room.id,
            "token":        room.token,
            "mode":         room.mode,
            "join_url":     relative_join_url,
            "status":       room.status,
            "affected_side": room.affected_side,
            "target_rom":   room.target_rom,
            "scheduled_at": room.scheduled_at.isoformat(),
            "expires_at":   room.expires_at.isoformat(),
        }

    # Fired OUTSIDE the `with get_db()` block, same reasoning as the
    # session-results webhook elsewhere in this file: the retry schedule
    # (worst case ~6.3 minutes) must never delay this response back to
    # the therapist.
    if scheduled_webhook_payload:
        asyncio.create_task(send_session_scheduled_webhook(scheduled_webhook_payload))

    return JSONResponse(response_body)


NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@router.get("/join/{room_id}", response_class=HTMLResponse)
async def page_join(request: Request, room_id: str):
    """
    Patient-facing join page (Remote + Self Training). Moved here from the
    old routers/pages.py during the UI-pages removal pass — this is the
    one page that's still actively used (the therapist-side pages —
    patients/session/analytics/reports — were removed since the
    therapist/doctor UI now lives outside thero). Keeps living next to the
    rest of the telehealth room logic instead of a standalone pages router
    that would otherwise only exist for this single route.
    """
    return templates.TemplateResponse(request, "patient.html", {"room_id": room_id}, headers=NO_CACHE_HEADERS)


@router.get("/api/telehealth/room/{room_id}")
async def get_room(room_id: str, token: str):
    # Patient-facing route — gated by the room's own token, not a
    # therapist JWT (patient never logs in). Intentionally does NOT use
    # room_repo, which is therapist-scoped only.
    with get_db() as db:
        room = db.query(TelehealthRoom).filter(TelehealthRoom.id == room_id).first()
        if not room or room.token != token:
            raise HTTPException(404, "Room not found or invalid link")
        if room.status == "closed":
            raise HTTPException(410, "This session has ended")
        if _room_expired(room):
            # Not "closed" in the DB (maybe nobody ever joined), but the
            # 2h window from scheduled_at has passed either way.
            raise HTTPException(410, "This link has expired — ask your therapist to schedule a new session")

        # Self Training never opens a doctor-visible WS (unlike Remote,
        # which flips to "live" the moment the patient connects to
        # /ws/signal/{room_id}) — the patient's page does pose detection
        # fully client-side and only talks to the backend again at the very
        # end (POST .../self-training/save). So THIS request — the patient's
        # page loading and calling GET /room right before showing the
        # "Start Camera & Begin" button — is the only signal we get that
        # they've opened the link. Use it as the "joined" transition so the
        # doctor's Self Training list can show "Joined" instead of being
        # stuck on "Scheduled" until the session is already finished.
        if room.mode == "self_training" and room.status == "pending":
            room.status = "live"
            room.started_at = utcnow()
            db.commit()
            db.refresh(room)

        patient = db.query(Patient).filter(Patient.id == room.patient_id).first()
        return JSONResponse({
            "room_id":       room.id,
            "mode":          room.mode,
            "status":        room.status,
            "patient_name":  patient.name if patient else "Patient",
            "exercise_type": room.exercise_type,
            "target_rom":    room.target_rom,
            "scheduled_at":  room.scheduled_at.isoformat(),
            "expires_at":    room.expires_at.isoformat(),
        })


@router.get("/api/telehealth/room-status/{room_id}")
async def room_status(
    room_id: str,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    """
    Doctor-side polling endpoint for session.html's "Remote Sessions" /
    "Self Training Sessions" lists. Authenticated by the therapist's own
    session (like create-room/close-room), NOT the room's patient-facing
    token, since the doctor is already logged in.

    TelehealthRoom.status -> what the doctor UI shows:
      "pending" -> "scheduled"   patient hasn't opened the link yet
      "live"    -> "joined"      patient opened the link / is exercising
      "closed"  -> "completed"   patient finished — session_id is set and
                                  the saved metrics are returned inline so
                                  the doctor can pop the report without a
                                  second request
    A Remote room closed with nobody having joined is also "closed" in the
    DB but has no session_id — that's returned as "completed" with
    session: null rather than pretending it's still pending, since the
    doctor already ended it.
    """
    with get_db() as db:
        room = get_owned_room(db, room_id, therapist)

        session_payload = None
        if room.status == "closed" and room.session_id:
            sess = db.query(SessionModel).filter(SessionModel.id == room.session_id).first()
            if sess:
                session_payload = {
                    "exercise_type":       sess.exercise_type,
                    "duration_seconds":    sess.duration_seconds,
                    "completed_reps":      sess.completed_reps,
                    "total_reps":          sess.total_reps,
                    "accuracy_percentage": sess.accuracy_percentage,
                    "average_rom":         sess.average_rom,
                    "stability_score":     sess.stability_score,
                    "session_data":        sess.session_data,
                    "end_reason":          sess.end_reason,  # item 24
                }

        mapped = {"pending": "scheduled", "live": "joined", "closed": "completed"}.get(room.status, room.status)
        return JSONResponse({"status": mapped, "session": session_payload})


async def _finalize_remote_session(db, room: TelehealthRoom, default_end_reason: str) -> Optional[Dict[str, Any]]:
    """
    Item 5: persist whatever this Remote room's server-side CameraManager
    (room_mgr.cameras[room.id] — the same pipeline driving the doctor's
    live reps/accuracy view during the call) has accumulated, as a normal
    SessionModel row. Same destination/shape as Self Training's save
    endpoint and routers/ws.py's abandon-on-disconnect handler — Reports/
    Analytics/patient history don't need to know this came from a Remote
    call. Call this from wherever a Remote room's life actually ends:
    the doctor's explicit "End Remote Session" (close_room) AND a socket
    drop (ws_signal's finally block) both call this.

    Idempotent by design: a room only ever gets ONE session_id, so
    whichever of those two call sites gets there first wins — the other
    sees room.session_id already set and no-ops. Returns None when there
    is nothing worth saving (nobody ever actually joined, so no camera
    ever started) or when a save already happened.
    """
    if room.session_id:
        return None  # already saved by the other call site

    cam = room_mgr.get_camera(room.id)
    if cam is None or room.started_at is None:
        return None  # room never went live — nothing was ever captured

    m = cam.metrics
    reps = m.get_rep_count()
    exercise_type, target_rom = m.get_exercise_state()

    end_reason = room_mgr.pending_end_reason.pop(room.id, None) or default_end_reason
    if end_reason not in VALID_END_REASONS:
        end_reason = "disconnected"

    # A bare, unreported socket drop is "abandoned" (same convention as
    # routers/ws.py's abandon-on-disconnect). An explicit therapist end,
    # or a reason the patient actually reported on their stop sheet, is a
    # normal orderly close — "completed", same as Self Training treats
    # all 4 of its own reasons.
    status = "abandoned" if end_reason == "disconnected" else "completed"

    started = room.started_at or room.created_at
    ended   = utcnow()

    sess = SessionModel(
        patient_id           = room.patient_id,
        exercise_type        = exercise_type or room.exercise_type or "General Exercise",
        affected_side        = m.get_affected_side() or room.affected_side or "both",
        start_time           = started,
        end_time             = ended,
        duration_seconds     = int((ended - started).total_seconds()) if started else 0,
        total_reps           = reps,
        completed_reps       = reps,
        accuracy_percentage  = m.get_accuracy(),
        average_rom          = target_rom or 0.0,
        stability_score      = m.get_stability(),
        balance_score        = m.get_balance(),
        movement_smoothness  = m.get_smoothness(),
        fatigue_estimation   = m.get_current_fatigue(),
        recovery_score       = round((m.get_accuracy() + m.get_stability() + m.get_balance()) / 3, 1),
        status               = status,
        end_reason           = end_reason,
    )
    db.add(sess)
    db.flush()

    room.session_id = sess.id
    db.add(History(
        patient_id = room.patient_id,
        action     = "Remote Session Completed" if status == "completed" else "Remote Session Disconnected",
        details    = f"Room {room.id} — {reps} reps, ended: {end_reason}",
    ))

    # Item 27: build BEFORE commit/context exit — see
    # build_session_result_payload() docstring for why.
    return build_session_result_payload(sess)


@router.post("/api/telehealth/close-room/{room_id}")
async def close_room(
    room_id: str,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    """
    Doctor's "End Session" button (Remote mode). Marks the room closed
    (link stops working from here on) and notifies any connected socket so
    the patient's page can show a "session ended" state immediately.
    """
    webhook_payload = None
    with get_db() as db:
        room = get_owned_room(db, room_id, therapist)
        webhook_payload = await _finalize_remote_session(db, room, default_end_reason="stopped_by_therapist")
        room.status    = "closed"
        room.closed_at = utcnow()
        db.commit()

    if webhook_payload:
        asyncio.create_task(send_session_result_webhook(webhook_payload))

    await room_mgr.broadcast(room_id, {"type": "session_closed"})
    return JSONResponse({"success": True, "message": "Room closed"})


@router.get("/api/telehealth/turn-credentials")
async def turn_credentials():
    ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]

    turn_url  = os.getenv("METERED_TURN_URL")
    turn_user = os.getenv("METERED_TURN_USERNAME")
    turn_cred = os.getenv("METERED_TURN_CREDENTIAL")

    if turn_url and turn_user and turn_cred:
        ice_servers.append({
            "urls":       turn_url,
            "username":   turn_user,
            "credential": turn_cred,
        })

    return JSONResponse({"iceServers": ice_servers})


#WebSocket: pose-data live room
@router.websocket("/ws/signal/{room_id}")
async def ws_signal(websocket: WebSocket, room_id: str, role: str, token: str):

    if not check_ws_origin(websocket):
        await websocket.close(code=4009)  # custom code: origin not allowed
        return

    if role not in ("doctor", "patient"):
        await websocket.close(code=4000)
        return

    with get_db() as db:
        room = db.query(TelehealthRoom).filter(TelehealthRoom.id == room_id).first()
        if not room or room.token != token or room.mode != "remote":
            await websocket.close(code=4001)
            return
        if room.status == "closed":
            await websocket.close(code=4002)
            return
        if _room_expired(room):
            await websocket.close(code=4003)
            return

        await websocket.accept()
        await room_mgr.register(room_id, role, websocket)

        # First patient connection flips the room live and stamps started_at
        if role == "patient" and room.status == "pending":
            room.status = "live"
            room.started_at = utcnow()
            db.commit()

    # If the other side is already in the room, tell the socket that just
    # joined right away — this is what triggers the patient side to start
    # streaming frames without waiting for a fresh "peer_joined" event.
    other_role = "patient" if role == "doctor" else "doctor"
    other_ws = room_mgr.rooms.get(room_id, {}).get(other_role)
    if other_ws is not None:
        try:
            await websocket.send_json(_stamp({"type": "peer_already_present", "role": other_role}))
        except Exception:
            pass

    await room_mgr.broadcast(room_id, {"type": "peer_joined", "role": role})

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            # Item 26: check first, same as /ws/pose — see
            # routers/ws.py.ws_pose for the identical pattern.
            mismatch_code = check_client_protocol_version(data)
            if mismatch_code is not None:
                try:
                    await websocket.send_json(build_version_mismatch_message(
                        client_v=client_message_version(data),
                    ))
                except Exception:
                    pass
                await websocket.close(code=mismatch_code)
                return

            if role == "patient" and msg_type == "frame":
                # Patient's camera frame -> process through the pose
                # pipeline -> annotated frame + pose_data broadcast to
                # both doctor and patient.
                await room_mgr.process_and_broadcast(room_id, data.get("data", ""))
                continue

            if role == "doctor" and msg_type == "set_exercise":
                # Apply to THIS room's own server-side scoring camera —
                # previously this only relayed to the patient's browser
                # (below) and never reached room_mgr.cameras[room_id], so
                # the server-side pipeline that actually scores the
                # session (process_and_broadcast → CameraManager) stayed
                # on whatever exercise_type/target_rom/affected_side the
                # camera was constructed with (defaults), regardless of
                # what the doctor picked on their screen.
                cam = room_mgr.get_camera(room_id)
                if cam is not None:
                    cam.metrics.set_exercise_state(
                        exercise_type=data.get("exercise_type"),
                        target_rom=data.get("target_rom"),
                    )
                    cam.metrics.set_affected_side(data.get("affected_side"))

            if role == "patient" and msg_type == "patient_leaving":
                # Item 5: stash the patient's own reported reason so the
                # disconnect handler below can use it instead of a bare
                # "disconnected" once this socket actually closes a
                # moment later. Still relayed to the doctor live, below,
                # same as before.
                reason = data.get("reason")
                if reason in VALID_END_REASONS:
                    room_mgr.pending_end_reason[room_id] = reason

            # Doctor control messages (set_exercise, end_session, etc.) go
            # to the patient only.
            await room_mgr.relay(room_id, role, data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"ws_signal error ({role}, room={room_id}): {e}")
    finally:
        # Item 3: figure out BEFORE unregistering whether this socket was
        # still the one actually registered for this role. If a second
        # tab/device already took over (register() above kicked this one
        # with code 4010), this connection's own receive loop is now
        # unwinding too — without this check, that would fall through to
        # closing the whole room / broadcasting "peer_left" and immediately
        # kill the NEW connection that just replaced it. Only the
        # currently-registered socket disconnecting should end the room.
        was_current = room_mgr.rooms.get(room_id, {}).get(role) is websocket
        room_mgr.unregister(room_id, role, websocket)

        if not was_current:
            return

        await room_mgr.broadcast(room_id, {"type": "peer_left", "role": role})

        if role == "doctor":
            # DOCTOR disconnect: don't kill the room. patient.html already
            # just shows "Waiting for doctor to join..." and keeps
            # streaming (see 'peer_left' handler there), and unregister()
            # above only stops the room's CameraManager once BOTH sides
            # are gone — so reps/metrics keep accumulating while the
            # doctor's away. Give the doctor DOCTOR_RECONNECT_GRACE_SECONDS
            # to reopen the room (Join Room again) before we give up and
            # actually finalize/close it for real.
            room_mgr.cancel_doctor_grace(room_id)
            room_mgr.doctor_grace_tasks[room_id] = asyncio.create_task(
                _doctor_grace_timeout_close(room_id)
            )
            return

        # PATIENT disconnect (network loss, tab closed, phone locked,
        # etc.) — patient.html has no reconnect flow, so unlike the
        # doctor this really does mean the session is over. Finalize and
        # close immediately, same as before. Also cancel any doctor grace
        # timer still pending so it doesn't try to double-close later.
        room_mgr.cancel_doctor_grace(room_id)
        webhook_payload = None
        with get_db() as db:
            r = db.query(TelehealthRoom).filter(TelehealthRoom.id == room_id).first()
            if r and r.status != "closed":
                # Item 5: finalize (save) BEFORE flipping status to
                # "closed", using whatever the room's camera captured.
                # Read room_mgr.get_camera(room_id) here specifically
                # because it's still alive at this point even when
                # unregister() above didn't pop it (the doctor may still
                # be connected) — see _finalize_remote_session.
                webhook_payload = await _finalize_remote_session(db, r, default_end_reason="disconnected")
                r.status = "closed"
                r.closed_at = utcnow()
                db.commit()

        if webhook_payload:
            asyncio.create_task(send_session_result_webhook(webhook_payload))

        # Doctor may still be connected watching an now-abandoned room —
        # let them know explicitly rather than leaving them hanging.
        doctor_ws = room_mgr.rooms.get(room_id, {}).get("doctor")
        if doctor_ws is not None:
            try:
                await doctor_ws.send_json(_stamp({"type": "session_closed", "reason": "patient_disconnected"}))
            except Exception:
                pass


async def _doctor_grace_timeout_close(room_id: str):
    """Fires DOCTOR_RECONNECT_GRACE_SECONDS after a doctor disconnect. If
    the doctor still hasn't reconnected by then, actually finalize and
    close the room (same as an immediate close used to do), and tell the
    patient the session is over. If the doctor DID reconnect,
    register() already cancelled this task and we never get here."""
    try:
        await asyncio.sleep(DOCTOR_RECONNECT_GRACE_SECONDS)
    except asyncio.CancelledError:
        return

    room_mgr.doctor_grace_tasks.pop(room_id, None)

    webhook_payload = None
    with get_db() as db:
        r = db.query(TelehealthRoom).filter(TelehealthRoom.id == room_id).first()
        if r and r.status != "closed":
            webhook_payload = await _finalize_remote_session(db, r, default_end_reason="disconnected")
            r.status = "closed"
            r.closed_at = utcnow()
            db.commit()

    if webhook_payload:
        asyncio.create_task(send_session_result_webhook(webhook_payload))

    patient_ws = room_mgr.rooms.get(room_id, {}).get("patient")
    if patient_ws is not None:
        try:
            await patient_ws.send_json(_stamp({"type": "session_closed", "reason": "doctor_disconnected"}))
        except Exception:
            pass
        try:
            await patient_ws.close(code=4004)  # custom: doctor never reconnected within grace window
        except Exception:
            pass

# ── Self Training mode ──────────────────────────────────────────────────
# Doctor schedules it (create_room, mode="self_training"), patient opens
# the link ALONE later — no doctor socket, no room_mgr broadcast. Patient
# runs the same per-connection pose pipeline as /ws/pose, but scoped to
# this room's token instead of a therapist JWT (the patient never logs
# into thero or MedNova). On disconnect/finish, results save automatically
# to this patient's record — same SessionModel row a normal doctor-run
# session would produce, so analytics/reports/dashboard don't need to
# know the difference.
#
# Note: each connection below keeps its own local `camera` variable —
# there's no need for a module-level registry the way room_mgr.cameras
# exists for Remote mode, since nothing outside this one connection's
# handler (e.g. no doctor-side "set_exercise" relay) ever needs to reach
# another self-training session's CameraManager from the outside.


def _get_open_self_training_room(db, room_id: str, token: str) -> TelehealthRoom:
    # Patient-facing route — gated by the room's own token + expiry, NOT
    # therapist identity (patient never logs in). Intentionally does not
    # use room_repo, which is therapist-scoped only.
    room = db.query(TelehealthRoom).filter(TelehealthRoom.id == room_id).first()
    if not room or room.token != token or room.mode != "self_training":
        raise HTTPException(404, "Room not found or invalid link")
    if room.status == "closed":
        raise HTTPException(410, "This session has already been completed")
    if _room_expired(room):
        raise HTTPException(410, "This link has expired — ask your therapist to schedule a new session")
    return room


@router.websocket("/ws/self-training/{room_id}")
async def ws_self_training(websocket: WebSocket, room_id: str, token: str):
    """
    Patient-only pose pipeline for a scheduled Self Training room. Same
    message shapes and same per-connection CameraManager as /ws/pose
    (routers/ws.py) — just gated by the room token + expiry instead of a
    therapist JWT. This socket only streams pose_data back; it does NOT
    save anything itself. Exactly like the normal in-app flow, the
    patient's page accumulates reps/accuracy/etc. from these pose_data
    messages client-side (same math session.html already does — see
    saveSession() there) and then POSTs the finished payload to
    POST /api/telehealth/self-training/save/{room_id} below.
    """
    import base64
    import cv2

    if not check_ws_origin(websocket):
        await websocket.close(code=4009)  # custom code: origin not allowed
        return

    with get_db() as db:
        try:
            room = _get_open_self_training_room(db, room_id, token)
        except HTTPException:
            await websocket.close(code=4001)
            return
        if room.status == "pending":
            room.status = "live"
            room.started_at = utcnow()
            db.commit()

    await websocket.accept()
    camera = CameraManager()
    camera.start()
    if room.exercise_type:
        camera.metrics.set_exercise_state(exercise_type=room.exercise_type, target_rom=room.target_rom)
    camera.metrics.set_affected_side(room.affected_side)

    try:
        await websocket.send_json(_stamp({"type": "connected", "room_id": room_id}))
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            # Item 26: check first, same pattern as /ws/pose and
            # /ws/signal — see routers/ws.py.ws_pose for the original.
            mismatch_code = check_client_protocol_version(msg)
            if mismatch_code is not None:
                try:
                    await websocket.send_json(build_version_mismatch_message(
                        client_v=client_message_version(msg),
                    ))
                except Exception:
                    pass
                await websocket.close(code=mismatch_code)
                return

            if msg_type == "set_exercise":
                camera.metrics.set_exercise_state(
                    exercise_type=msg.get("exercise_type"), target_rom=msg.get("target_rom"),
                )
                camera.metrics.set_affected_side(msg.get("affected_side"))
                continue
            if msg_type != "frame":
                continue
            if not camera.should_process() or not camera.fps_throttle():
                continue
            try:
                raw = base64.b64decode(msg.get("data", ""))
            except Exception:
                continue
            frame = camera.decode_frame(raw)
            if frame is None:
                continue
            annotated, pose_data = camera.process_frame(frame)
            if annotated is None:
                continue
            success, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not success:
                continue
            await websocket.send_json(_stamp({
                "type":      "pose_data",
                "frame":     base64.b64encode(buf).decode(),
                "pose_data": pose_data,
                "ts":        utcnow_iso(),
            }))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"ws_self_training error (room={room_id}): {e}")
    finally:
        camera.stop()


@router.post("/api/telehealth/self-training/save/{room_id}")
async def save_self_training_session(room_id: str, token: str, payload: Dict[str, Any]):
    """
    Patient-side save — same payload shape as POST /api/sessions
    (total_reps, completed_reps, accuracy_percentage, joint_angles, ...;
    see sessions.save_session), but authenticated by the room's token
    instead of a therapist JWT, since the patient never logs in anywhere.
    The therapist-facing side (analytics/reports/history) never has to
    know a session came from Self Training vs a normal in-app one — it's
    the same SessionModel row, tied to the same patient_id.
    """
    with get_db() as db:
        room = _get_open_self_training_room(db, room_id, token)

        def _dt(key):
            v = payload.get(key)
            if not v:
                return None
            if v.endswith("Z"):
                v = v[:-1] + "+00:00"
            return datetime.datetime.fromisoformat(v)

        # Same guard as sessions.save_session — real semantics still
        # pending Nada's confirmation, this just stops obviously-invalid
        # data (completed exceeding total) from being persisted.
        total_reps     = payload.get("total_reps", 0) or 0
        completed_reps = payload.get("completed_reps", 0) or 0
        if total_reps > 0 and completed_reps > total_reps:
            completed_reps = total_reps

        # Server-derived from the itemized exercise_results/joint_angles
        # below, not trusted directly from the client's top-level summary
        # fields — same as sessions.save_session, see
        # helpers.derive_session_summary_stats docstring.
        accuracy_percentage, average_rom, incorrect_movements = \
            derive_session_summary_stats(payload)

        # Item 24 — same validated contract as routers/sessions.py.save_session.
        end_reason = payload.get("end_reason")
        if end_reason not in VALID_END_REASONS:
            end_reason = "completed"

        sess = SessionModel(
            patient_id          = room.patient_id,
            exercise_type       = payload.get("exercise_type") or room.exercise_type or "General Exercise",
            affected_side       = payload.get("affected_side") or room.affected_side or "both",
            start_time          = _dt("start_time") or room.started_at or room.created_at,
            end_time            = _dt("end_time") or utcnow(),
            duration_seconds    = payload.get("duration_seconds", 0),
            total_reps          = total_reps,
            completed_reps      = completed_reps,
            accuracy_percentage = accuracy_percentage,
            average_rom         = average_rom,
            incorrect_movements = incorrect_movements,
            stability_score     = payload.get("stability_score", 0.0),
            balance_score       = payload.get("balance_score", 0.0),
            movement_smoothness = payload.get("movement_smoothness", 0.0),
            fatigue_estimation  = payload.get("fatigue_estimation", 0.0),
            recovery_score      = payload.get("recovery_score", 0.0),
            session_data        = payload.get("session_data", {}),
            # Both of these were previously left at the model default
            # ("in_progress" / NULL) on this path — this endpoint only
            # ever runs once the patient has actually finished, so the row
            # should land as "completed" immediately, same as the normal
            # in-app save_session flow. Left uncaught before, this made
            # every Self Training session look permanently in-progress to
            # any code (or dashboard) filtering on status.
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
        for er in payload.get("exercise_results", []):
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

        room.session_id = sess.id
        room.status      = "closed"
        room.closed_at   = utcnow()
        db.add(History(
            patient_id = room.patient_id,
            action     = "Self Training Session Completed",
            details    = f"Room {room_id} — {sess.completed_reps} reps, ended: {end_reason}, patient alone (no therapist present)",
        ))

        # Item 27: webhook payload must be built BEFORE commit/context
        # exit — see services/webhook.build_session_result_payload()
        # docstring for why.
        webhook_payload = build_session_result_payload(sess)
        db.commit()

    # Outside the `with get_db()` block on purpose — same reasoning as
    # routers/sessions.py.save_session: the ~6-minute worst-case retry
    # schedule must never delay this response back to the patient.
    asyncio.create_task(send_session_result_webhook(webhook_payload))

    return JSONResponse({"success": True, "message": "Session saved", "session_id": sess.id})