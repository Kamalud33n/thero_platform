"""
Bridge-mode API — MedNova Care's Laravel backend calls these SERVER-TO-SERVER
mid-consultation (not a therapist using thero's own UI). Auth is HMAC via
auth.verify_bridge_hmac, NOT the therapist bridge JWT (get_current_therapist)
— there is no logged-in therapist browser session behind these requests.
"""
import datetime
import secrets
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from auth import verify_bridge_hmac
from database import get_db
from models import TelehealthRoom
from repositories.patient_repo import get_or_create_patient_by_external_id
from services.exercise_defs import EXERCISE_DEFS, resolve_exercise_code
from services.timeutils import utcnow
from telehealth import BRIDGE_LINK_VALID_HOURS

router = APIRouter()


@router.get("/api/bridge/exercise-types", dependencies=[Depends(verify_bridge_hmac)])
async def bridge_exercise_types():
    """
    Nada's Task 4. Static reference list for MedNovaCare's doctor-facing
    dropdown when requesting a measurement — the `exercise_type` value
    sent to POST /api/bridge/create-session should be one of these `key`s
    (services.exercise_defs.resolve_exercise_code will still fall back
    sensibly on an unrecognized value, but Laravel shouldn't rely on
    that; it's a legacy-data safety net, not a supported contract).

    Same HMAC dependency as the other bridge endpoints even though this
    one has no request body to sign over — verify_bridge_hmac only checks
    a static shared-secret header (see auth.py), so a GET with no body
    works the same as the POSTs above.

    Returns the current in-memory EXERCISE_DEFS as-is — there's no DB
    table behind this list (see services/exercise_defs.py's own module
    docstring for why codes are fixed/never renamed once shipped), so
    nothing here can go stale between requests within a single deploy.
    """
    return JSONResponse([
        {"key": code, "label": definition["display_name"]}
        for code, definition in EXERCISE_DEFS.items()
    ])


@router.post("/api/bridge/create-session", dependencies=[Depends(verify_bridge_hmac)])
async def bridge_create_session(payload: Dict[str, Any]):
    """
    Nada's Task 1. Auto-creates the patient if patient_external_id doesn't
    match an existing one, creates a room already configured (no
    SessionModel yet — see module docstring), and returns two URLs sharing
    the room's own token, differentiated by role= — same scheme
    remote/self_training rooms already use (telehealth.py's create_room,
    token = secrets.token_urlsafe(24)).

    Request body (as documented — see module docstring item 1 for the
    identity gap this doesn't yet resolve):
        consultation_id, patient_name, patient_external_id, exercise_type,
        affected_side, target_rom, target_reps, duration_seconds

    target_rom and target_reps are REQUIRED unless exercise_type resolves
    to a whole-body exercise (currently only BALANCE) — see the
    is_whole_body_exercise check below.
    """
    consultation_id = payload.get("consultation_id")
    patient_name = payload.get("patient_name")
    patient_external_id = payload.get("patient_external_id")
    exercise_type = payload.get("exercise_type")
    affected_side = payload.get("affected_side")
    target_rom_raw = payload.get("target_rom")
    target_reps_raw = payload.get("target_reps")
    duration_seconds = payload.get("duration_seconds")

    # GUESS at the eventual field name — see module docstring item 1.
    mednova_consultant_id = payload.get("mednova_consultant_id")

    if not consultation_id:
        raise HTTPException(400, "consultation_id is required")
    if not patient_name:
        raise HTTPException(400, "patient_name is required")
    if not patient_external_id:
        raise HTTPException(400, "patient_external_id is required")
    if not mednova_consultant_id:
        # Fails loudly rather than creating an unowned patient/room — see
        # get_or_create_patient_by_external_id()'s own fail-closed check,
        # this is the same guarantee enforced one layer up.
        raise HTTPException(
            400,
            "mednova_consultant_id is required — this field is not yet "
            "confirmed with Nada, see routers/bridge.py module docstring",
        )
    # (affected_side validated further down — needs is_whole_body_exercise
    # resolved first.)

    # Resolved once, up front — both the affected_side exception below and
    # the target_rom/target_reps requirement further down key off whether
    # this exercise is "whole-body" (today: BALANCE — see
    # services/exercise_defs.py, normal_range_deg=None marks "not an
    # angle-based exercise", and BALANCE's own progress metric is
    # duration_seconds/hold time, not a rep count or a side). Resolved via
    # resolve_exercise_code() so this also handles a legacy display name
    # or free-text exercise_type value the same way the rest of the
    # codebase does, instead of a raw string comparison against "BALANCE".
    resolved_code = resolve_exercise_code(exercise_type)
    is_whole_body_exercise = EXERCISE_DEFS[resolved_code]["normal_range_deg"] is None

    # affected_side has no meaning for a whole-body exercise (BALANCE) —
    # there's no single limb/side being measured — so it's optional there
    # and quietly defaults to "both" (matches TelehealthRoom.affected_side's
    # own column default) when the caller omits it. Every other exercise
    # type still requires an explicit left/right/both, same as before.
    if is_whole_body_exercise:
        if affected_side is None:
            affected_side = "both"
        elif affected_side not in ("left", "right", "both"):
            raise HTTPException(400, "affected_side must be one of: left, right, both")
    else:
        if affected_side not in ("left", "right", "both"):
            raise HTTPException(400, "affected_side must be one of: left, right, both")

    if not is_whole_body_exercise:
        if target_rom_raw is None:
            raise HTTPException(
                400,
                f"target_rom is required for exercise_type '{exercise_type}'",
            )
        if target_reps_raw is None:
            raise HTTPException(
                400,
                f"target_reps is required for exercise_type '{exercise_type}'",
            )

    target_rom = None
    if target_rom_raw is not None:
        try:
            target_rom = float(target_rom_raw)
        except (TypeError, ValueError):
            raise HTTPException(400, "target_rom must be a number")
        if target_rom <= 0:
            raise HTTPException(400, "target_rom must be greater than 0")

    # Same optional/validated-if-present treatment as target_rom above —
    # doctor sets this in MedNovaCare's measurement form (Kamal's Task 1),
    # not derived from anything Thero itself computes. Whether it's
    # actually optional or required at this point was decided above
    # (is_whole_body_exercise) — this block only handles type/value
    # validation for whichever of the two it turned out to be.
    target_reps = None
    if target_reps_raw is not None:
        try:
            target_reps = int(target_reps_raw)
        except (TypeError, ValueError):
            raise HTTPException(400, "target_reps must be a whole number")
        if target_reps <= 0:
            raise HTTPException(400, "target_reps must be greater than 0")

    # duration_seconds — required for EVERY bridge session, including
    # BALANCE (models.py TelehealthRoom docstring: "Nullable: remote/
    # self_training don't currently set a target duration at scheduling
    # time ... only bridge mode requires it"). Bridge rooms have no live
    # doctor connection to set/adjust a target length later the way
    # remote mode can, so this has to arrive up front — previously
    # accepted whatever the caller sent (including nothing at all) with
    # zero validation.
    if duration_seconds is None:
        raise HTTPException(400, "duration_seconds is required")
    try:
        duration_seconds = int(duration_seconds)
    except (TypeError, ValueError):
        raise HTTPException(400, "duration_seconds must be a whole number")
    if duration_seconds <= 0:
        raise HTTPException(400, "duration_seconds must be greater than 0")

    now = utcnow()
    expires_at = now + datetime.timedelta(hours=BRIDGE_LINK_VALID_HOURS)

    with get_db() as db:
        patient = get_or_create_patient_by_external_id(
            db,
            external_id=patient_external_id,
            name=patient_name,
            mednova_consultant_id=mednova_consultant_id,
        )

        room = TelehealthRoom(
            token=secrets.token_urlsafe(24),  # same generation as telehealth.py's create_room
            patient_id=patient.id,
            mednova_consultant_id=mednova_consultant_id,
            exercise_type=exercise_type,
            affected_side=affected_side,
            target_rom=target_rom,
            target_reps=target_reps,
            duration_seconds=duration_seconds,
            consultation_id=consultation_id,
            mode="bridge",
            status="pending",
            scheduled_at=now,
            expires_at=expires_at,
        )
        db.add(room)
        db.commit()
        db.refresh(room)

        # therapist_url points at the existing /session dashboard page
        # (routers/pages.py) — its page-shell route has no server-side auth
        # at all (the JWT check happens per-API-call from inside the page,
        # not on the page load itself), so it was already safe to open
        # without a MedNova dashboard session. session.html's own
        # initBridgeMode() reads bridge_room_id/bridge_token from the
        # query string, hides the normal dashboard chrome, and joins
        # directly using room.token — see templates/session.html.
        #
        # patient_url reuses the EXISTING /join/{room_id} patient page
        # (telehealth.py's page_join) completely unchanged — patient.html
        # only special-cases mode == "self_training"; everything else
        # (including "bridge") already falls through to the same live-call
        # UI remote-mode patients use, so no patient-side page was needed.
        therapist_url = f"/session?bridge_room_id={room.id}&bridge_token={room.token}"
        patient_url = f"/join/{room.id}?token={room.token}"

        return JSONResponse({
            "room_id": room.id,
            "therapist_url": therapist_url,
            "patient_url": patient_url,
            "expires_at": expires_at.isoformat(),
        })


@router.post("/api/bridge/cancel-session", dependencies=[Depends(verify_bridge_hmac)])
async def bridge_cancel_session(payload: Dict[str, Any]):
    """
    Nada's Task 3, confirmed 2026-08-30: room_id in the body. Only valid
    before the patient has joined (room.status == "pending") — matches her
    stated use case ("doctor changes their mind before patient joins").
    No webhook fires on cancel, per her spec. (Nothing to finalize/save
    either — no SessionModel exists yet at this point, see module
    docstring.)
    """
    room_id = payload.get("room_id")
    if not room_id:
        raise HTTPException(400, "room_id is required")

    with get_db() as db:
        room = db.query(TelehealthRoom).filter(TelehealthRoom.id == room_id).first()
        if room is None:
            raise HTTPException(404, "Room not found")
        if room.mode != "bridge":
            # Bridge cancel is only for bridge-created rooms — a
            # remote/self_training room scheduled from thero's own UI has
            # its own lifecycle (telehealth.py close-room) and no HMAC
            # caller should be able to touch it.
            raise HTTPException(404, "Room not found")
        if room.status != "pending":
            raise HTTPException(
                409,
                f"Cannot cancel — room status is '{room.status}', not 'pending' "
                "(patient may have already joined)",
            )

        room.status = "cancelled"
        room.closed_at = utcnow()
        db.commit()

        return JSONResponse({"ok": True, "room_id": room.id, "status": "cancelled"})
