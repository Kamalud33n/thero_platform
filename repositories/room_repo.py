"""
Single source of truth for "which telehealth rooms does this therapist own".

Same rationale as patient_repo.py — telehealth.py's room-status/close-room
endpoints check TelehealthRoom.mednova_consultant_id directly today,
duplicated across routes. Centralised here for the same reason: one place
where a forgotten filter is impossible instead of N places where it's
merely unlikely.

Note: patient-facing self-training/remote-join endpoints (token + room_id,
no logged-in therapist) intentionally do NOT go through this file — those
are gated by the room's own token + expiry, not therapist identity. This
file is only for the doctor-side, JWT-authenticated routes.
"""
from fastapi import HTTPException

from auth import CurrentTherapist
from models import TelehealthRoom


def owned_rooms_query(db, therapist: CurrentTherapist):
    """Base query scoped to this therapist's rooms only."""
    return db.query(TelehealthRoom).filter(
        TelehealthRoom.mednova_consultant_id == therapist.mednova_consultant_id
    )


def get_owned_room(db, room_id: str, therapist: CurrentTherapist) -> TelehealthRoom:
    """Fetch one room, scoped to this therapist. 404 whether it doesn't
    exist or belongs to someone else — same reasoning as get_owned_patient."""
    room = owned_rooms_query(db, therapist).filter(TelehealthRoom.id == room_id).first()
    if not room:
        raise HTTPException(404, "Room not found")
    return room