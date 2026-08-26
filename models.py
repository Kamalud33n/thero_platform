import uuid

from sqlalchemy import (
    Column, String, Integer, Float,
    DateTime, Text, Boolean, ForeignKey, JSON, LargeBinary
)
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


# ID generators 
def new_patient_id() -> str:
    return f"PAT-{uuid.uuid4().hex[:8].upper()}"


def new_session_id() -> str:
    return f"SES-{uuid.uuid4().hex[:8].upper()}"


def new_room_id() -> str:
    return f"ROOM-{uuid.uuid4().hex[:8].upper()}"


# Item 24: finalized contract for SessionModel.end_reason, per Nada's
# case-monitoring spec (2026-08-23). Seven values:
#   "completed"             — patient finished the full planned rep/set count.
#   "stopped_by_patient"    — patient voluntarily ended the session early
#                              (fatigue, ran out of time, etc.) — NOT pain,
#                              that's its own value below. Renamed from the
#                              old catch-all "stopped" to make room for
#                              "stopped_by_therapist" as a distinct value.
#   "pain"                  — patient pressed the pain button.
#   "technical_error"       — client-side/system malfunction ended the
#                              session (not the patient's doing — clinically
#                              this must NOT be read as "exercise was wrong
#                              for this patient", per Nada).
#   "disconnected"          — WS dropped and never reconnected (crash,
#                              network loss, tab closed) — see
#                              routers/ws.py._abandon_if_unfinished.
#   "timeout"                — session expired without an explicit end.
#   "stopped_by_therapist"  — therapist ended the session from the
#                              supervised/telehealth room (see telehealth.py
#                              close-room flow).
# Callers should validate incoming values against this set (see
# routers/sessions.py, telehealth.py) rather than trusting an arbitrary
# client-sent string in a column that reports/analytics will eventually
# group by.
VALID_END_REASONS = (
    "completed",
    "stopped_by_patient",
    "pain",
    "technical_error",
    "disconnected",
    "timeout",
    "stopped_by_therapist",
)


# Models 
class Patient(Base):
    __tablename__ = "patients"
    id                 = Column(String(50), primary_key=True, default=new_patient_id)
    name               = Column(String(100), nullable=False)
    age                = Column(Integer, nullable=False)
    gender             = Column(String(10), nullable=False)
    weight             = Column(Float, nullable=True)
    height             = Column(Float, nullable=True)
    diagnosis          = Column(String(200), nullable=True)
    affected_body_part = Column(String(100), nullable=True)
    phone              = Column(String(20), nullable=True)
    email              = Column(String(100), nullable=True)
    external_id        = Column(String(100), nullable=True, index=True)  # ID from an external system (e.g. MedNova); nullable since not every patient will have one
    mednova_consultant_id = Column(String(100), nullable=True, index=True)  # owning therapist — populated from the verified JWT (auth.get_current_therapist), never typed by hand
    medical_history    = Column(Text, nullable=True)
    previous_injury    = Column(Text, nullable=True)
    current_treatment  = Column(Text, nullable=True)
    exercise_plan      = Column(Text, nullable=True)
    photo              = Column(LargeBinary().with_variant(LONGBLOB, "mysql"), nullable=True)
    date_created       = Column(DateTime, default=func.now())
    is_active          = Column(Boolean, default=True)
    sessions = relationship("SessionModel", back_populates="patient", cascade="all, delete-orphan")


class SessionModel(Base):
    __tablename__ = "sessions"
    id                  = Column(String(50), primary_key=True, default=new_session_id)
    patient_id          = Column(String(50), ForeignKey("patients.id"), nullable=False)
    exercise_type       = Column(String(100), nullable=False)
    # "left" | "right" | "both" — which side was treated as primary for rep
    # counting/scoring during this session. Per Nada: "both sides scored
    # and stored, with affected_side treated as primary" — the l_/r_
    # angles for BOTH sides still live in JointAngle rows regardless.
    affected_side       = Column(String(10), nullable=True, default="both")
    # "in_progress" | "completed" | "abandoned"
    #   in_progress — row created up front by POST /api/sessions/start,
    #                 before the patient's /ws/pose token is even issued
    #                 (see routers/sessions.py, routers/ws.py). This is
    #                 what makes a session_id real: it always points at an
    #                 actual DB row + patient from the moment it exists,
    #                 instead of being an arbitrary client-chosen string.
    #   completed   — the normal finishing POST /api/sessions arrived and
    #                 updated this row with the full client-computed
    #                 metrics (joint_angles/exercise_results included).
    #   abandoned   — the patient's /ws/pose WebSocket disconnected (crash,
    #                 network loss, tab closed) before that finishing POST
    #                 ever arrived. routers/ws.py's disconnect handler
    #                 auto-saves whatever the live pipeline captured so far
    #                 instead of silently losing the session — see the
    #                 disconnect block there for exactly what gets written.
    # Legacy rows created before this column existed (or by any caller
    # still using the old direct-insert flow) have status=NULL — treated
    # the same as "completed" everywhere this is read, since that was the
    # only kind of row that used to exist.
    status               = Column(String(20), nullable=True, default="in_progress")
    # Item 24: WHY a completed/stopped session ended, orthogonal to
    # `status` above (status = did the row get a proper finishing POST at
    # all; end_reason = what the patient/therapist reported as the cause
    # once it did). One of the 7 VALID_END_REASONS above — see that
    # constant's docstring for the full breakdown. Clinically each value
    # is distinct and reportable (per Nada): pain vs technical_error in
    # particular must never be conflated, since one implies the exercise
    # plan needs revision and the other doesn't.
    # NULL for rows that never got an explicit reason: legacy rows saved
    # before this column existed. NULL is treated like "completed" for
    # display purposes anywhere this is read, same convention as the
    # legacy status=NULL handling above.
    #
    # String(20) is now a tight fit — "stopped_by_therapist" is exactly
    # 20 chars. Do not add a longer value without widening this column.
    end_reason           = Column(String(20), nullable=True, default=None)
    # Missing from this model even though routers/sessions.py, routers/ws.py
    # (abandon-on-disconnect), telehealth.py, and app.py's demo seed all
    # read/write these three unconditionally — found via the two-tab
    # engine test (item 4), which failed at POST /api/sessions/start with
    # "'start_time' is an invalid keyword argument for SessionModel"
    # before these were added. Restored to match how every caller already
    # uses them: start_time set at session creation, end_time/duration on
    # finish (or on abandon-on-disconnect).
    start_time           = Column(DateTime, default=func.now())
    end_time             = Column(DateTime, nullable=True)
    duration_seconds     = Column(Integer, default=0)
    total_reps          = Column(Integer, default=0)
    completed_reps      = Column(Integer, default=0)
    accuracy_percentage = Column(Float, default=0.0)
    average_rom         = Column(Float, default=0.0)
    incorrect_movements = Column(Integer, default=0)
    stability_score     = Column(Float, default=0.0)
    balance_score       = Column(Float, default=0.0)
    movement_smoothness = Column(Float, default=0.0)
    fatigue_estimation  = Column(Float, default=0.0)
    recovery_score      = Column(Float, default=0.0)
    session_data        = Column(JSON, nullable=True)
    patient          = relationship("Patient", back_populates="sessions")
    joint_angles     = relationship("JointAngle", back_populates="session", cascade="all, delete-orphan")
    exercise_results = relationship("ExerciseResult", back_populates="session", cascade="all, delete-orphan")


class JointAngle(Base):
    __tablename__ = "joint_angles"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(String(50), ForeignKey("sessions.id"), nullable=False)
    timestamp    = Column(DateTime, default=func.now())
    joint_name   = Column(String(50), nullable=False)
    angle_value  = Column(Float, nullable=False)
    target_angle = Column(Float, nullable=True)
    deviation    = Column(Float, nullable=True)
    is_correct   = Column(Boolean, default=True)
    session = relationship("SessionModel", back_populates="joint_angles")


class ExerciseResult(Base):
    __tablename__ = "exercise_results"
    id                 = Column(Integer, primary_key=True, autoincrement=True)
    session_id         = Column(String(50), ForeignKey("sessions.id"), nullable=False)
    exercise_name      = Column(String(100), nullable=False)
    repetition_number  = Column(Integer, nullable=False)
    accuracy           = Column(Float, default=0.0)
    rom_achieved       = Column(Float, default=0.0)
    speed              = Column(Float, default=0.0)
    hold_duration      = Column(Float, default=0.0)
    compensation_score = Column(Float, default=0.0)
    is_completed       = Column(Boolean, default=False)
    feedback           = Column(Text, nullable=True)
    timestamp          = Column(DateTime, default=func.now())
    session = relationship("SessionModel", back_populates="exercise_results")


class Report(Base):
    __tablename__ = "reports"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    patient_id     = Column(String(50), ForeignKey("patients.id"), nullable=False)
    report_type    = Column(String(50), nullable=False)
    generated_date = Column(DateTime, default=func.now())
    file_path      = Column(String(200), nullable=True)
    report_data    = Column(JSON, nullable=True)


class Setting(Base):
    __tablename__ = "settings"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    key         = Column(String(50), unique=True, nullable=False)
    value       = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    updated_at  = Column(DateTime, default=func.now(), onupdate=func.now())


class History(Base):
    __tablename__ = "history"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), ForeignKey("patients.id"), nullable=False)
    action     = Column(String(100), nullable=False)
    details    = Column(Text, nullable=True)
    timestamp  = Column(DateTime, default=func.now())


class TelehealthRoom(Base):
    """
    A scheduled session room — covers BOTH session modes on the /session
    page (clinic/local-camera mode removed):

      mode = "remote"         doctor + patient live together, pose data
                               broadcast to both sides (telehealth.py
                               RoomManager, unchanged behaviour).
      mode = "self_training"  doctor schedules it, patient opens the link
                               ALONE later, camera + pose pipeline runs for
                               that patient only, metrics auto-save to the
                               patient's record exactly like a normal
                               session — no doctor socket involved.

    Both modes are scheduled: scheduled_at is the appointment time, and the
    link is only valid until expires_at = scheduled_at + 2h (see
    ROOM_LINK_VALID_HOURS in telehealth.py). This replaces "create whenever,
    open-ended link" behaviour.
    """
    __tablename__ = "telehealth_rooms"
    id            = Column(String(50), primary_key=True, default=new_room_id)
    token         = Column(String(64), unique=True, nullable=False, index=True)
    patient_id    = Column(String(50), ForeignKey("patients.id"), nullable=False)
    mednova_consultant_id = Column(String(100), nullable=True, index=True)  # owning therapist — from the verified JWT, replaces the old free-text doctor_name
    exercise_type = Column(String(100), nullable=True)
    affected_side = Column(String(10), nullable=True, default="both")  # "left" | "right" | "both" — set by the doctor when scheduling, carried into the session's own affected_side on save
    # Target range-of-motion the patient should hit for this scheduled
    # exercise. Required for mode="self_training" (no live doctor socket
    # to set it later — see telehealth.py's create_room), optional for
    # mode="remote" (doctor can still set it live once both sides connect).
    target_rom    = Column(Float, nullable=True)
    mode          = Column(String(20), nullable=False, default="remote")  # "remote" | "self_training"
    status        = Column(String(20), default="pending")  # pending -> live -> closed
    session_id    = Column(String(50), ForeignKey("sessions.id"), nullable=True)
    scheduled_at  = Column(DateTime, nullable=False)   # appointment time set by the doctor
    expires_at    = Column(DateTime, nullable=False)   # scheduled_at + 2h — link stops working after this, in any status
    created_at    = Column(DateTime, default=func.now())
    started_at    = Column(DateTime, nullable=True)
    closed_at     = Column(DateTime, nullable=True)
    patient = relationship("Patient")


class UsedPatientToken(Base):
    """
    jti replay-protection ledger for the patient session-scoped /ws/pose
    token (auth.issue_patient_session_token / consume_patient_session_jti).

    This token is single-use by design: it's minted by POST
    /api/sessions/start right before the patient's page opens the
    WebSocket, and is meant to be redeemed by exactly one /ws/pose
    connection. The instant routers/ws.py accepts a patient connection
    with a given jti, it inserts a row here in the SAME transaction as the
    validation — so a captured/replayed token (e.g. copy-pasted join link
    reused, or a MITM'd token replayed) can never open a second
    connection: the INSERT on an already-used jti hits the primary-key
    constraint and routers/ws.py treats that as "already used" -> reject.

    expires_at mirrors the token's own `exp` claim so old rows are safe to
    prune later (no cleanup job is required for correctness — an expired,
    already-rejected-at-decode-time token doesn't need its jti checked
    here at all — this column exists purely so a future housekeeping job
    has something to filter on instead of the table growing forever).
    """
    __tablename__ = "used_patient_tokens"
    jti        = Column(String(64), primary_key=True)
    session_id = Column(String(50), nullable=False, index=True)
    used_at    = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=False)