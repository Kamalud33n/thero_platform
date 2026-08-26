"""
Patient CRUD — backs templates/patients.html.

This router didn't exist yet in the repo (patients/analytics/reports/session
UI had been stripped out earlier, see app.py's root() comment), but the
frontend, patient_repo.py's own docstring ("...every router (patients.py,
sessions.py, analytics.py, reports.py, telehealth.py)...") and
services/report_builder.py / routers/analytics.py all assume it exists.
Added here, scoped through repositories/patient_repo.py like every other
router — never query Patient directly by mednova_consultant_id.

Note: patients.html's form also posts `doctor_name` and `therapist_name`
fields, but the Patient model (models.py) has no matching columns. They're
accepted here (so the form submit doesn't 422) but silently dropped rather
than persisted — add columns + a migration if these need to be stored.
"""
import base64
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from auth import CurrentTherapist, get_current_therapist
from database import get_db
from models import Patient, SessionModel
from repositories.patient_repo import (
    assert_owns_patient,
    get_owned_patient,
    list_owned_patients,
)

router = APIRouter()


def _photo_b64(patient: Patient) -> Optional[str]:
    if not patient.photo:
        return None
    return base64.b64encode(patient.photo).decode("ascii")


def _session_stats(db, patient_id: str) -> tuple[int, float]:
    sessions = (
        db.query(SessionModel.accuracy_percentage)
        .filter(SessionModel.patient_id == patient_id)
        .all()
    )
    n = len(sessions)
    if n == 0:
        return 0, 0.0
    avg = round(sum((s.accuracy_percentage or 0.0) for s in sessions) / n, 1)
    return n, avg


def _patient_card(db, p: Patient) -> dict:
    sessions_count, avg_accuracy = _session_stats(db, p.id)
    return {
        "id": p.id,
        "name": p.name,
        "age": p.age,
        "gender": p.gender,
        "weight": p.weight,
        "height": p.height,
        "diagnosis": p.diagnosis,
        "affected_body_part": p.affected_body_part,
        "phone": p.phone,
        "email": p.email,
        "is_active": p.is_active,
        "photo": _photo_b64(p),
        "date_created": p.date_created.isoformat() if p.date_created else None,
        "sessions_count": sessions_count,
        "avg_accuracy": avg_accuracy,
    }


def _patient_detail(db, p: Patient) -> dict:
    out = _patient_card(db, p)
    out.update({
        "medical_history": p.medical_history,
        "previous_injury": p.previous_injury,
        "current_treatment": p.current_treatment,
        "exercise_plan": p.exercise_plan,
        "external_id": p.external_id,
        # Not real columns on Patient — see module docstring. Included so
        # the edit-modal prefill code in patients.html doesn't choke on a
        # missing key; always empty until those columns exist.
        "doctor_name": None,
        "therapist_name": None,
    })
    sessions = (
        db.query(SessionModel)
        .filter(SessionModel.patient_id == p.id)
        .order_by(SessionModel.start_time.desc())
        .limit(10)
        .all()
    )
    out["sessions"] = [
        {
            "id": s.id,
            "exercise_type": s.exercise_type,
            "start_time": s.start_time.isoformat() if s.start_time else None,
            "accuracy_percentage": s.accuracy_percentage,
        }
        for s in sessions
    ]
    return out


@router.get("/api/patients")
async def list_patients(
    search: str = "",
    include_inactive: bool = False,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    with get_db() as db:
        patients = list_owned_patients(db, therapist, search=search, include_inactive=include_inactive)
        return JSONResponse([_patient_card(db, p) for p in patients])


@router.get("/api/patients/{patient_id}")
async def get_patient(
    patient_id: str,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    with get_db() as db:
        p = get_owned_patient(db, patient_id, therapist)
        return JSONResponse(_patient_detail(db, p))


async def _read_photo(photo: Optional[UploadFile]) -> Optional[bytes]:
    if photo is None or not photo.filename:
        return None
    data = await photo.read()
    return data or None


@router.post("/api/patients")
async def create_patient(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    weight: Optional[float] = Form(None),
    height: Optional[float] = Form(None),
    diagnosis: Optional[str] = Form(None),
    affected_body_part: Optional[str] = Form(None),
    doctor_name: Optional[str] = Form(None),
    therapist_name: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    medical_history: Optional[str] = Form(None),
    previous_injury: Optional[str] = Form(None),
    current_treatment: Optional[str] = Form(None),
    exercise_plan: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    photo_bytes = await _read_photo(photo)
    with get_db() as db:
        p = Patient(
            name=name, age=age, gender=gender, weight=weight, height=height,
            diagnosis=diagnosis, affected_body_part=affected_body_part,
            phone=phone, email=email, medical_history=medical_history,
            previous_injury=previous_injury, current_treatment=current_treatment,
            exercise_plan=exercise_plan, photo=photo_bytes,
            mednova_consultant_id=therapist.mednova_consultant_id,
            is_active=True,
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        return JSONResponse(_patient_detail(db, p))


@router.put("/api/patients/{patient_id}")
async def update_patient(
    patient_id: str,
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    weight: Optional[float] = Form(None),
    height: Optional[float] = Form(None),
    diagnosis: Optional[str] = Form(None),
    affected_body_part: Optional[str] = Form(None),
    doctor_name: Optional[str] = Form(None),
    therapist_name: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    medical_history: Optional[str] = Form(None),
    previous_injury: Optional[str] = Form(None),
    current_treatment: Optional[str] = Form(None),
    exercise_plan: Optional[str] = Form(None),
    is_active: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    photo_bytes = await _read_photo(photo)
    with get_db() as db:
        p = get_owned_patient(db, patient_id, therapist)
        p.name, p.age, p.gender = name, age, gender
        p.weight, p.height = weight, height
        p.diagnosis, p.affected_body_part = diagnosis, affected_body_part
        p.phone, p.email = phone, email
        p.medical_history = medical_history
        p.previous_injury = previous_injury
        p.current_treatment = current_treatment
        p.exercise_plan = exercise_plan
        if photo_bytes is not None:
            p.photo = photo_bytes
        if is_active is not None:
            p.is_active = is_active.lower() == "true"
        db.commit()
        db.refresh(p)
        return JSONResponse(_patient_detail(db, p))


@router.delete("/api/patients/{patient_id}")
async def deactivate_patient(
    patient_id: str,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    """Soft-delete only — records are kept, patient is just hidden from the
    active list (matches the confirm-dialog copy in patients.html)."""
    with get_db() as db:
        assert_owns_patient(db, patient_id, therapist)
        p = get_owned_patient(db, patient_id, therapist)
        p.is_active = False
        db.commit()
        return JSONResponse({"ok": True, "id": patient_id, "is_active": False})
