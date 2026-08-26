"""
Single source of truth for "which patients does this therapist own".

Per the integration plan (thero x MedNova Care Integration Plan, step 6):
isolation must be enforced at the repository/service layer, not just the
route level, so no individual endpoint can forget to filter by therapist.

Before this file existed, every router (patients.py, sessions.py,
analytics.py, reports.py, telehealth.py) repeated its own inline
`Patient.mednova_consultant_id == therapist.mednova_consultant_id` filter.
That works today, but it means the isolation guarantee lived in N copies
instead of one — a new route that forgets the filter would silently leak
another therapist's patient data with no error anywhere.

Rule: no route or service should query the Patient table directly by
mednova_consultant_id. Always go through one of these functions instead.
"""
from typing import List

from fastapi import HTTPException

from auth import CurrentTherapist
from models import Patient


def owned_patients_query(db, therapist: CurrentTherapist):
    """
    Base query scoped to this therapist's patients only. Chain further
    .filter() / .order_by() / .with_entities() on the result for
    route-specific needs (search, active-only, etc.) — never start a
    fresh `db.query(Patient)` in a route.
    """
    return db.query(Patient).filter(
        Patient.mednova_consultant_id == therapist.mednova_consultant_id
    )


def list_owned_patients(
    db,
    therapist: CurrentTherapist,
    search: str = "",
    include_inactive: bool = False,
) -> List[Patient]:
    """Full patient list for the patients page — search + active filter."""
    q = owned_patients_query(db, therapist)
    if not include_inactive:
        q = q.filter(Patient.is_active == True)
    if search:
        q = q.filter(
            Patient.name.contains(search)
            | Patient.id.contains(search)
            | Patient.phone.contains(search)
        )
    return q.order_by(Patient.date_created.desc()).all()


def list_owned_patient_ids(db, therapist: CurrentTherapist) -> List[str]:
    """Cheap id-only list — used to scope session/analytics queries when
    no specific patient_id was given (e.g. GET /api/analytics with no
    patient_id filter)."""
    return [
        pid for (pid,) in owned_patients_query(db, therapist)
        .with_entities(Patient.id)
        .all()
    ]


def get_owned_patient(db, patient_id: str, therapist: CurrentTherapist) -> Patient:
    """
    Fetch one patient, scoped to this therapist. Raises 404 — not 403 —
    whether the patient doesn't exist at all OR belongs to a different
    therapist. That's deliberate: the two cases must be indistinguishable
    from the response, so this endpoint can't be used to probe which
    patient IDs exist under other therapists.
    """
    patient = owned_patients_query(db, therapist).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")
    return patient


def assert_owns_patient(db, patient_id: str, therapist: CurrentTherapist) -> None:
    """
    Existence-only ownership check, for routes that just need to validate
    before doing something else (saving a session, generating a report,
    scheduling a telehealth room) without loading the full Patient row.
    """
    exists = (
        owned_patients_query(db, therapist)
        .filter(Patient.id == patient_id)
        .with_entities(Patient.id)
        .first()
    )
    if not exists:
        raise HTTPException(404, "Patient not found")