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


def get_or_create_patient_by_external_id(
    db,
    external_id: str,
    name: str,
    mednova_consultant_id: str,
) -> Patient:
    """
    Bridge mode's patient step (routers/bridge.py POST /api/bridge/create-session,
    Nada's Task 1: "Auto-create patient record if external_id doesn't exist").

    Takes mednova_consultant_id as a plain string rather than a
    CurrentTherapist, because the bridge endpoint is HMAC-authenticated
    server-to-server (auth.verify_bridge_hmac) — there is no bridge JWT and
    therefore no CurrentTherapist to build in that request path. See the
    OPEN QUESTION note on auth.issue_bridge_doctor_token: as of this
    writing, Nada's documented bridge request body has no field carrying
    this value at all, so callers currently have nothing correct to pass
    here. Do not call this with a placeholder/empty value — that would
    silently attach a bridge-created patient to the wrong therapist (or an
    unowned one), defeating the isolation guarantee every other function in
    this module exists to enforce. Confirm the field name with Nada before
    wiring this into routers/bridge.py.

    Lookup is scoped by (external_id, mednova_consultant_id) together, not
    external_id alone — two different therapists could plausibly have two
    different patients that happen to share an external_id from MedNova's
    side (e.g. a data entry collision), and this must not cross-match them.

    Unlike every other write in this module, this one does NOT go through
    owned_patients_query()'s existing-patient-only assumption: it creates a
    new row on a miss rather than raising 404, since "doesn't exist yet" is
    the expected, common case here (Nada's spec), not an error.
    """
    if not mednova_consultant_id:
        raise HTTPException(500, "get_or_create_patient_by_external_id called without a mednova_consultant_id — refusing to create an unowned patient row")
    if not external_id:
        raise HTTPException(400, "patient_external_id is required")

    patient = (
        db.query(Patient)
        .filter(
            Patient.external_id == external_id,
            Patient.mednova_consultant_id == mednova_consultant_id,
        )
        .first()
    )
    if patient is not None:
        return patient

    # Minimal record — bridge mode's create-session payload only supplies
    # patient_name + patient_external_id, none of the clinical fields
    # (age/gender/diagnosis/etc.) that patients.py's own Add Patient form
    # requires. age/gender are NOT NULL on the Patient model (see
    # models.py), so this can't literally omit them — flagged here as an
    # open item for Nada: confirm what age/gender should be for a
    # bridge-created patient (placeholder pending sync from MedNova's own
    # patient record, vs. Laravel sending real values in a future payload
    # revision) rather than this silently defaulting to something
    # clinically meaningless.
    patient = Patient(
        name=name,
        external_id=external_id,
        mednova_consultant_id=mednova_consultant_id,
        age=0,       # placeholder — see docstring note above, confirm with Nada
        gender="",   # placeholder — see docstring note above, confirm with Nada
        is_active=True,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient