import os
import asyncio
import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from auth import CurrentTherapist, get_current_therapist
from database import get_db
from models import SessionModel, Report
from repositories.patient_repo import assert_owns_patient
from services.report_builder import build_report_sync

router = APIRouter()


@router.get("/api/reports/{patient_id}")
async def generate_report(
    patient_id: str,
    report_type: str = "weekly",
    start_date: str | None = None,
    end_date: str | None = None,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    # Pre-check so the user gets a clean JSON error instead of a raw 404
    # blank tab when the patient doesn't exist / has no sessions yet.
    # Also doubles as the ownership check — the patient must belong to the
    # requesting therapist before a report can be generated for them.
    with get_db() as db:
        assert_owns_patient(db, patient_id, therapist)
        has_sessions = (
            db.query(SessionModel.id)
            .filter(SessionModel.patient_id == patient_id)
            .first()
            is not None
        )
        if not has_sessions:
            raise HTTPException(404, "No sessions found for this patient yet — complete a session first")

    # Custom date-range report needs both dates, validated + parsed here so
    # build_report_sync (which runs in a thread executor) only ever sees
    # clean datetime objects, not raw query strings.
    range_start = range_end = None
    if report_type == "custom":
        if not start_date or not end_date:
            raise HTTPException(400, "start_date and end_date are required for a custom report")
        try:
            range_start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            range_end = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(
                hours=23, minutes=59, seconds=59
            )
        except ValueError:
            raise HTTPException(400, "Dates must be in YYYY-MM-DD format")
        if range_start > range_end:
            raise HTTPException(400, "start_date must be before end_date")

    loop = asyncio.get_event_loop()
    filepath = await loop.run_in_executor(
        None, build_report_sync, patient_id, report_type, range_start, range_end
    )

    fname_suffix = (
        f"{start_date}_to_{end_date}" if report_type == "custom" else report_type
    )
    return FileResponse(
        filepath,
        media_type="application/pdf",
        filename=f"report_{patient_id}_{fname_suffix}.pdf",
    )


@router.get("/api/reports/history/{patient_id}")
async def report_history(
    patient_id: str,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    """List previously generated reports for a patient, newest first."""
    with get_db() as db:
        assert_owns_patient(db, patient_id, therapist)
        rows = (
            db.query(Report)
            .filter(Report.patient_id == patient_id)
            .order_by(Report.generated_date.desc())
            .all()
        )
        out = []
        for r in rows:
            out.append({
                "id":             r.id,
                "report_type":    r.report_type,
                "generated_date": r.generated_date.isoformat(),
                "available":      bool(r.file_path and os.path.exists(r.file_path)),
            })
        return JSONResponse(out)


@router.get("/api/reports/file/{report_id}")
async def download_report_file(
    report_id: int,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    """Download a specific, already-generated report by its Report row id."""
    with get_db() as db:
        r = db.query(Report).filter(Report.id == report_id).first()
        if not r:
            raise HTTPException(404, "Report record not found")
        # Ownership check via the report's patient — a report id alone
        # shouldn't be enough to pull another therapist's file. Same
        # repository function every other route uses, so a report id
        # can't be used to probe patient ownership across therapists
        # (404 either way, just like get_owned_patient).
        try:
            assert_owns_patient(db, r.patient_id, therapist)
        except HTTPException:
            raise HTTPException(404, "Report record not found")
        if not r.file_path or not os.path.exists(r.file_path):
            raise HTTPException(410, "This report file is no longer available — please regenerate it")
        return FileResponse(
            r.file_path,
            media_type="application/pdf",
            filename=f"report_{r.patient_id}_{r.report_type}.pdf",
        )