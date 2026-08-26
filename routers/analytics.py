import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from auth import CurrentTherapist, get_current_therapist
from database import get_db
from models import SessionModel
from repositories.patient_repo import assert_owns_patient, list_owned_patient_ids
from services.helpers import calculate_recovery_score, session_summary

router = APIRouter()


@router.get("/api/analytics")
async def analytics(
    patient_id: Optional[str] = None,
    therapist: CurrentTherapist = Depends(get_current_therapist),
):
    with get_db() as db:
        q = db.query(SessionModel)
        if patient_id:
            # Ownership check — same repository function every other route
            # uses, so this can't drift from patients.py/sessions.py.
            assert_owns_patient(db, patient_id, therapist)
            q = q.filter(SessionModel.patient_id == patient_id)
        else:
            # No patient_id given — scope to only THIS therapist's patients,
            # never return sessions across all therapists.
            own_patient_ids = list_owned_patient_ids(db, therapist)
            q = q.filter(SessionModel.patient_id.in_(own_patient_ids))
        sessions = q.order_by(SessionModel.start_time).all()

        today      = datetime.date.today()
        yesterday  = today - datetime.timedelta(days=1)

        def _day_range(d):
            return (
                datetime.datetime.combine(d, datetime.time.min),
                datetime.datetime.combine(d, datetime.time.max),
            )

        def _bucket(d):
            lo, hi = _day_range(d)
            return [s for s in sessions if lo <= s.start_time <= hi]

        def _summary(bucket):
            n = len(bucket)
            avg = lambda attr: round(sum(getattr(s, attr) or 0 for s in bucket) / n, 1) if n else 0.0
            return {
                "sessions":       n,
                "total_reps":     sum(s.completed_reps or 0 for s in bucket),
                "avg_accuracy":   avg("accuracy_percentage"),
                "avg_rom":        avg("average_rom"),
                "avg_stability":  avg("stability_score"),
                "avg_balance":    avg("balance_score"),
                "avg_smoothness": avg("movement_smoothness"),
                "recovery_score": round(calculate_recovery_score(bucket), 1),
                "incorrect_movements": sum(s.incorrect_movements or 0 for s in bucket),
            }

        today_bucket     = _bucket(today)
        yesterday_bucket = _bucket(yesterday)
        today_sum        = _summary(today_bucket)
        yesterday_sum    = _summary(yesterday_bucket)

        def _delta(key):
            t, y = today_sum[key], yesterday_sum[key]
            if y == 0:
                return None if t == 0 else 100.0
            return round(((t - y) / y) * 100, 1)

        deltas = {
            key: _delta(key)
            for key in ("sessions", "total_reps", "avg_accuracy", "avg_rom",
                        "avg_stability", "avg_balance", "recovery_score")
        }

        # Last 7 days trend, for context below the today-vs-yesterday cards
        trend = []
        for i in range(6, -1, -1):
            d = today - datetime.timedelta(days=i)
            b = _bucket(d)
            s = _summary(b)
            trend.append({"date": d.isoformat(), **s})

        return JSONResponse({
            "today":     {"date": today.isoformat(),     **today_sum},
            "yesterday": {"date": yesterday.isoformat(), **yesterday_sum},
            "deltas":    deltas,
            "trend":     trend,
            "today_sessions":     [session_summary(s) for s in today_bucket],
            "yesterday_sessions": [session_summary(s) for s in yesterday_bucket],
        })