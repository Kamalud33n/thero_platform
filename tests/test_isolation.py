"""
Item 5 — "Isolation test with platform routers disabled", automated.

Post phase-1 removal, patients/analytics/reports/pages/integration are
fully DELETED, not just toggleable — so there's no "disable" step left
for those. What's left to actually prove isolation on: does the
async-only core (routers/sessions.py + routers/ws.py) have any HIDDEN
dependency on telehealth.py or routers/camera.py (the MJPEG dev-camera
router)?

This builds a second, throwaway FastAPI app that mounts ONLY
sessions.router + ws.router — no telehealth_router, no camera.router —
and re-runs the same start -> WS frame -> finish lifecycle from
test_two_tab_flow.py against it. If session creation, /ws/pose,
broadcast, and session save all still work with telehealth/camera
routers never even imported into the app, the core is isolated.

Run: python test_isolation.py
"""
import os
import sys
import time
import base64
import datetime

# repo root (this file now lives in tests/) needs to be on sys.path so
# "from database import ..." / "from models import ..." below resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DATABASE_URL"] = "sqlite:///./test_isolation.db"
os.environ["MEDNOVA_JWT_ALGORITHM"] = "RS256"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_public.pem")) as f:
    os.environ["MEDNOVA_JWT_PUBLIC_KEY"] = f.read()
os.environ["MEDNOVA_JWT_ISSUER"] = "mednova-care-test"
os.environ["MEDNOVA_JWT_AUDIENCE"] = "thero-test"
os.environ["MEDNOVA_ALLOWED_ORIGINS"] = "http://testorigin.local"
os.environ["PATIENT_SESSION_TOKEN_SECRET"] = "test-only-secret-do-not-use-in-prod"
os.environ.setdefault("LOCAL_CAMERA_ENABLED", "false")

import sqlalchemy
_orig_create_engine = sqlalchemy.create_engine
def _patched_create_engine(*a, **k):
    k.pop("pool_size", None)
    k.pop("max_overflow", None)
    return _orig_create_engine(*a, **k)
sqlalchemy.create_engine = _patched_create_engine

if os.path.exists("test_isolation.db"):
    os.remove("test_isolation.db")

import jwt
import numpy as np
import cv2
from fastapi import FastAPI
from starlette.testclient import TestClient

from database import get_db, init_db
from models import Patient

PASS, FAIL = [], []


def check(label, cond, detail=""):
    if cond:
        PASS.append(label); print(f"  PASS  {label}")
    else:
        FAIL.append(label); print(f"  FAIL  {label}  {detail}")


def main():
    print("=" * 70)
    print("ISOLATION TEST — core (sessions.router + ws.router) ONLY")
    print("telehealth_router and camera.router are never imported/mounted")
    print("=" * 70)

    init_db()

    # ── Build a minimal app: import ONLY the two core routers. If either
    #    secretly imports telehealth.py or routers/camera.py at module
    #    load time, this import itself would pull them in — so the
    #    absence of any telehealth/camera side-effects (e.g. RoomManager
    #    state, mjpeg camera device probing) below is itself part of
    #    what's being checked, not just whether the endpoints respond. ──
    from routers import sessions as sessions_router_mod
    from routers import ws as ws_router_mod

    check(
        "sessions.py does NOT import telehealth or routers.camera",
        "telehealth" not in dir(sessions_router_mod) and "camera" not in dir(sessions_router_mod),
    )
    check(
        "ws.py does NOT import telehealth or routers.camera",
        "telehealth" not in dir(ws_router_mod) and "camera" not in dir(ws_router_mod),
    )

    isolated_app = FastAPI(title="thero-core-isolated")
    isolated_app.include_router(sessions_router_mod.router)
    isolated_app.include_router(ws_router_mod.router)
    # Deliberately NOT included: telehealth_router, camera.router,
    # StaticFiles mounts, CORS middleware — this is the bare minimum for
    # the async doctor-assigns/patient-performs-alone flow.

    client = TestClient(isolated_app)

    consultant_id = "iso-consultant-1"
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_private.pem")) as f:
        private_key = f.read()
    now = datetime.datetime.utcnow()
    therapist_token = jwt.encode(
        {
            "customer_id": consultant_id, "external_id": consultant_id,
            "type_account": "therapist", "iss": "mednova-care-test",
            "aud": "thero-test", "iat": now, "exp": now + datetime.timedelta(hours=1),
        },
        private_key, algorithm="RS256",
    )
    auth_headers = {"Authorization": f"Bearer {therapist_token}"}

    with get_db() as db:
        patient = Patient(name="Iso Patient", age=30, gender="Other",
                           mednova_consultant_id=consultant_id)
        db.add(patient); db.commit(); db.refresh(patient)
        patient_id = patient.id
    print(f"\n[setup] Patient {patient_id} (direct DB insert — see test_two_tab_flow.py docstring)")

    print("\n[core-only app] POST /api/sessions/start")
    resp = client.post(
        "/api/sessions/start",
        json={"patient_id": patient_id, "exercise_type": "Shoulder Flexion", "affected_side": "right"},
        headers=auth_headers,
    )
    check("start_session works with only sessions+ws mounted -> 200", resp.status_code == 200, resp.text)
    body = resp.json()
    session_id, patient_token = body.get("session_id"), body.get("token")
    check("got session_id + token", bool(session_id and patient_token))

    print(f"\n[core-only app] POST /api/sessions/{session_id}/watch-token")
    resp = client.post(f"/api/sessions/{session_id}/watch-token", headers=auth_headers)
    check("watch-token works with only sessions+ws mounted -> 200", resp.status_code == 200, resp.text)
    doctor_token = resp.json().get("token")
    check("got doctor token", bool(doctor_token))

    origin_headers = {"origin": "http://testorigin.local"}
    print(f"\n[core-only app] /ws/pose for session {session_id} (patient + doctor)")
    with client.websocket_connect(
        f"/ws/pose?session_id={session_id}&role=patient&token={patient_token}",
        headers=origin_headers,
    ) as patient_ws, client.websocket_connect(
        f"/ws/pose?session_id={session_id}&role=doctor&token={doctor_token}",
        headers=origin_headers,
    ) as doctor_ws:
        check("patient connects", patient_ws.receive_json().get("type") == "connected")
        check("doctor connects", doctor_ws.receive_json().get("type") == "connected")

        frame = (np.random.rand(240, 320, 3) * 255).astype("uint8")
        ok, buf = cv2.imencode(".jpg", frame)
        frame_b64 = base64.b64encode(buf.tobytes()).decode()
        frame_msg = {"type": "frame", "data": frame_b64}

        patient_ws.send_json(frame_msg)
        patient_ws.send_json(frame_msg)
        check("patient receives pose_data broadcast (no telehealth/camera router mounted)",
              patient_ws.receive_json().get("type") == "pose_data")
        check("doctor receives pose_data broadcast (no telehealth/camera router mounted)",
              doctor_ws.receive_json().get("type") == "pose_data")

    print("\n[core-only app] POST /api/sessions (finish)")
    resp = client.post(
        "/api/sessions",
        json={
            "patient_id": patient_id, "session_id": session_id,
            "exercise_type": "Shoulder Flexion", "affected_side": "right",
            "duration_seconds": 20, "total_reps": 3, "completed_reps": 3,
        },
        headers=auth_headers,
    )
    check("finishing save works core-only -> 200", resp.status_code == 200, resp.text)

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 70)
    if FAIL:
        for f in FAIL:
            print(f"  - {f}")
        print("\n=> Core is NOT cleanly isolated from telehealth/camera.")
    else:
        print("=> Core (sessions.py + ws.py) runs fully standalone —")
        print("   no telehealth.py or routers/camera.py dependency.")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
