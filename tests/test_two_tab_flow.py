"""
Item 4 — "Actual two-tab engine test", automated.

Simulates the real browser flow (therapist tab + patient tab) end-to-end
against the live FastAPI app, using TestClient's WebSocket support to
stand in for two browser tabs:

  Tab A (therapist, HTTP only):
    1. POST /api/sessions/start  -> creates the SessionModel row + mints
       a single-use patient session token.

  Tab B (patient, WS):
    2. Connects to /ws/pose?role=patient with that token.
    3. Sends a "frame" message (a real JPEG).

  Tab C (doctor, WS — e.g. the therapist watching live):
    4. Connects to /ws/pose?role=doctor for the SAME session_id.
    5. Asserts it receives the broadcast pose_data that patient's frame
       produced — this is the actual cross-tab behaviour being tested.

  Tab A again:
    6. POST /api/sessions (finish/save) -> asserts the row flips to
       "completed" with the right patient_id.

KNOWN GAP this test has to work around (tracked as pending item #10,
"session creation without patient DB row"): now that patient
management (#6) and the Laravel patient-sync webhook (#8) are both
removed, there is currently NO API path left to create a Patient row —
assert_owns_patient() in sessions.py still requires one to exist. This
test creates that row directly via the ORM, the same way the old
integration.py webhook or patients.py POST used to. That's a real
product gap, not just a test-harness detail — flagging it again here
because this test is where it becomes visible end-to-end.

Run: python test_two_tab_flow.py
"""
import os
import sys
import time
import base64
import datetime

# ── env, BEFORE importing app/auth/database ────────────────────────────
os.environ["DATABASE_URL"] = "sqlite:///./test_e2e.db"
os.environ["MEDNOVA_JWT_ALGORITHM"] = "RS256"
with open("test_public.pem") as f:
    os.environ["MEDNOVA_JWT_PUBLIC_KEY"] = f.read()
os.environ["MEDNOVA_JWT_ISSUER"] = "mednova-care-test"
os.environ["MEDNOVA_JWT_AUDIENCE"] = "thero-test"
os.environ["MEDNOVA_ALLOWED_ORIGINS"] = "http://testorigin.local"
os.environ["PATIENT_SESSION_TOKEN_SECRET"] = "test-only-secret-do-not-use-in-prod"
os.environ.setdefault("LOCAL_CAMERA_ENABLED", "false")

# sqlite branch's pool_size/max_overflow kwargs aren't valid for the
# default sqlite pool class — patch around it the same way the phase-1
# import smoke test did, purely so this script can run standalone.
import sqlalchemy
_orig_create_engine = sqlalchemy.create_engine
def _patched_create_engine(*a, **k):
    k.pop("pool_size", None)
    k.pop("max_overflow", None)
    return _orig_create_engine(*a, **k)
sqlalchemy.create_engine = _patched_create_engine

if os.path.exists("test_e2e.db"):
    os.remove("test_e2e.db")

import jwt
import numpy as np
import cv2
from starlette.testclient import TestClient

import app as app_module
from database import get_db
from models import Patient

client = TestClient(app_module.app)

PASS = []
FAIL = []


def check(label, cond, detail=""):
    if cond:
        PASS.append(label)
        print(f"  PASS  {label}")
    else:
        FAIL.append(label)
        print(f"  FAIL  {label}  {detail}")


def make_therapist_token(customer_id="1"):
    with open("test_private.pem") as f:
        private_key = f.read()
    now = datetime.datetime.utcnow()
    payload = {
        "customer_id": customer_id,
        "external_id": customer_id,
        "type_account": "therapist",
        "iss": "mednova-care-test",
        "aud": "thero-test",
        "iat": now,
        "exp": now + datetime.timedelta(hours=1),
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def make_test_jpeg_b64():
    # Doesn't need a real person in it — this test is checking the WS
    # wiring/broadcast path, not MediaPipe detection accuracy.
    frame = (np.random.rand(240, 320, 3) * 255).astype("uint8")
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def main():
    print("=" * 70)
    print("TWO-TAB ENGINE TEST (automated stand-in for therapist + patient tabs)")
    print("=" * 70)

    consultant_id = "test-consultant-1"
    therapist_token = make_therapist_token(customer_id=consultant_id)
    auth_headers = {"Authorization": f"Bearer {therapist_token}"}

    # ── Setup: create the Patient row directly (see module docstring —
    #    there's no API for this anymore post item #6/#8 removal) ───────
    with get_db() as db:
        patient = Patient(
            name="Test Patient", age=40, gender="Other",
            mednova_consultant_id=consultant_id,
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)
        patient_id = patient.id
    print(f"\n[setup] created Patient row {patient_id} for therapist {consultant_id}")

    # ── Tab A (therapist): POST /api/sessions/start ─────────────────────
    print("\n[Tab A / therapist] POST /api/sessions/start")
    resp = client.post(
        "/api/sessions/start",
        json={"patient_id": patient_id, "exercise_type": "Knee Flexion", "affected_side": "left"},
        headers=auth_headers,
    )
    check("start_session -> 200", resp.status_code == 200, resp.text)
    body = resp.json()
    session_id = body.get("session_id")
    patient_token = body.get("token")
    check("start_session returns session_id", bool(session_id))
    check("start_session returns patient token", bool(patient_token))

    # ── Tab A (therapist): POST /api/sessions/{id}/watch-token ──────────
    # New in this run — doctor connections now require a signed watch
    # token (item: "doctor connection has no signed role claim"), minted
    # by this endpoint after re-confirming the therapist owns the patient.
    print(f"\n[Tab A / therapist] POST /api/sessions/{session_id}/watch-token")
    resp = client.post(f"/api/sessions/{session_id}/watch-token", headers=auth_headers)
    check("watch-token -> 200", resp.status_code == 200, resp.text)
    doctor_token = resp.json().get("token")
    check("watch-token returns a token", bool(doctor_token))

    # ── Tab B (patient) + Tab C (doctor) connect concurrently ───────────
    origin_headers = {"origin": "http://testorigin.local"}
    print(f"\n[Tab B / patient] connecting /ws/pose?session_id={session_id}&role=patient")
    print(f"[Tab C / doctor]  connecting /ws/pose?session_id={session_id}&role=doctor")

    with client.websocket_connect(
        f"/ws/pose?session_id={session_id}&role=patient&token={patient_token}",
        headers=origin_headers,
    ) as patient_ws, client.websocket_connect(
        f"/ws/pose?session_id={session_id}&role=doctor&token={doctor_token}",
        headers=origin_headers,
    ) as doctor_ws:

        patient_ack = patient_ws.receive_json()
        check("patient tab gets 'connected' ack", patient_ack.get("type") == "connected", patient_ack)
        doctor_ack = doctor_ws.receive_json()
        check("doctor tab gets 'connected' ack", doctor_ack.get("type") == "connected", doctor_ack)

        # frame-skip logic processes only every 2nd frame — send 2 so one
        # actually gets processed and broadcast.
        frame_msg = {"type": "frame", "data": make_test_jpeg_b64()}
        print("\n[Tab B / patient] sending 2 frames (frame-skip processes every 2nd)")
        patient_ws.send_json(frame_msg)
        patient_ws.send_json(frame_msg)

        # The patient's own client also receives the broadcast (renders
        # its own overlay) — same as the doctor tab.
        patient_broadcast = patient_ws.receive_json()
        check(
            "patient tab receives its own pose_data broadcast",
            patient_broadcast.get("type") == "pose_data",
            patient_broadcast,
        )

        print("[Tab C / doctor] waiting for broadcast pose_data from patient's frame")
        doctor_broadcast = doctor_ws.receive_json()
        check(
            "doctor tab receives pose_data broadcast from patient tab (cross-tab relay)",
            doctor_broadcast.get("type") == "pose_data",
            doctor_broadcast,
        )
        check(
            "broadcast payload has pose_data key",
            "pose_data" in doctor_broadcast,
            doctor_broadcast,
        )

        # Doctor tab is read-only — sending a frame from it must not crash
        # the connection or get processed.
        print("[Tab C / doctor] sending a frame anyway (should be ignored, read-only role)")
        doctor_ws.send_json(frame_msg)
        # No response expected; prove the connection is still alive by
        # having the patient send one more frame and the doctor still
        # receiving it normally afterwards. Sleep past the server's own
        # 10 FPS broadcast throttle (CameraManager.fps_throttle, 100ms
        # budget) first, or this second broadcast gets silently dropped
        # and both receive_json() calls below hang forever.
        time.sleep(0.15)
        patient_ws.send_json(frame_msg)
        patient_ws.send_json(frame_msg)
        patient_ws.receive_json()  # patient's own echo
        post_doctor_msg = doctor_ws.receive_json()
        check(
            "doctor connection still alive/functioning after sending a (ignored) frame",
            post_doctor_msg.get("type") == "pose_data",
            post_doctor_msg,
        )

    print("\n[Tab B / patient] disconnected -> mid-session abandon path should NOT fire")
    print("                  (we're about to POST the normal finishing save instead)")

    # ── Tab A (therapist): finishing POST /api/sessions ──────────────────
    print("\n[Tab A / therapist] POST /api/sessions (finish/save)")
    resp = client.post(
        "/api/sessions",
        json={
            "patient_id": patient_id,
            "session_id": session_id,
            "exercise_type": "Knee Flexion",
            "affected_side": "left",
            "duration_seconds": 42,
            "total_reps": 5,
            "completed_reps": 4,
            "accuracy_percentage": 80.0,
            "average_rom": 65.0,
        },
        headers=auth_headers,
    )
    check("save_session -> 200", resp.status_code == 200, resp.text)

    with get_db() as db:
        from models import SessionModel
        sess = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        check("session row exists after save", sess is not None)
        if sess is not None:
            check("session status == 'completed'", sess.status == "completed", sess.status)
            check("session patient_id matches", sess.patient_id == patient_id)
            check("total_reps persisted", sess.total_reps == 5, sess.total_reps)

    # ── Bonus: get_sessions (therapist reads it back) ────────────────────
    resp = client.get(f"/api/sessions/{patient_id}", headers=auth_headers)
    check("GET /api/sessions/{patient_id} -> 200", resp.status_code == 200, resp.text)
    sessions_list = resp.json() if resp.status_code == 200 else []
    check("session appears in patient's session list", any(s["session_id"] == session_id for s in sessions_list))

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 70)
    if FAIL:
        print("Failed checks:")
        for f in FAIL:
            print(f"  - {f}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
