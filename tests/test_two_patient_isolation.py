"""
Item — "Real two-patient / two-tab isolation only automated-tested, not
manually confirmed on live uvicorn."

The existing test_two_tab_flow.py proves patient+doctor cross-tab relay
WITHIN one session. It does NOT prove the thing that actually matters for
multi-tenant safety: that Session A's frames/broadcasts never leak into
Session B's tabs when TWO DIFFERENT PATIENTS are running sessions at the
same time. That's what RoomManager (services/camera_ws.py) is supposed to
guarantee by keying Rooms off session_id — this test is the first thing
that actually exercises two Rooms concurrently and asserts no cross-talk.

Two variants in this file:
  1. test_in_process()  — FastAPI TestClient (in-process ASGI), fast,
     good for CI. Same technique test_two_tab_flow.py already uses.
  2. test_live_uvicorn() — spins up a REAL uvicorn process on a local
     port and drives it with a real `websockets` client over an actual
     TCP socket. This is the "manually confirmed on live uvicorn" leg
     the checklist item asked for — same code path, but now proven
     against the real server/event-loop instead of the test harness.

Flow (both variants):
  Patient A + Doctor A  -> Session A
  Patient B + Doctor B  -> Session B (different patient, different
                            therapist token, concurrently open)

  Patient A sends a frame -> assert Doctor A receives it AND Doctor B
  does NOT receive anything for Session A. Same in reverse for B.

Run: python test_two_patient_isolation.py
"""
import os
import sys
import time
import base64
import datetime
import subprocess
import socket

os.environ["DATABASE_URL"] = "sqlite:///./test_two_patient.db"
os.environ["MEDNOVA_JWT_ALGORITHM"] = "RS256"
with open("test_public.pem") as f:
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

if os.path.exists("test_two_patient.db"):
    os.remove("test_two_patient.db")

import jwt
import numpy as np
import cv2

PASS, FAIL = [], []


def check(label, cond, detail=""):
    if cond:
        PASS.append(label); print(f"  PASS  {label}")
    else:
        FAIL.append(label); print(f"  FAIL  {label}  {detail}")


def make_therapist_token(customer_id: str, private_key: str) -> str:
    now = datetime.datetime.utcnow()
    payload = {
        "customer_id": customer_id, "external_id": customer_id,
        "type_account": "therapist", "iss": "mednova-care-test",
        "aud": "thero-test", "iat": now, "exp": now + datetime.timedelta(hours=1),
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def make_test_jpeg_b64():
    frame = (np.random.rand(240, 320, 3) * 255).astype("uint8")
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _receive_json_with_timeout(ws, timeout=2.0):
    """TestClient's websocket test session's receive_json() blocks forever
    with no built-in timeout — run it in a daemon thread and bound the
    wait explicitly instead. Returns (got_message: bool, message).
    Daemon=True is required here: if nothing ever arrives, the abandoned
    background thread must not block process exit (a plain
    ThreadPoolExecutor's atexit hook would join it and hang)."""
    import threading
    box = {}
    def _run():
        try:
            box["result"] = ws.receive_json()
            box["ok"] = True
        except Exception as exc:
            box["ok"] = False
            box["error"] = exc
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return False, None  # still blocked -> nothing arrived in time
    return box.get("ok", False), box.get("result")
def test_in_process():
    from fastapi.testclient import TestClient
    from database import get_db, init_db
    from models import Patient
    import app as app_module

    init_db()
    client = TestClient(app_module.app)

    with open("test_private.pem") as f:
        private_key = f.read()

    def setup_patient(consultant_id, name):
        with get_db() as db:
            p = Patient(name=name, age=35, gender="Other",
                        mednova_consultant_id=consultant_id)
            db.add(p); db.commit(); db.refresh(p)
            return p.id

    consultant_a, consultant_b = "consultant-A", "consultant-B"
    patient_a_id = setup_patient(consultant_a, "Patient A")
    patient_b_id = setup_patient(consultant_b, "Patient B")
    headers_a = {"Authorization": f"Bearer {make_therapist_token(consultant_a, private_key)}"}
    headers_b = {"Authorization": f"Bearer {make_therapist_token(consultant_b, private_key)}"}

    def start_session(patient_id, headers):
        resp = client.post("/api/sessions/start",
                            json={"patient_id": patient_id, "exercise_type": "Knee Flexion",
                                  "affected_side": "left"},
                            headers=headers)
        check(f"start_session({patient_id}) -> 200", resp.status_code == 200, resp.text)
        body = resp.json()
        return body["session_id"], body["token"]

    def watch_token(session_id, headers):
        resp = client.post(f"/api/sessions/{session_id}/watch-token", headers=headers)
        check(f"watch-token({session_id}) -> 200", resp.status_code == 200, resp.text)
        return resp.json()["token"]

    session_a, patient_token_a = start_session(patient_a_id, headers_a)
    session_b, patient_token_b = start_session(patient_b_id, headers_b)
    doctor_token_a = watch_token(session_a, headers_a)
    doctor_token_b = watch_token(session_b, headers_b)

    origin = {"origin": "http://testorigin.local"}
    print(f"\n[in-process] opening 4 concurrent WS: patient A/B + doctor A/B, "
          f"session A={session_a} session B={session_b}")

    with client.websocket_connect(
        f"/ws/pose?session_id={session_a}&role=patient&token={patient_token_a}", headers=origin
    ) as pa, client.websocket_connect(
        f"/ws/pose?session_id={session_a}&role=doctor&token={doctor_token_a}", headers=origin
    ) as da, client.websocket_connect(
        f"/ws/pose?session_id={session_b}&role=patient&token={patient_token_b}", headers=origin
    ) as pb, client.websocket_connect(
        f"/ws/pose?session_id={session_b}&role=doctor&token={doctor_token_b}", headers=origin
    ) as db_:

        for ws, label in [(pa, "patient A"), (da, "doctor A"), (pb, "patient B"), (db_, "doctor B")]:
            ack = ws.receive_json()
            check(f"{label} connected", ack.get("type") == "connected", ack)

        # Patient A sends 2 frames (frame-skip processes every 2nd)
        jpeg = make_test_jpeg_b64()
        for _ in range(2):
            pa.send_json({"type": "frame", "data": jpeg})
        pa_ack = pa.receive_json()
        check("patient A gets own broadcast", pa_ack.get("type") == "pose_data")
        da_ack = da.receive_json()
        check("doctor A gets session A broadcast", da_ack.get("type") == "pose_data")

        # Doctor B must NOT receive anything for session A's frame — bound
        # the wait explicitly since receive_json() has no native timeout.
        leaked, _ = _receive_json_with_timeout(db_, timeout=2.0)
        check("doctor B receives NOTHING from patient A's frame (no cross-session leak)", not leaked,
              "doctor B got a message meant for session A!" if leaked else "")

        # Now patient B sends a frame -> doctor B should get it, doctor A must not
        jpeg2 = make_test_jpeg_b64()
        for _ in range(2):
            pb.send_json({"type": "frame", "data": jpeg2})
        pb_ack = pb.receive_json()
        check("patient B gets own broadcast", pb_ack.get("type") == "pose_data")
        db_ack = db_.receive_json()
        check("doctor B gets session B broadcast", db_ack.get("type") == "pose_data")

        leaked2, _ = _receive_json_with_timeout(da, timeout=2.0)
        check("doctor A receives NOTHING from patient B's frame (no cross-session leak)", not leaked2,
              "doctor A got a message meant for session B!" if leaked2 else "")


# ─────────────────────────── Variant 2: live uvicorn ──────────────────────
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_live_uvicorn():
    import asyncio
    import websockets
    import httpx

    port = _free_port()
    env = dict(os.environ)
    env["DATABASE_URL"] = "sqlite:///./test_two_patient_live.db"
    if os.path.exists("test_two_patient_live.db"):
        os.remove("test_two_patient_live.db")

    print(f"\n[live uvicorn] starting real server on 127.0.0.1:{port} ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        env=env, cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )
    try:
        # Wait for the server to actually accept connections.
        deadline = time.time() + 20
        up = False
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    up = True
                    break
            except OSError:
                time.sleep(0.3)
        check("live uvicorn accepted a TCP connection", up)
        if not up:
            return

        async def run():
            with open("test_private.pem") as f:
                private_key = f.read()

            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as http:
                # Give the app a moment to finish init_db()/model loading.
                for _ in range(20):
                    try:
                        r = await http.get("/api/camera/status")
                        if r.status_code == 200:
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)

                consultant_a, consultant_b = "live-consultant-A", "live-consultant-B"
                headers_a = {"Authorization": f"Bearer {make_therapist_token(consultant_a, private_key)}"}
                headers_b = {"Authorization": f"Bearer {make_therapist_token(consultant_b, private_key)}"}

                # Need real Patient rows — insert directly via the live
                # server's own DB file (same technique test_two_tab_flow.py
                # uses; no patient-creation API exists post phase-1 removal).
                os.environ["DATABASE_URL"] = env["DATABASE_URL"]
                import importlib
                import database as _database
                importlib.reload(_database)
                from models import Patient
                with _database.get_db() as db:
                    pa_row = Patient(name="Live Patient A", age=30, gender="Other",
                                      mednova_consultant_id=consultant_a)
                    pb_row = Patient(name="Live Patient B", age=30, gender="Other",
                                      mednova_consultant_id=consultant_b)
                    db.add(pa_row); db.add(pb_row); db.commit()
                    db.refresh(pa_row); db.refresh(pb_row)
                    patient_a_id, patient_b_id = pa_row.id, pb_row.id

                async def start_session(patient_id, headers):
                    r = await http.post("/api/sessions/start",
                                         json={"patient_id": patient_id,
                                               "exercise_type": "Knee Flexion",
                                               "affected_side": "left"},
                                         headers=headers)
                    check(f"[live] start_session({patient_id}) -> 200", r.status_code == 200, r.text)
                    b = r.json()
                    return b["session_id"], b["token"]

                async def watch_token(session_id, headers):
                    r = await http.post(f"/api/sessions/{session_id}/watch-token", headers=headers)
                    check(f"[live] watch-token({session_id}) -> 200", r.status_code == 200, r.text)
                    return r.json()["token"]

                session_a, patient_token_a = await start_session(patient_a_id, headers_a)
                session_b, patient_token_b = await start_session(patient_b_id, headers_b)
                doctor_token_a = await watch_token(session_a, headers_a)
                doctor_token_b = await watch_token(session_b, headers_b)

                origin_hdr = [("origin", "http://testorigin.local")]
                url = lambda sid, role, tok: (
                    f"ws://127.0.0.1:{port}/ws/pose?session_id={sid}&role={role}&token={tok}"
                )

                async with websockets.connect(url(session_a, "patient", patient_token_a),
                                               extra_headers=origin_hdr) as pa, \
                           websockets.connect(url(session_a, "doctor", doctor_token_a),
                                               extra_headers=origin_hdr) as da, \
                           websockets.connect(url(session_b, "patient", patient_token_b),
                                               extra_headers=origin_hdr) as pb, \
                           websockets.connect(url(session_b, "doctor", doctor_token_b),
                                               extra_headers=origin_hdr) as db_:

                    import json as _json
                    for ws, label in [(pa, "patient A"), (da, "doctor A"),
                                       (pb, "patient B"), (db_, "doctor B")]:
                        ack = _json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                        check(f"[live] {label} connected", ack.get("type") == "connected", ack)

                    jpeg = make_test_jpeg_b64()
                    for _ in range(2):
                        await pa.send(_json.dumps({"type": "frame", "data": jpeg}))
                    pa_msg = _json.loads(await asyncio.wait_for(pa.recv(), timeout=10))
                    check("[live] patient A gets own broadcast", pa_msg.get("type") == "pose_data")
                    da_msg = _json.loads(await asyncio.wait_for(da.recv(), timeout=10))
                    check("[live] doctor A gets session A broadcast", da_msg.get("type") == "pose_data")

                    leaked = True
                    try:
                        await asyncio.wait_for(db_.recv(), timeout=2)
                    except asyncio.TimeoutError:
                        leaked = False
                    check("[live] doctor B receives NOTHING from patient A's frame "
                          "(real uvicorn, no cross-session leak)", not leaked)

                    jpeg2 = make_test_jpeg_b64()
                    for _ in range(2):
                        await pb.send(_json.dumps({"type": "frame", "data": jpeg2}))
                    pb_msg = _json.loads(await asyncio.wait_for(pb.recv(), timeout=10))
                    check("[live] patient B gets own broadcast", pb_msg.get("type") == "pose_data")
                    db_msg = _json.loads(await asyncio.wait_for(db_.recv(), timeout=10))
                    check("[live] doctor B gets session B broadcast", db_msg.get("type") == "pose_data")

                    leaked2 = True
                    try:
                        await asyncio.wait_for(da.recv(), timeout=2)
                    except asyncio.TimeoutError:
                        leaked2 = False
                    check("[live] doctor A receives NOTHING from patient B's frame "
                          "(real uvicorn, no cross-session leak)", not leaked2)

        asyncio.run(run())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    print("=" * 70)
    print("TWO-PATIENT / TWO-SESSION ISOLATION TEST")
    print("=" * 70)

    print("\n--- Variant 1: in-process (TestClient) ---")
    try:
        test_in_process()
    except Exception as exc:
        FAIL.append("test_in_process crashed")
        print(f"  FAIL  test_in_process crashed: {exc}")

    print("\n--- Variant 2: live uvicorn (real TCP server) ---")
    try:
        test_live_uvicorn()
    except Exception as exc:
        FAIL.append("test_live_uvicorn crashed")
        print(f"  FAIL  test_live_uvicorn crashed: {exc}")

    print("\n" + "=" * 70)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 70)
    if FAIL:
        sys.exit(1)
