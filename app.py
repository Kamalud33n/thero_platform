import os
import random
import datetime

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from database import get_db, init_db
from models import Patient, SessionModel, JointAngle, Setting
from telehealth import router as telehealth_router
from auth import ALLOWED_ORIGINS

from config import pose as _pose  # for /api/health mediapipe_ready flag
from services.camera_ws import ws_mgr, CameraManager
from services import mjpeg_camera

from routers import sessions, camera, ws, reports, analytics, patients, pages

init_db()  # creates all tables (and the MySQL database itself, if missing)

# ─── FastAPI app 
app = FastAPI(title="Rehabilitation AI System", version="2.2.0")

# allow_origins=["*"] together with allow_credentials=True is insecure once
# JWTs are carried in the Authorization header — a wildcard origin + credentials
# combo lets any site read authenticated responses. Restrict to the actual
# MedNova Care frontend domain(s) via env var (comma-separated for multiple).
# ALLOWED_ORIGINS is imported from auth.py rather than parsed again here so
# the HTTP CORS allowlist and the WebSocket origin checks (auth.check_ws_origin,
# used in routers/ws.py and telehealth.py) can never drift out of sync.
_allowed_origins = ALLOWED_ORIGINS
if not _allowed_origins:
    print("WARNING: MEDNOVA_ALLOWED_ORIGINS is not set — CORS will allow no cross-origin "
          "requests. Set it to your MedNova Care frontend domain(s), comma-separated.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static",  StaticFiles(directory="static"),  name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/assets",  StaticFiles(directory="assets"),  name="assets")

app.include_router(telehealth_router)   # includes /join/{room_id} (patient page) + telehealth API/WS
app.include_router(sessions.router)     # /api/sessions/* — engine-facing, async-only
app.include_router(camera.router)       # local-only MJPEG dev camera, gated by LOCAL_CAMERA_ENABLED
app.include_router(ws.router)           # /ws/pose — landmarks-only pipeline
app.include_router(reports.router)      # /api/reports/* — PDF session report generate + download
app.include_router(analytics.router)    # /api/analytics — today/yesterday/trend dashboard data
app.include_router(patients.router)     # /api/patients/* — patient CRUD (backs patients.html)
app.include_router(pages.router)        # /, /patients, /session, /reports, /analytics — dashboard UI pages


#Startup seed
# Off by default — set SEED_DEMO_DATA=true in the environment if you ever want
# the 3 sample patients + random demo sessions back (e.g. for a fresh demo).
@app.on_event("startup")
async def seed():
    if os.getenv("SEED_DEMO_DATA", "false").lower() != "true":
        return
    with get_db() as db:
        if db.query(Patient).count() > 0:
            return

        # mednova_consultant_id here is a placeholder demo id — in real use
        # it's always populated from the verified JWT (auth.get_current_therapist),
        # never typed by hand.
        _demo_consultant_id = "DEMO-CONSULTANT-1"
        sample_patients = [
            Patient(name="John Smith",      age=45, gender="Male",   weight=82.5, height=178.0,
                    diagnosis="Rotator Cuff Tear",       affected_body_part="Right Shoulder",
                    mednova_consultant_id=_demo_consultant_id,
                    phone="+1 (555) 123-4567",           email="john.smith@email.com"),
            Patient(name="Maria Garcia",    age=62, gender="Female", weight=68.0, height=165.0,
                    diagnosis="Knee Osteoarthritis",     affected_body_part="Left Knee",
                    mednova_consultant_id=_demo_consultant_id,
                    phone="+1 (555) 234-5678",           email="maria.garcia@email.com"),
            Patient(name="Robert Williams", age=38, gender="Male",   weight=90.0, height=183.0,
                    diagnosis="Lumbar Disc Herniation",  affected_body_part="Lower Back",
                    mednova_consultant_id=_demo_consultant_id,
                    phone="+1 (555) 345-6789",           email="robert.williams@email.com"),
        ]

        for p in sample_patients:
            db.add(p)
        db.flush()

        exercises = ["Shoulder Rehab", "Knee Flexion", "Arm Raise", "Balance Exercise"]
        joints    = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_knee", "right_knee"]

        for p in sample_patients:
            for _ in range(5):
                sess = SessionModel(
                    patient_id          = p.id,
                    exercise_type       = random.choice(exercises),
                    start_time          = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 30)),
                    duration_seconds    = random.randint(180, 600),
                    total_reps          = random.randint(10, 20),
                    completed_reps      = random.randint(5, 18),
                    accuracy_percentage = random.uniform(60, 95),
                    average_rom         = random.uniform(40, 85),
                    incorrect_movements = random.randint(0, 5),
                    stability_score     = random.uniform(50, 90),
                    balance_score       = random.uniform(45, 85),
                    movement_smoothness = random.uniform(50, 90),
                    fatigue_estimation  = random.uniform(10, 40),
                    recovery_score      = random.uniform(40, 80),
                )
                db.add(sess)
                db.flush()

                for joint in joints:
                    db.add(JointAngle(
                        session_id   = sess.id,
                        joint_name   = joint,
                        angle_value  = random.uniform(30, 120),
                        target_angle = random.uniform(40, 110),
                        deviation    = random.uniform(-15, 15),
                        is_correct   = random.choice([True, True, True, False]),
                    ))

        for key, val, desc in [
            ("fps_target",           "10",      "Target FPS for pose streaming"),
            ("camera_resolution",    "320x240", "Camera resolution"),
            ("confidence_threshold", "0.5",     "Min confidence threshold"),
            ("rom_warning",          "30",      "Min ROM warning threshold"),
        ]:
            db.add(Setting(key=key, value=val, description=desc))

        db.commit()
        print("Seed data inserted.")


# Root ("/") is now handled by routers/pages.py (page_index — redirects to
# /analytics). The therapist-facing dashboard UI (patients/session/
# analytics/reports pages) is back in this service; see pages.router.

# Health check
@app.get("/api/health")
async def health():
    # camera_running used to be one global flag (single shared camera).
    # Now every /ws/pose connection owns its own CameraManager, so "is a
    # camera running" is answered per-session — ws_sessions_active is how
    # many of those are live right now.
    return {
        "status":             "healthy",
        "timestamp":          datetime.datetime.now().isoformat(),
        "ws_sessions_active": len(ws_mgr.rooms),
        "mjpeg_running":      mjpeg_camera.is_active(),
        "ws_connections":     len(ws_mgr.connections),
        "mediapipe_ready":    _pose is not None,
        "target_fps":         CameraManager.TARGET_FPS,
        "resolution":         "320x240",
    }


# Entry point 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)