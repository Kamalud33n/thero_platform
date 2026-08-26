# Thero — Physiotherapy / Telehealth Platform

FastAPI backend for MedNova Care's physiotherapy module: real-time MediaPipe pose tracking over WebSocket, WebRTC-style telehealth rooms (Remote + Self-Training), patient records, session history, analytics, and PDF reports. Sits behind MedNova Care (Laravel) via a JWT bridge — see `API_REFERENCE.md` for the full integration model, `JSON_PAYLOADS.md` for every request/response shape.

---

## 1. Requirements

- Python 3.10 (matches the committed `__pycache__` build; MediaPipe 0.10.14 does not support newer Pythons reliably)
- MySQL 8.x (or MariaDB) reachable from this machine
- A webcam **only** if you're running the local clinic camera pipeline (`LOCAL_CAMERA_ENABLED=true`) — not needed for the cloud/telehealth path

---

## 2. Install

```bash
cd platform_thero
python3.10 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create the MySQL database (or let it auto-create — `database.py` creates the DB itself if missing, then `init_db()` creates all tables on startup):

```sql
CREATE DATABASE rehab_db_v2 CHARACTER SET utf8mb4;
```

---

## 3. Configure `.env`

Copy this template to `platform_thero/.env` and fill in the values Nada gives you. **Nothing here should be committed to git with real secrets in it.**

```env
# ── Database ────────────────────────────────────────────────────────────
DATABASE_URL=mysql+pymysql://<user>:<password>@localhost:3306/rehab_db_v2?charset=utf8mb4

# true only on a machine with a physical webcam attached (in-clinic dev
# camera pipeline). MUST be false on any cloud/telehealth deployment.
LOCAL_CAMERA_ENABLED=false

# ── MedNova Care SSO bridge (auth.py) ──────────────────────────────────
# HS256 (shared secret) or RS256 (Laravel's public key) — ask Nada which
# one staging/prod uses. RS256 is what's currently configured.
MEDNOVA_JWT_ALGORITHM=RS256

# Required only if MEDNOVA_JWT_ALGORITHM=HS256
# MEDNOVA_JWT_SECRET=

# Required only if MEDNOVA_JWT_ALGORITHM=RS256 — Laravel's PUBLIC key (PEM).
# The private key stays on MedNova Care's server, never share it here.
MEDNOVA_JWT_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----
...
-----END PUBLIC KEY-----"

# Expected iss/aud claims on the bridge JWT — get these from Nada, both
# are REQUIRED (server 500s on every request if either is unset).
MEDNOVA_JWT_ISSUER=mednovacare-staging
MEDNOVA_JWT_AUDIENCE=thero-bridge

# How far into the future a token's iat is tolerated (clock drift between
# Laravel and Thero servers). Default 60 is fine unless told otherwise.
MEDNOVA_JWT_CLOCK_SKEW_SECONDS=60

# Comma-separated MedNova Care frontend origin(s) allowed to call this API
# (HTTP CORS + WebSocket origin check — both read this same list).
# FAILS CLOSED if empty: nothing is allowed cross-origin.
MEDNOVA_ALLOWED_ORIGINS=https://mednovacare.com,https://www.mednovacare.com

# ── Thero's own tokens (NOT from MedNova — generate these yourself) ────
# python -c "import secrets; print(secrets.token_urlsafe(48))"
PATIENT_SESSION_TOKEN_SECRET=<generate a random 48+ char secret>
PATIENT_SESSION_TOKEN_TTL_SECONDS=900

# Optional — defaults to PATIENT_SESSION_TOKEN_SECRET if unset. Set it
# separately in production so a leaked patient secret can't also forge
# doctor watch tokens.
DOCTOR_SESSION_TOKEN_SECRET=<generate a random 48+ char secret>
DOCTOR_SESSION_TOKEN_TTL_SECONDS=3600

# ── Outbound webhooks: Thero -> Laravel (services/webhook.py) ──────────
# Session RESULTS webhook — fires after every session finishes/abandons.
# Get both from Nada; without them the webhook just logs and skips.
THERO_RESULTS_WEBHOOK_SECRET=
MEDNOVA_RESULTS_WEBHOOK_URL=

# Session SCHEDULED webhook — fires when a Remote/Self-Training room is
# booked, so Laravel can route the patient to their join link. Own secret,
# do not reuse THERO_RESULTS_WEBHOOK_SECRET. Laravel's receiving endpoint
# may not exist yet — confirm with Nada before relying on this.
MEDNOVA_SCHEDULE_WEBHOOK_SECRET=
MEDNOVA_SCHEDULE_WEBHOOK_URL=

# Public base URL of THIS Thero deployment, no trailing slash — needed to
# turn the relative join_url into an absolute link before sending it to
# Laravel (Laravel is on a different domain).
THERO_PUBLIC_BASE_URL=https://thero.mednovacare.com

# ── Legacy / not currently wired to any route ───────────────────────────
# Only needed if routers/integration.py (inbound Laravel -> Thero patient
# sync) ever comes back. Leave blank if you don't know what this is.
MEDNOVA_WEBHOOK_SECRET=

# ── Telehealth TURN server (optional — telehealth.py WebRTC ICE) ───────
# Leave blank to use STUN-only (works on most networks, may fail behind
# strict corporate NAT/firewalls).
METERED_TURN_URL=
METERED_TURN_USERNAME=
METERED_TURN_CREDENTIAL=

# ── Demo data ────────────────────────────────────────────────────────────
# true seeds 3 sample patients + demo sessions on a FRESH (empty) DB only.
# Leave false once real MedNova data is flowing.
SEED_DEMO_DATA=false
```

### `.env` variable reference (what breaks if you skip it)

| Variable | Required? | If missing |
|---|---|---|
| `DATABASE_URL` | Yes | App won't start — can't reach MySQL |
| `LOCAL_CAMERA_ENABLED` | No (default `true`) | Set `false` on cloud — there's no webcam on a server |
| `MEDNOVA_JWT_ALGORITHM` | No (default `HS256`) | — |
| `MEDNOVA_JWT_SECRET` | Yes if algorithm is HS256 | Every therapist request 500s |
| `MEDNOVA_JWT_PUBLIC_KEY` | Yes if algorithm is RS256 | Every therapist request 500s |
| `MEDNOVA_JWT_ISSUER` / `MEDNOVA_JWT_AUDIENCE` | **Yes, always** | Every therapist request 500s (fail-closed by design) |
| `MEDNOVA_ALLOWED_ORIGINS` | Yes for any real frontend | No cross-origin request or WebSocket will be allowed at all |
| `PATIENT_SESSION_TOKEN_SECRET` | **Yes** | `POST /api/sessions/start` 500s — no patient can join a live session |
| `DOCTOR_SESSION_TOKEN_SECRET` | No (falls back to patient secret) | Works, but weaker isolation — set separately in prod |
| `THERO_RESULTS_WEBHOOK_SECRET` / `MEDNOVA_RESULTS_WEBHOOK_URL` | No, but sessions won't reach MedNova | Session finishes normally in Thero, but the results webhook is skipped (logged) |
| `MEDNOVA_SCHEDULE_WEBHOOK_SECRET` / `MEDNOVA_SCHEDULE_WEBHOOK_URL` | No | Room is created fine, but Laravel never learns the join link |
| `THERO_PUBLIC_BASE_URL` | Needed for the schedule webhook | Scheduled-webhook build is skipped (can't make an absolute URL) |
| `SEED_DEMO_DATA` | No (default `false`) | — |

---

## 4. Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
or
```bash
python app.py
```

First run creates the database (if missing) and all tables automatically (`database.init_db()` in `app.py`).

Check it's alive:
```bash
curl http://localhost:8000/api/health
```

---

## 5. Project layout

```
platform_thero/
├── app.py                  # FastAPI app, CORS, static mounts, /api/health, demo seed
├── auth.py                 # MedNova JWT bridge + Thero's own session tokens + webhook auth
├── config.py                # MediaPipe init, PDF colors, Jinja2 templates, LOCAL_CAMERA_ENABLED
├── database.py               # SQLAlchemy engine/session, init_db()
├── models.py                # Patient, SessionModel, JointAngle, ExerciseResult, TelehealthRoom, ...
├── telehealth.py            # Remote + Self-Training rooms, /join page, /ws/signal, /ws/self-training
├── routers/
│   ├── patients.py          # Patient CRUD
│   ├── sessions.py          # In-app session start/finish, /ws/pose token issuance
│   ├── ws.py                 # /ws/pose WebSocket (in-app live pose pipeline)
│   ├── reports.py           # PDF report generation/history/download
│   ├── analytics.py          # Dashboard today/yesterday/trend
│   ├── camera.py             # LOCAL-ONLY clinic webcam pipeline
│   └── pages.py              # Server-rendered dashboard HTML pages
├── services/
│   ├── webhook.py             # Outbound webhooks to Laravel (results + scheduled)
│   ├── camera_ws.py           # RoomManager / CameraManager for /ws/pose
│   ├── metrics.py             # Rep counting, accuracy/stability/ROM scoring
│   ├── helpers.py             # protocol-version checks, summary derivation
│   ├── exercise_defs.py       # exercise → joints/target-ROM definitions
│   ├── mjpeg_camera.py       # local webcam MJPEG generator
│   └── report_builder.py     # PDF report builder (ReportLab)
├── repositories/
│   ├── patient_repo.py        # therapist-scoped patient queries (assert_owns_patient, ...)
│   └── room_repo.py            # therapist-scoped TelehealthRoom queries
├── templates/                # Jinja2 HTML (patients/session/reports/analytics/patient.html)
├── static/                   # JS/CSS (app.js, pose_ws_client.js, auth-bridge.js, style.css)
└── data/reports/, reports/    # generated PDF output
```

---

## 6. Things NOT yet finished (per code comments — flag these to Nada, don't assume)

- `routers/integration.py` (inbound Laravel → Thero patient sync) is referenced in comments but **does not exist** in this codebase — it's being replaced by the session-token model. `MEDNOVA_WEBHOOK_SECRET` is legacy/unused until/unless it comes back.
- `MEDNOVA_RESULTS_WEBHOOK_URL` and `MEDNOVA_SCHEDULE_WEBHOOK_URL` — Laravel's receiving endpoints may not be live yet. Confirm with Nada before assuming results/schedule data is actually reaching MedNova.
- `auth.py`'s `results_webhook_headers()` assumes a **plain shared secret** (`X-Webhook-Secret` header), not an RSA signing scheme — flagged as an open question for Nada, not confirmed.
- Patient sync currently has no automated way to populate `Patient.external_id` — until that exists, `create-room`'s scheduled webhook will skip patients created directly in Thero (no MedNova account to route to).
