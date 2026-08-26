# thero

FastAPI physiotherapy engine (pose/hand tracking, telehealth, session
metrics, PDF reports) integrating with the MedNova Care platform.

## Local (no Docker)

Requires Python 3.10.11.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in real values
uvicorn app:app --reload
```

## Docker

```bash
cp .env.example .env            # then fill in real values
docker compose up --build
```

This starts two containers:
- `app` — the FastAPI service on `http://localhost:8000`
- `db` — MySQL 8.0, with data persisted in the `thero_mysql_data` volume

`data/`, `reports/`, and `uploads/` are named volumes so generated PDFs
and session data survive container restarts.

`LOCAL_CAMERA_ENABLED` is forced to `false` in both the Dockerfile and
compose file — there is no physical camera attached to a cloud/container
host, so the local MJPEG dev-camera pipeline (`routers/camera.py`) is
disabled there. It only makes sense when running directly on a machine
that has a webcam attached.

## Tests

```bash
pytest
```

Run from the project root — `tests/conftest.py` adds the root to
`sys.path` so the test files can import `app`, `database`, `models`,
etc. as top-level modules.

## Notes

- `.env` is git-ignored. Never commit real secrets — `.env.example` has
  placeholders only.
- `MEDNOVA_JWT_PUBLIC_KEY` in `.env.example` is a placeholder; use the
  real PEM from MedNova Care.
- **Time/timezone:** everything internal (DB, tokens, WS `"ts"` fields,
  `/api/health`) runs on UTC — see `services/timeutils.py`. This is
  deployment-independent by design (no reliance on the server OS's
  clock/timezone, and no network/IP-based auto-detection, which is
  unreliable for a backend). `DISPLAY_TIMEZONE` (default `Asia/Muscat`)
  only affects server-rendered PDF reports, since those have no browser
  to localize the time themselves; anything shown in the dashboard/session
  pages localizes automatically on the viewer's own device.
  `docker-compose.yml` also pins the MySQL container's own clock to UTC
  (`--default-time-zone=+00:00`) so `func.now()` DB-side defaults in
  `models.py` match — if you ever point this at a different MySQL
  instance, make sure its `time_zone` is UTC too, or the two will drift.
- The `custom` date-range report filter (`routers/reports.py`) takes a
  plain `YYYY-MM-DD` string and compares it directly against UTC
  `start_time` values — for a Muscat-based caller near midnight this can
  be off by up to `DISPLAY_TIMEZONE`'s offset. Not fixed here since it
  needs a decision on how the date range should be interpreted; worth a
  look before it matters in practice.
