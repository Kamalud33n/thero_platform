"""
MedNova Care SSO bridge — JWT verification.

Env vars:
    MEDNOVA_JWT_ALGORITHM     "HS256" (default) or "RS256"
    MEDNOVA_JWT_SECRET        shared secret, required for HS256
    MEDNOVA_JWT_PUBLIC_KEY    PEM public key, required for RS256
    MEDNOVA_JWT_ISSUER        expected `iss` claim — required, server
                               returns 500 on any request if unset
    MEDNOVA_JWT_AUDIENCE      expected `aud` claim — required, same as above
    MEDNOVA_JWT_CLOCK_SKEW_SECONDS
                               how far into the future an `iat` is allowed
                               to be before the token is rejected
                               (default 60)
    MEDNOVA_WEBHOOK_SECRET    separate shared secret for the Laravel ->
                               thero patient-sync webhook (routers/integration.py,
                               not yet built as of this writing),
                               NOT the therapist JWT.
    THERO_RESULTS_WEBHOOK_SECRET
                               separate shared secret for the OPPOSITE
                               direction: thero -> Laravel session-results
                               webhook (item 27, services/webhook.py).
                               Deliberately its own env var, not reused
                               from MEDNOVA_WEBHOOK_SECRET — per Nada
                               (2026-08-23): "the two directions are
                               opposite in purpose and permission, and the
                               old one was used in development so I'd
                               rather start clean." Nada is generating and
                               sending this value; thero does not generate
                               it.
    MEDNOVA_RESULTS_WEBHOOK_URL
                               the Laravel endpoint thero POSTs session
                               results to (services/webhook.py). Not yet
                               supplied by Nada as of this writing — get
                               this from her along with the secret above.
    MEDNOVA_SCHEDULE_WEBHOOK_SECRET
                               separate shared secret for a THIRD direction:
                               thero -> Laravel "session scheduled" webhook
                               (services/webhook.py, fired from
                               telehealth.create_room). Deliberately its own
                               env var, not reused from
                               THERO_RESULTS_WEBHOOK_SECRET or
                               MEDNOVA_WEBHOOK_SECRET — same "opposite
                               purpose/permission, start clean" reasoning
                               as the results webhook above applies here
                               too. Laravel doesn't have a receiving
                               endpoint for this yet as of this writing —
                               "inime than build pannuvanga" (their side
                               will build it later); get the URL + secret
                               from them once it exists.
    MEDNOVA_SCHEDULE_WEBHOOK_URL
                               the Laravel endpoint thero POSTs the
                               join_url to once a telehealth room (Remote
                               or Self Training) is scheduled, so Laravel
                               can route the patient to it (their delivery
                               mechanism — in-app notification, SMS, etc.
                               — thero has no visibility into which one).
                               Not yet supplied — Laravel side isn't built
                               yet.
"""
import os
import time
import uuid
import datetime
from dataclasses import dataclass

import jwt
from fastapi import Header, HTTPException

from services.timeutils import utcnow

JWT_ALGORITHM = os.getenv("MEDNOVA_JWT_ALGORITHM", "HS256")
JWT_SECRET = os.getenv("MEDNOVA_JWT_SECRET")
JWT_PUBLIC_KEY = os.getenv("MEDNOVA_JWT_PUBLIC_KEY")
JWT_ISSUER = os.getenv("MEDNOVA_JWT_ISSUER")       # required — see _verify_iss_aud_configured()
JWT_AUDIENCE = os.getenv("MEDNOVA_JWT_AUDIENCE")   # required — see _verify_iss_aud_configured()
WEBHOOK_SECRET = os.getenv("MEDNOVA_WEBHOOK_SECRET")
# Item 25 / item 27 prep: separate secret for the OUTBOUND direction
# (thero -> Laravel session-results webhook). See module docstring above
# for why this is not the same value as WEBHOOK_SECRET. Not consumed
# anywhere yet — item 27's webhook sender will read this when it's built.
RESULTS_WEBHOOK_SECRET = os.getenv("THERO_RESULTS_WEBHOOK_SECRET")

# Third direction, own secret again (see module docstring): thero -> Laravel
# "session scheduled" webhook, fired from telehealth.create_room() so
# Laravel can route the join_url to the right patient's MedNova account.
# Laravel's receiving endpoint doesn't exist yet as of this writing.
SCHEDULE_WEBHOOK_SECRET = os.getenv("MEDNOVA_SCHEDULE_WEBHOOK_SECRET")

# Single source of truth for allowed frontend origins — used both by
# app.py's CORSMiddleware (HTTP requests) and by check_ws_origin() below
# (WebSocket handshakes). Starlette/FastAPI's CORSMiddleware does NOT
# protect WebSocket routes — it only inspects regular HTTP requests — so
# every websocket.websocket(...) endpoint must call check_ws_origin()
# itself before accepting the connection, or origin is effectively
# unchecked for that channel.
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("MEDNOVA_ALLOWED_ORIGINS", "").split(",") if o.strip()
]


def check_ws_origin(websocket) -> bool:
    """
    Call this first, before accepting any WebSocket connection that a
    browser (not a server-to-server client) is expected to open. Mirrors
    the HTTP CORS allowlist so a WS channel can't be opened from a page
    on a different origin even though CORSMiddleware itself never sees it.

    Fails closed: if MEDNOVA_ALLOWED_ORIGINS isn't set, no origin is
    considered allowed (same fail-closed behavior as app.py's CORS setup).
    """
    origin = websocket.headers.get("origin")
    if not ALLOWED_ORIGINS or not origin:
        return False
    return origin in ALLOWED_ORIGINS


@dataclass
class CurrentTherapist:
    """Decoded identity of the therapist making the request."""
    customer_id: str
    external_id: str
    mednova_consultant_id: str
    type_account: str


def _verify_key() -> str:
    if JWT_ALGORITHM.upper() == "RS256":
        if not JWT_PUBLIC_KEY:
            raise RuntimeError("MEDNOVA_JWT_PUBLIC_KEY is not set (required for RS256)")
        return JWT_PUBLIC_KEY
    if not JWT_SECRET:
        raise RuntimeError("MEDNOVA_JWT_SECRET is not set (required for HS256)")
    return JWT_SECRET


def _verify_iss_aud_configured() -> None:
    """iss/aud used to be validated only when the corresponding env var
    happened to be set — meaning an ops mistake (forgetting to set
    MEDNOVA_JWT_ISSUER/AUDIENCE) silently turned the check off instead of
    failing. Both are required now: same fail-closed posture as the
    secret/key check above."""
    if not JWT_ISSUER:
        raise RuntimeError("MEDNOVA_JWT_ISSUER is not set (required)")
    if not JWT_AUDIENCE:
        raise RuntimeError("MEDNOVA_JWT_AUDIENCE is not set (required)")


# Clock-skew tolerance for the iat check below, in seconds. Laravel and
# thero run on different machines — small drift is normal and shouldn't
# 401 legitimate tokens.
JWT_CLOCK_SKEW_SECONDS = int(os.getenv("MEDNOVA_JWT_CLOCK_SKEW_SECONDS", "60"))


async def get_current_therapist(authorization: str = Header(None)) -> CurrentTherapist:
    """
    FastAPI dependency — decodes and verifies the bridge JWT issued by
    Laravel. Raises 401 on missing/invalid/expired token, or if the token
    doesn't belong to a therapist account.

    Expects: Authorization: Bearer <jwt>
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(401, "Missing bridge token")

    try:
        key = _verify_key()
        _verify_iss_aud_configured()
        payload = jwt.decode(
            token,
            key,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            # Fail closed on missing claims instead of silently treating
            # an absent iss/aud/iat as "nothing to check".
            options={"require": ["exp", "iss", "aud", "iat"]},
        )
    except RuntimeError as exc:
        # Server misconfiguration (no secret/key, or iss/aud not configured)
        # — not the caller's fault, but we still must not let the request through.
        raise HTTPException(500, str(exc))
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Bridge token has expired — please reload from MedNova Care")
    except jwt.InvalidIssuerError:
        raise HTTPException(401, "Bridge token has an unexpected issuer")
    except jwt.InvalidAudienceError:
        raise HTTPException(401, "Bridge token has an unexpected audience")
    except jwt.MissingRequiredClaimError as exc:
        raise HTTPException(401, f"Bridge token is missing required claim: {exc.claim}")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid bridge token")

    # PyJWT verifies exp/iss/aud but does NOT check iat against the clock —
    # a token "issued" in the future (clock skew, or a forged/backdated
    # token) would otherwise sail through as long as exp is still ahead of
    # now. Reject anything issued more than JWT_CLOCK_SKEW_SECONDS in the
    # future.
    iat = payload.get("iat")
    now = time.time()
    if iat > now + JWT_CLOCK_SKEW_SECONDS:
        raise HTTPException(401, "Bridge token has an iat in the future")

    if payload.get("type_account") != "therapist":
        raise HTTPException(401, "This token does not belong to a therapist account")

    # Single source of truth for identity: customer_id only. We intentionally
    # do NOT read a separate `mednova_consultant_id` claim — see module
    # docstring. Both customers.id and therapists.id are valid-looking
    # integers in Laravel's DB, so trusting the wrong claim here would
    # silently attach records to the wrong therapist instead of erroring.
    customer_id = payload.get("customer_id")
    if not customer_id:
        raise HTTPException(401, "Bridge token is missing customer_id")

    return CurrentTherapist(
        customer_id=str(customer_id),
        external_id=str(payload.get("external_id", "")),
        mednova_consultant_id=str(customer_id),
        type_account=payload.get("type_account"),
    )


# ─── Patient session-scoped token (/ws/pose) ──────────────────────────────
#
# thero's OWN token, unrelated to the MedNova bridge JWT above. The patient
# never logs into MedNova or thero — Laravel has no identity for them at
# all — so this can't be a Laravel-issued token; thero has to mint and
# verify it itself, hence a separate signing secret.
#
# Lifecycle:
#   1. Therapist calls POST /api/sessions/start (routers/sessions.py) — this
#      creates a real SessionModel row (status="in_progress") tied to a
#      patient_id that assert_owns_patient() has already confirmed exists
#      and belongs to this therapist. Session creation WITHOUT a real
#      patient DB row is exactly what this closes off: there is no code
#      path to get a patient session token without that row existing first.
#   2. That endpoint calls issue_patient_session_token(session_id, patient_id)
#      and returns the token to the therapist's page, which hands it to the
#      patient (join link / QR / etc.).
#   3. The patient's browser opens /ws/pose?session_id=...&role=patient&token=...
#      routers/ws.py calls decode_patient_session_token() then
#      consume_patient_session_jti() before accepting the connection.
#
# Required env var:
#   PATIENT_SESSION_TOKEN_SECRET   thero's own HS256 signing secret for
#                                   this token type. Generate with e.g.
#                                   `python -c "import secrets; print(secrets.token_urlsafe(48))"`
#                                   and set it in .env — fails closed
#                                   (500) if unset, same posture as the
#                                   MEDNOVA_JWT_* checks above.
PATIENT_TOKEN_SECRET   = os.getenv("PATIENT_SESSION_TOKEN_SECRET")
PATIENT_TOKEN_ISSUER   = os.getenv("PATIENT_SESSION_TOKEN_ISSUER", "thero")
PATIENT_TOKEN_AUDIENCE = os.getenv("PATIENT_SESSION_TOKEN_AUDIENCE", "thero-patient-session")
# Short TTL on purpose — this token is meant to be consumed within seconds
# of being issued (patient's page loads and immediately opens the WS). It
# is NOT a "join whenever within 2h" link like TelehealthRoom.token; if the
# patient needs longer to join, re-call POST /api/sessions/start for a
# fresh token rather than extending this one.
PATIENT_TOKEN_TTL_SECONDS = int(os.getenv("PATIENT_SESSION_TOKEN_TTL_SECONDS", "900"))  # 15 min


@dataclass
class PatientSessionClaims:
    session_id: str
    patient_id: str
    jti: str


def issue_patient_session_token(session_id: str, patient_id: str) -> str:
    if not PATIENT_TOKEN_SECRET:
        raise RuntimeError("PATIENT_SESSION_TOKEN_SECRET is not set (required)")
    now = int(time.time())
    payload = {
        "session_id": str(session_id),
        "patient_id": str(patient_id),
        "role":       "patient",
        "iss":        PATIENT_TOKEN_ISSUER,
        "aud":        PATIENT_TOKEN_AUDIENCE,
        "iat":        now,
        "exp":        now + PATIENT_TOKEN_TTL_SECONDS,
        # Unique per issuance (not per session) — calling
        # POST /api/sessions/start again for the same session_id (e.g. the
        # therapist regenerates the join link) mints a fresh jti, so the
        # old token and the new one are tracked as separate single-use
        # redemptions rather than colliding.
        "jti":        uuid.uuid4().hex,
    }
    return jwt.encode(payload, PATIENT_TOKEN_SECRET, algorithm="HS256")


def decode_patient_session_token(token: str) -> PatientSessionClaims:
    """
    Verifies signature/exp/iss/aud/iat — mirrors get_current_therapist()'s
    fail-closed posture (missing config -> 500, not a silent pass). Does
    NOT check or consume the jti — that needs a DB transaction, done
    separately by consume_patient_session_jti() so the caller (routers/ws.py)
    controls exactly when the "used" row is committed relative to accepting
    the WebSocket connection.
    """
    if not PATIENT_TOKEN_SECRET:
        raise HTTPException(500, "PATIENT_SESSION_TOKEN_SECRET is not set on this server")
    if not token:
        raise HTTPException(401, "Missing session token")

    try:
        payload = jwt.decode(
            token,
            PATIENT_TOKEN_SECRET,
            algorithms=["HS256"],
            issuer=PATIENT_TOKEN_ISSUER,
            audience=PATIENT_TOKEN_AUDIENCE,
            options={"require": ["exp", "iss", "aud", "iat", "jti", "session_id", "patient_id"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Session link has expired — ask your therapist to start a new session")
    except jwt.InvalidIssuerError:
        raise HTTPException(401, "Invalid session token issuer")
    except jwt.InvalidAudienceError:
        raise HTTPException(401, "Invalid session token audience")
    except jwt.MissingRequiredClaimError as exc:
        raise HTTPException(401, f"Session token is missing required claim: {exc.claim}")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid session token")

    # Same clock-skew guard as the therapist bridge token — PyJWT verifies
    # exp/iss/aud but never checks iat against the clock on its own.
    iat = payload.get("iat")
    now = time.time()
    if iat > now + JWT_CLOCK_SKEW_SECONDS:
        raise HTTPException(401, "Session token has an iat in the future")

    return PatientSessionClaims(
        session_id=str(payload["session_id"]),
        patient_id=str(payload["patient_id"]),
        jti=str(payload["jti"]),
    )


def consume_patient_session_jti(db, claims: PatientSessionClaims) -> None:
    """
    Single-use / replay-protection enforcement. Call this only after
    decode_patient_session_token() has already verified the token is
    signature-valid and unexpired.

    Raises HTTPException(401) if this jti has already been redeemed
    (replay). Caller must be inside an open `with get_db() as db:` block
    and commit immediately after this returns successfully — see
    routers/ws.py for the exact sequencing (insert + commit BEFORE
    accepting the WebSocket, so a race between two connections racing the
    same token can't both slip through before either commits).
    """
    from models import UsedPatientToken

    if db.query(UsedPatientToken).filter(UsedPatientToken.jti == claims.jti).first() is not None:
        raise HTTPException(401, "This session link has already been used — ask your therapist for a new one")

    db.add(UsedPatientToken(
        jti=claims.jti,
        session_id=claims.session_id,
        expires_at=utcnow() + datetime.timedelta(seconds=PATIENT_TOKEN_TTL_SECONDS),
    ))


# ─── Doctor session-scoped token (/ws/pose?role=doctor) ──────────────────
#
# Closes the gap flagged in services/camera_ws.py's RoomManager docstring
# ("role is currently trusted from the query string ... there's no signed
# claim behind it yet"). Mirrors the patient session token above, but:
#   - minted for a THERAPIST who is already authenticated via the MedNova
#     bridge JWT (get_current_therapist), not an anonymous patient link
#   - NOT single-use: a doctor may legitimately reconnect/refresh the
#     observation tab several times during one live session, so there's
#     no jti replay-consumption step like the patient token has. Ownership
#     is re-checked at issuance time (see routers/sessions.py), and the
#     short TTL bounds how long a leaked token stays useful.
#   - carries role="doctor" explicitly so routers/ws.py can trust the
#     claim instead of the raw query-string role param.
#
# Lifecycle:
#   1. Therapist's page (already holding a valid MedNova bridge JWT) calls
#      POST /api/sessions/{session_id}/watch-token (routers/sessions.py).
#      That endpoint re-confirms this therapist owns the patient behind
#      session_id via assert_owns_patient() before minting anything.
#   2. The doctor's browser opens
#      /ws/pose?session_id=...&role=doctor&token=<this token>
#      routers/ws.py calls decode_doctor_session_token() and checks role
#      + session_id before accepting the connection.
DOCTOR_TOKEN_SECRET   = os.getenv("DOCTOR_SESSION_TOKEN_SECRET", PATIENT_TOKEN_SECRET)
DOCTOR_TOKEN_ISSUER   = os.getenv("DOCTOR_SESSION_TOKEN_ISSUER", "thero")
DOCTOR_TOKEN_AUDIENCE = os.getenv("DOCTOR_SESSION_TOKEN_AUDIENCE", "thero-doctor-session")
# Longer than the patient TTL on purpose — a doctor is watching a live
# session that can run for many minutes, and unlike the patient token
# this one is meant to stay valid for the observation window, not just
# long enough for one page load.
DOCTOR_TOKEN_TTL_SECONDS = int(os.getenv("DOCTOR_SESSION_TOKEN_TTL_SECONDS", "3600"))  # 1h


@dataclass
class DoctorSessionClaims:
    session_id: str
    mednova_consultant_id: str
    role: str


def issue_doctor_session_token(session_id: str, therapist: "CurrentTherapist") -> str:
    if not DOCTOR_TOKEN_SECRET:
        raise RuntimeError("DOCTOR_SESSION_TOKEN_SECRET (or PATIENT_SESSION_TOKEN_SECRET) is not set (required)")
    now = int(time.time())
    payload = {
        "session_id":             str(session_id),
        "mednova_consultant_id":  str(therapist.mednova_consultant_id),
        "role":                   "doctor",
        "iss":                    DOCTOR_TOKEN_ISSUER,
        "aud":                    DOCTOR_TOKEN_AUDIENCE,
        "iat":                    now,
        "exp":                    now + DOCTOR_TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, DOCTOR_TOKEN_SECRET, algorithm="HS256")


def decode_doctor_session_token(token: str) -> DoctorSessionClaims:
    """Verifies signature/exp/iss/aud/iat — same fail-closed posture as
    decode_patient_session_token(). No jti/replay check by design (see
    module note above): a doctor token is meant to be reusable for
    reconnects within its TTL."""
    if not DOCTOR_TOKEN_SECRET:
        raise HTTPException(500, "DOCTOR_SESSION_TOKEN_SECRET is not set on this server")
    if not token:
        raise HTTPException(401, "Missing session token")

    try:
        payload = jwt.decode(
            token,
            DOCTOR_TOKEN_SECRET,
            algorithms=["HS256"],
            issuer=DOCTOR_TOKEN_ISSUER,
            audience=DOCTOR_TOKEN_AUDIENCE,
            options={"require": ["exp", "iss", "aud", "iat", "session_id", "role"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Watch link has expired — ask to reopen the session from MedNova Care")
    except jwt.InvalidIssuerError:
        raise HTTPException(401, "Invalid doctor session token issuer")
    except jwt.InvalidAudienceError:
        raise HTTPException(401, "Invalid doctor session token audience")
    except jwt.MissingRequiredClaimError as exc:
        raise HTTPException(401, f"Doctor session token is missing required claim: {exc.claim}")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid doctor session token")

    iat = payload.get("iat")
    now = time.time()
    if iat > now + JWT_CLOCK_SKEW_SECONDS:
        raise HTTPException(401, "Doctor session token has an iat in the future")

    if payload.get("role") != "doctor":
        raise HTTPException(401, "Token does not carry a doctor role claim")

    return DoctorSessionClaims(
        session_id=str(payload["session_id"]),
        mednova_consultant_id=str(payload.get("mednova_consultant_id", "")),
        role="doctor",
    )


def verify_webhook_secret(x_webhook_secret: str = Header(None)) -> None:
    """
    FastAPI dependency for the Laravel -> thero patient-sync webhook.
    This is a SEPARATE shared secret from the therapist JWT — Laravel's
    backend calls this directly, there's no logged-in therapist involved.
    """
    if not WEBHOOK_SECRET:
        raise HTTPException(500, "MEDNOVA_WEBHOOK_SECRET is not set on this server")
    if not x_webhook_secret or x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(401, "Invalid webhook secret")


def results_webhook_headers() -> dict:
    """
    Item 25/27 prep: builds the auth header thero attaches to its OUTBOUND
    session-results webhook call to Laravel (opposite direction from
    verify_webhook_secret above — that one checks an INBOUND header,
    this one builds an OUTBOUND one).

    Not wired into anything yet — item 27's webhook sender will call this
    when constructing the request. Kept here (next to the inbound
    counterpart) rather than in the not-yet-built item 27 module, so both
    directions' auth live in one place and are easy to compare.

    Raises at call time (not at import time) if the secret isn't
    configured, same fail-closed posture as verify_webhook_secret — a
    session-results webhook silently going out with no auth header would
    be worse than one that fails loudly and gets retried/alerted on by
    item 27's retry logic.

    NOTE (open question for Nada — flagged, not resolved): her message
    said "I'll generate it with the public key and send both together."
    That phrasing could mean either (a) a plain shared secret sent to us
    over an already-secure channel, or (b) an RSA keypair where thero
    signs outgoing webhook payloads and Laravel verifies with a public
    key — a materially different scheme (X-Webhook-Secret header vs.
    a signature header like X-Webhook-Signature). This function assumes
    (a), the same shared-secret-header pattern as the inbound webhook,
    since that's the simpler/existing pattern in this codebase. Confirm
    with Nada before item 27 ships — if it's actually (b), this function
    and its call sites need reworking to sign rather than just attach a
    static secret.
    """
    if not RESULTS_WEBHOOK_SECRET:
        raise RuntimeError("THERO_RESULTS_WEBHOOK_SECRET is not set on this server")
    return {"X-Webhook-Secret": RESULTS_WEBHOOK_SECRET}


def schedule_webhook_headers() -> dict:
    """
    Auth header for the OUTBOUND "session scheduled" webhook (thero ->
    Laravel), fired from telehealth.create_room() so Laravel can route the
    join_url to the correct patient's MedNova account. Own secret, same
    fail-closed-at-call-time posture as results_webhook_headers() above —
    see that function's docstring for why this isn't checked at import
    time. Laravel's receiving endpoint doesn't exist yet as of this
    writing ("inime than build pannuvanga") — get the real secret from
    them once it does, same as MEDNOVA_SCHEDULE_WEBHOOK_URL.
    """
    if not SCHEDULE_WEBHOOK_SECRET:
        raise RuntimeError("MEDNOVA_SCHEDULE_WEBHOOK_SECRET is not set on this server")
    return {"X-Webhook-Secret": SCHEDULE_WEBHOOK_SECRET}