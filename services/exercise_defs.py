"""
Stable exercise machine codes — replaces substring matching on the
free-text `exercise_type` display name (the old `"shoulder" in ex.lower()`
pattern scattered across metrics.py / mjpeg_camera.py / camera_ws.py).

Item on Nada's list: "Stable machine codes for exercises instead of
substring matching on display names."

WHY substring matching was a real bug, not just style:
  - "Balance Exercise" contains no joint keyword, so it fell through
    every substring check to whatever the `else` branch matched.
  - "Hand Grip" vs "Hand Rehab" both contain "hand" — grip only detected
    correctly because "grip" happened to also be checked first, order-
    dependent and fragile.
  - Any therapist-typed display-name variant (e.g. "L Shoulder Flexion",
    "shoulder raise (post-op)") could silently mismatch and fall into the
    generic elbow/knee fallback in compute_primary_angle().

Design:
  - EXERCISE_DEFS is keyed by a stable machine `code` (upper-snake-case,
    never shown to the patient, never renamed once shipped — MedNova's
    `exercise_type` JWT/webhook claim should carry this code, not the
    display name).
  - `resolve_exercise_code()` is the single place that turns whatever
    string arrives (a code OR a legacy free-text display name, for
    backward compat with existing session_data / Setting rows / demo
    data) into a canonical code. Everything else in the codebase should
    call this instead of doing its own `.lower()` / `in` check.

IMPORTANT — normal_range_deg values below are generic physiotherapy
textbook reference ranges (standard active-ROM figures), used only as a
fallback default. They are NOT clinically verified for MedNova's patient
population and should NOT be treated as authoritative — this is exactly
the open question on Nada's list ("what normal range"). Per-session
target_rom (already a field the therapist sets — see
SessionMetrics.set_exercise_state / TelehealthRoom / JWT target_angle)
always overrides this default and is what actually drives rep counting.
Flag NEEDS_CLINICAL_REVIEW is on the module so this isn't silently
forgotten.
"""
from typing import Dict, Optional, TypedDict


NEEDS_CLINICAL_REVIEW = True  # normal_range_deg defaults below are unverified — confirm with Nada/clinical lead before relying on them for anything patient-facing


class ExerciseDef(TypedDict):
    code: str
    display_name: str          # current default display name (UI can still relabel)
    joint: str
    movement: str
    uses_hands: bool           # True → MediaPipe Hands pipeline, False → Pose
    normal_range_deg: Optional[tuple]  # (min, max) — UNVERIFIED, see module docstring


EXERCISE_DEFS: Dict[str, ExerciseDef] = {
    "SHOULDER_FLEXION": {
        "code": "SHOULDER_FLEXION",
        "display_name": "Shoulder Rehab",
        "joint": "shoulder",
        "movement": "flexion / abduction",
        "uses_hands": False,
        "normal_range_deg": (0, 180),
    },
    "ELBOW_FLEXION": {
        "code": "ELBOW_FLEXION",
        "display_name": "Elbow Flexion",
        "joint": "elbow",
        "movement": "flexion / extension",
        "uses_hands": False,
        "normal_range_deg": (0, 150),
    },
    "HIP_FLEXION": {
        "code": "HIP_FLEXION",
        "display_name": "Hip Flexion",
        "joint": "hip",
        "movement": "flexion / extension",
        "uses_hands": False,
        "normal_range_deg": (0, 120),
    },
    "KNEE_FLEXION": {
        "code": "KNEE_FLEXION",
        "display_name": "Knee Flexion",
        "joint": "knee",
        "movement": "flexion / extension",
        "uses_hands": False,
        "normal_range_deg": (0, 135),
    },
    "ANKLE_DORSIFLEXION": {
        "code": "ANKLE_DORSIFLEXION",
        "display_name": "Ankle Rehab",
        "joint": "ankle",
        "movement": "dorsiflexion / plantarflexion",
        "uses_hands": False,
        "normal_range_deg": (0, 45),
    },
    "WRIST_REHAB": {
        "code": "WRIST_REHAB",
        "display_name": "Hand Rehab",
        "joint": "wrist",
        "movement": "elbow-to-wrist articulation (pose-based, pre-grip)",
        "uses_hands": False,
        "normal_range_deg": (0, 90),
    },
    "BALANCE": {
        "code": "BALANCE",
        "display_name": "Balance Exercise",
        "joint": "whole-body",
        "movement": "static / dynamic balance",
        "uses_hands": False,
        "normal_range_deg": None,  # not an angle-based exercise
    },
    "HAND_GRIP": {
        "code": "HAND_GRIP",
        "display_name": "Hand Grip",
        "joint": "fingers (index/middle/ring/pinky — thumb excluded, see metrics.compute_grip_angle)",
        "movement": "grip / finger flexion",
        "uses_hands": True,
        "normal_range_deg": (40, 175),
    },
}

DEFAULT_CODE = "SHOULDER_FLEXION"

# Legacy display-name → code map, for backward compat with existing
# session_data JSON blobs, the `Setting` table, demo seed data in app.py,
# and any therapist-typed free text already in the DB. New callers should
# send a code directly; this map is a bridge, not the long-term contract.
_LEGACY_DISPLAY_NAME_MAP = {
    "shoulder rehab":    "SHOULDER_FLEXION",
    "arm raise":         "SHOULDER_FLEXION",
    "elbow flexion":     "ELBOW_FLEXION",
    "hip flexion":       "HIP_FLEXION",
    "knee flexion":      "KNEE_FLEXION",
    "leg extension":     "KNEE_FLEXION",
    "squat":             "KNEE_FLEXION",
    "ankle rehab":       "ANKLE_DORSIFLEXION",
    "ankle flexion":     "ANKLE_DORSIFLEXION",
    "hand rehab":        "WRIST_REHAB",
    "balance exercise":  "BALANCE",
    "hand grip":         "HAND_GRIP",
    "finger flexion":    "HAND_GRIP",
    "grip strength":     "HAND_GRIP",
}


def resolve_exercise_code(raw: Optional[str]) -> str:
    """
    Turn whatever arrives on the wire (a machine code, a legacy display
    name, or free text a therapist typed) into a canonical code.

    Resolution order:
      1. Exact code match (case-insensitive) — e.g. "KNEE_FLEXION"
      2. Exact legacy display-name match — e.g. "Knee Flexion"
      3. Substring fallback over the legacy map's keys — kept ONLY so an
         un-migrated free-text value ("Left Knee Flexion — post-op") still
         resolves sensibly instead of hard-failing. This is the same
         fuzzy matching we're trying to move away from, but as a last
         resort instead of the primary mechanism it used to be.
      4. DEFAULT_CODE — never raises, callers always get a valid code.
    """
    if not raw:
        return DEFAULT_CODE

    key = raw.strip().upper().replace(" ", "_").replace("-", "_")
    if key in EXERCISE_DEFS:
        return key

    lowered = raw.strip().lower()
    if lowered in _LEGACY_DISPLAY_NAME_MAP:
        return _LEGACY_DISPLAY_NAME_MAP[lowered]

    for name_fragment, code in _LEGACY_DISPLAY_NAME_MAP.items():
        if name_fragment in lowered:
            return code

    return DEFAULT_CODE


def get_def(code: str) -> ExerciseDef:
    """Definition lookup with fallback — never raises."""
    return EXERCISE_DEFS.get(resolve_exercise_code(code), EXERCISE_DEFS[DEFAULT_CODE])


def is_hand_exercise(exercise_type: Optional[str]) -> bool:
    """Replaces the old `"grip" in ex or "finger" in ex` checks in
    camera_ws.py and mjpeg_camera.py."""
    return get_def(exercise_type)["uses_hands"]


def get_display_name(code: str) -> str:
    return get_def(code)["display_name"]


def get_default_range(code: str) -> Optional[tuple]:
    """UNVERIFIED default — see module docstring. Session/therapist-set
    target_rom should always be preferred over this when available."""
    return get_def(code)["normal_range_deg"]


def all_codes() -> list:
    """For frontend dropdowns / API responses that need the full list."""
    return list(EXERCISE_DEFS.keys())