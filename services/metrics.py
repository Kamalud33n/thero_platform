import threading
import collections
import statistics
from typing import Dict, Optional

# Active exercise / target ROM state 
_exercise_lock = threading.Lock()
_current_exercise_type = "Shoulder Rehab"
_current_target_rom = 90.0

# Affected side — "left" | "right" | "both" (default). Per Nada's
# requirement: "Both sides scored and stored, with affected_side treated
# as primary." Kept as separate state (not folded into exercise_type)
# since it can change independently — same exercise, different patient/limb.
_affected_side_lock = threading.Lock()
_current_affected_side: Optional[str] = "both"

# Rep counting 
_rep_lock = threading.Lock()
_rep_count = 0
_rep_stage: Optional[str] = None      # "up" / "down"

# Stability — hip-center landmark jitter over a rolling window 
_stability_lock = threading.Lock()
_landmark_jitter_buffer: "collections.deque" = collections.deque(maxlen=20)
_current_stability_score = 100.0

# Smoothness — frame-to-frame angular jerk variance 
# Jerky/shaky movement produces large, erratic frame-to-frame angle deltas;
# a controlled rep produces small, consistent deltas. Low delta-variance =
# high smoothness.
_smoothness_lock = threading.Lock()
_angle_velocity_buffer: "collections.deque" = collections.deque(maxlen=15)
_last_primary_angle: Optional[float] = None
_current_smoothness_score = 100.0

# Balance — shoulder-midpoint lateral sway
# Distinct signal from hip-based stability above: tracks horizontal drift of
# the shoulder midpoint, which picks up upper-body swaying/compensation that
# hip jitter alone wouldn't catch.
_balance_lock = threading.Lock()
_shoulder_sway_buffer: "collections.deque" = collections.deque(maxlen=20)
_current_balance_score = 100.0

# Fatigue — rep-quality decline over the session 
# Each completed rep's peak angle (as % of target ROM) is logged. Fatigue is
# derived from how much rep quality has dropped in the second half of the
# session vs. the first half — a real physiological proxy (form degrades as
# the patient tires), not a random number or a copy of another metric.
_fatigue_lock = threading.Lock()
_rep_quality_buffer: "collections.deque" = collections.deque(maxlen=30)
_current_fatigue_score = 0.0
_last_fatigue_rep_count = 0


# Exercise state getters/setters 
def set_exercise_state(exercise_type: Optional[str] = None, target_rom: Optional[float] = None):
    global _current_exercise_type, _current_target_rom
    if exercise_type:
        from services.exercise_defs import resolve_exercise_code
        with _exercise_lock:
            # Store the canonical machine code, not whatever free text /
            # legacy display name arrived — every downstream reader
            # (compute_primary_angle, mjpeg _get_active_connections, etc.)
            # can then rely on get_exercise_state() always returning a code.
            _current_exercise_type = resolve_exercise_code(exercise_type)
    if target_rom is not None:
        try:
            rom = float(target_rom)
        except (TypeError, ValueError):
            return
        with _exercise_lock:
            _current_target_rom = rom


def get_exercise_state():
    with _exercise_lock:
        return _current_exercise_type, _current_target_rom


_VALID_SIDES = ("left", "right", "both")


def set_affected_side(side: Optional[str]):
    """side: "left" | "right" | "both" (case-insensitive). Anything else
    (None, unrecognized string) is ignored — keeps whatever was set
    before rather than silently resetting to a wrong default mid-session."""
    global _current_affected_side
    if not side:
        return
    normalized = side.strip().lower()
    if normalized not in _VALID_SIDES:
        return
    with _affected_side_lock:
        _current_affected_side = normalized


def get_affected_side() -> str:
    with _affected_side_lock:
        return _current_affected_side or "both"


# Score getters (thread-safe reads) 
def get_stability() -> float:
    with _stability_lock:
        return _current_stability_score


def get_smoothness() -> float:
    with _smoothness_lock:
        return _current_smoothness_score


def get_balance() -> float:
    with _balance_lock:
        return _current_balance_score


def get_current_fatigue() -> float:
    with _fatigue_lock:
        return _current_fatigue_score


def get_accuracy() -> float:
    """
    Live accuracy score — rolling average of per-rep quality (peak angle
    as a % of target ROM, 0-100) across the current _rep_quality_buffer.
    This is the SAME real per-rep signal _record_rep_quality() already
    computes for the fatigue calculation (see below) — fatigue looks at
    the DECLINE between the first half and second half of that buffer,
    accuracy is just the buffer's own average, exposed as its own metric
    instead of only feeding into fatigue.

    Real signal, not a placeholder: 0.0 until at least one rep has been
    recorded (empty buffer), same "no data yet" convention as the other
    live scores default to 100.0/0.0 before their windows fill.
    """
    with _fatigue_lock:
        if not _rep_quality_buffer:
            return 0.0
        return round(sum(_rep_quality_buffer) / len(_rep_quality_buffer), 1)


def compute_primary_angle(
    angles: Dict[str, float],
    exercise_type: str,
    affected_side: Optional[str] = None,
) -> Optional[float]:
    """Pick the joint-angle relevant to the active exercise, using the
    stable exercise code (services/exercise_defs.py) instead of substring
    matching on the free-text display name.

    affected_side: "left" | "right" | "both" | None. When set to "left" or
    "right", that side alone drives rep counting / primary_angle — per
    Nada's "affected_side treated as primary" requirement. When None or
    "both" (or a side that has no reading this frame), both sides are
    averaged as before, so existing callers that don't pass affected_side
    keep the old behavior.
    """
    from services.exercise_defs import resolve_exercise_code

    def avg(*vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def pick(l_key: str, r_key: str):
        side = (affected_side or "").strip().lower()
        if side == "left" and angles.get(l_key) is not None:
            return angles.get(l_key)
        if side == "right" and angles.get(r_key) is not None:
            return angles.get(r_key)
        # side unset, "both", or the affected side isn't visible this
        # frame (occlusion) — fall back to averaging both sides.
        return avg(angles.get(l_key), angles.get(r_key))

    code = resolve_exercise_code(exercise_type)

    if code == "SHOULDER_FLEXION":
        return pick("l_shoulder", "r_shoulder")
    if code == "ELBOW_FLEXION":
        return pick("l_elbow", "r_elbow")
    if code == "KNEE_FLEXION":
        return pick("l_knee", "r_knee")
    if code == "HIP_FLEXION":
        return pick("l_hip", "r_hip")
    if code == "ANKLE_DORSIFLEXION":
        return pick("l_ankle", "r_ankle")
    if code == "HAND_GRIP":
        # Thumb excluded on purpose — see compute_grip_angle() docstring.
        return avg(angles.get("index"), angles.get("middle"),
                    angles.get("ring"), angles.get("pinky"))
    # WRIST_REHAB / BALANCE / anything else without a dedicated angle:
    # same generic fallback as before.
    return avg(angles.get("l_elbow"), angles.get("r_elbow"),
               angles.get("l_knee"), angles.get("r_knee"))


def compute_finger_curl_angles(hand_landmarks) -> Dict[str, float]:
    """Real per-finger curl angle from MediaPipe Hands landmarks (21 pts).
    ~170-180° = finger fully extended (open hand), ~40-70° = fully curled
    (closed fist). Thumb uses CMC-MCP-IP since its joint layout differs
    from the other four fingers."""
    from config import get_angle, HAND_LANDMARKS as HL

    def pt(name):
        return hand_landmarks[HL[name]]

    angles: Dict[str, float] = {}
    try:
        angles["thumb"]  = round(get_angle(pt("thumb_cmc"),  pt("thumb_mcp"),  pt("thumb_ip")), 1)
        angles["index"]  = round(get_angle(pt("index_mcp"),  pt("index_pip"),  pt("index_dip")), 1)
        angles["middle"] = round(get_angle(pt("middle_mcp"), pt("middle_pip"), pt("middle_dip")), 1)
        angles["ring"]   = round(get_angle(pt("ring_mcp"),   pt("ring_pip"),   pt("ring_dip")), 1)
        angles["pinky"]  = round(get_angle(pt("pinky_mcp"),  pt("pinky_pip"),  pt("pinky_dip")), 1)
    except Exception as e:
        print(f"finger curl angle calculation failed: {e}")
    return angles


def compute_grip_angle(finger_angles: Dict[str, float]) -> Optional[float]:
    """Average curl across the 4 main fingers (thumb excluded — its range
    of motion/joint geometry differs enough that mixing it in would skew
    the open/close signal used for rep counting). This is the primary_angle
    fed into update_rep_count() for Hand Grip / Finger Flexion exercises,
    same as an elbow/knee angle is for other exercise types."""
    vals = [finger_angles.get(f) for f in ("index", "middle", "ring", "pinky")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def update_rep_count(primary_angle: Optional[float], target_rom: float) -> int:
    """Hysteresis-based rep counter: counts one rep per full down→up cycle."""
    global _rep_count, _rep_stage
    if primary_angle is None or target_rom <= 0:
        with _rep_lock:
            return _rep_count

    high_thresh = target_rom * 0.80
    low_thresh  = target_rom * 0.35

    with _rep_lock:
        if primary_angle >= high_thresh:
            if _rep_stage == "down":
                _rep_count += 1
            _rep_stage = "up"
        elif primary_angle <= low_thresh:
            _rep_stage = "down"
        return _rep_count


def update_stability(landmarks) -> float:
    """Real stability score from hip-center landmark jitter over a short
    rolling window — steadier patient = less x/y variance = higher score."""
    global _current_stability_score
    try:
        hip_x = (landmarks[23].x + landmarks[24].x) / 2
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
    except Exception:
        return _current_stability_score

    with _stability_lock:
        _landmark_jitter_buffer.append((hip_x, hip_y))
        if len(_landmark_jitter_buffer) >= 5:
            xs = [p[0] for p in _landmark_jitter_buffer]
            ys = [p[1] for p in _landmark_jitter_buffer]
            jitter = statistics.pstdev(xs) + statistics.pstdev(ys)
            # jitter is in normalized [0,1] frame coords; scale empirically to 0-100
            score = max(0.0, min(100.0, 100.0 - jitter * 4000))
            _current_stability_score = round(score, 1)
        return _current_stability_score


def update_smoothness(primary_angle: Optional[float]) -> float:
    global _current_smoothness_score, _last_primary_angle
    if primary_angle is None:
        return _current_smoothness_score

    with _smoothness_lock:
        if _last_primary_angle is not None:
            delta = abs(primary_angle - _last_primary_angle)
            _angle_velocity_buffer.append(delta)
        _last_primary_angle = primary_angle

        if len(_angle_velocity_buffer) >= 5:
            jerk_variance = statistics.pstdev(_angle_velocity_buffer)
            # empirically scaled: ~0-2°/frame deltas (smooth) -> near 100,
            # large erratic swings -> drops toward 0
            score = max(0.0, min(100.0, 100.0 - jerk_variance * 8))
            _current_smoothness_score = round(score, 1)
        return _current_smoothness_score


def update_balance(landmarks) -> float:
    global _current_balance_score
    try:
        sh_x = (landmarks[11].x + landmarks[12].x) / 2
    except Exception:
        return _current_balance_score

    with _balance_lock:
        _shoulder_sway_buffer.append(sh_x)
        if len(_shoulder_sway_buffer) >= 5:
            sway = statistics.pstdev(_shoulder_sway_buffer)
            # normalized [0,1] frame coords; scale empirically to 0-100
            score = max(0.0, min(100.0, 100.0 - sway * 5000))
            _current_balance_score = round(score, 1)
        return _current_balance_score


def _record_rep_quality(peak_angle: Optional[float], target_rom: float):
    """Called once per completed rep. Logs how close that rep's peak angle
    got to the target ROM, then derives fatigue from the drop-off between
    the first half and second half of the session's rep quality — real
    physiological signal (form degrades as the patient tires), not a
    random number or an inverted copy of the stability score."""
    global _current_fatigue_score
    if peak_angle is None or target_rom <= 0:
        return

    quality = max(0.0, min(100.0, (peak_angle / target_rom) * 100))
    with _fatigue_lock:
        _rep_quality_buffer.append(quality)
        n = len(_rep_quality_buffer)
        if n >= 4:
            half   = n // 2
            buf    = list(_rep_quality_buffer)
            early  = buf[:half]
            recent = buf[half:]
            early_avg  = sum(early)  / len(early)
            recent_avg = sum(recent) / len(recent)
            decline = max(0.0, early_avg - recent_avg)  # >0 if quality dropped
            # scale a 0-40pt quality drop across the session to 0-100 fatigue
            _current_fatigue_score = round(min(100.0, decline * 2.5), 1)


def maybe_record_rep_quality(reps: int, primary_angle: Optional[float], target_rom: float) -> float:
    """Call once per frame with the latest rep count. Records rep quality
    exactly once per newly-completed rep (mirrors the original inline
    `if reps > _last_fatigue_rep_count` check), then returns the current
    fatigue score."""
    global _last_fatigue_rep_count
    with _fatigue_lock:
        already_recorded = reps <= _last_fatigue_rep_count
    if not already_recorded:
        _record_rep_quality(primary_angle, target_rom)
        with _fatigue_lock:
            _last_fatigue_rep_count = reps
    return get_current_fatigue()


class SessionMetrics:
    """Per-connection version of the module-level state/functions above.

    Cloud/concurrency refactor (Phase A): the WS pipeline (services/camera_ws.py)
    gives every WebSocket connection its own CameraManager + SessionMetrics
    instance instead of touching the shared globals above, so N patients can
    run sessions concurrently on one server process without rep counts /
    scores bleeding into each other.

    The MJPEG pipeline (services/mjpeg_camera.py) intentionally keeps using
    the module-level globals/functions above unchanged — it's being kept as
    a local-only, single-camera pipeline (see refactor plan Phase C), so it
    was never at risk of cross-session contamination and doesn't need this.
    """

    def __init__(self):
        self._exercise_lock = threading.Lock()
        self._current_exercise_type = "Shoulder Rehab"
        self._current_target_rom = 90.0

        self._affected_side_lock = threading.Lock()
        self._current_affected_side: Optional[str] = "both"

        self._rep_lock = threading.Lock()
        self._rep_count = 0
        self._rep_stage: Optional[str] = None

        self._stability_lock = threading.Lock()
        self._landmark_jitter_buffer: "collections.deque" = collections.deque(maxlen=20)
        self._current_stability_score = 100.0

        self._smoothness_lock = threading.Lock()
        self._angle_velocity_buffer: "collections.deque" = collections.deque(maxlen=15)
        self._last_primary_angle: Optional[float] = None
        self._current_smoothness_score = 100.0

        self._balance_lock = threading.Lock()
        self._shoulder_sway_buffer: "collections.deque" = collections.deque(maxlen=20)
        self._current_balance_score = 100.0

        self._fatigue_lock = threading.Lock()
        self._rep_quality_buffer: "collections.deque" = collections.deque(maxlen=30)
        self._current_fatigue_score = 0.0
        self._last_fatigue_rep_count = 0

    # Exercise state
    def set_exercise_state(self, exercise_type: Optional[str] = None, target_rom: Optional[float] = None):
        if exercise_type:
            from services.exercise_defs import resolve_exercise_code
            with self._exercise_lock:
                # Canonical code stored, not raw text — see module-level
                # set_exercise_state() for why.
                self._current_exercise_type = resolve_exercise_code(exercise_type)
        if target_rom is not None:
            try:
                rom = float(target_rom)
            except (TypeError, ValueError):
                return
            with self._exercise_lock:
                self._current_target_rom = rom

    def get_exercise_state(self):
        with self._exercise_lock:
            return self._current_exercise_type, self._current_target_rom

    def set_affected_side(self, side: Optional[str]):
        if not side:
            return
        normalized = side.strip().lower()
        if normalized not in _VALID_SIDES:
            return
        with self._affected_side_lock:
            self._current_affected_side = normalized

    def get_affected_side(self) -> str:
        with self._affected_side_lock:
            return self._current_affected_side or "both"

    # Score getters
    def get_stability(self) -> float:
        with self._stability_lock:
            return self._current_stability_score

    def get_smoothness(self) -> float:
        with self._smoothness_lock:
            return self._current_smoothness_score

    def get_balance(self) -> float:
        with self._balance_lock:
            return self._current_balance_score

    def get_current_fatigue(self) -> float:
        with self._fatigue_lock:
            return self._current_fatigue_score

    def get_accuracy(self) -> float:
        """Per-connection version of the module-level get_accuracy() above
        — same rolling rep-quality-buffer average, scoped to this
        connection's own SessionMetrics instance."""
        with self._fatigue_lock:
            if not self._rep_quality_buffer:
                return 0.0
            return round(sum(self._rep_quality_buffer) / len(self._rep_quality_buffer), 1)

    def get_rep_count(self) -> int:
        """Thread-safe read of the current rep count without needing a new
        primary_angle sample (unlike update_rep_count(), which requires
        one). Used by routers/ws.py's disconnect handler to persist
        whatever count was reached when the connection dropped mid-session."""
        with self._rep_lock:
            return self._rep_count

    # Rep counting (same hysteresis logic as update_rep_count() above)
    def update_rep_count(self, primary_angle: Optional[float], target_rom: float) -> int:
        if primary_angle is None or target_rom <= 0:
            with self._rep_lock:
                return self._rep_count

        high_thresh = target_rom * 0.80
        low_thresh  = target_rom * 0.35

        with self._rep_lock:
            if primary_angle >= high_thresh:
                if self._rep_stage == "down":
                    self._rep_count += 1
                self._rep_stage = "up"
            elif primary_angle <= low_thresh:
                self._rep_stage = "down"
            return self._rep_count

    def update_stability(self, landmarks) -> float:
        try:
            hip_x = (landmarks[23].x + landmarks[24].x) / 2
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
        except Exception:
            return self._current_stability_score

        with self._stability_lock:
            self._landmark_jitter_buffer.append((hip_x, hip_y))
            if len(self._landmark_jitter_buffer) >= 5:
                xs = [p[0] for p in self._landmark_jitter_buffer]
                ys = [p[1] for p in self._landmark_jitter_buffer]
                jitter = statistics.pstdev(xs) + statistics.pstdev(ys)
                score = max(0.0, min(100.0, 100.0 - jitter * 4000))
                self._current_stability_score = round(score, 1)
            return self._current_stability_score

    def update_smoothness(self, primary_angle: Optional[float]) -> float:
        if primary_angle is None:
            return self._current_smoothness_score

        with self._smoothness_lock:
            if self._last_primary_angle is not None:
                delta = abs(primary_angle - self._last_primary_angle)
                self._angle_velocity_buffer.append(delta)
            self._last_primary_angle = primary_angle

            if len(self._angle_velocity_buffer) >= 5:
                jerk_variance = statistics.pstdev(self._angle_velocity_buffer)
                score = max(0.0, min(100.0, 100.0 - jerk_variance * 8))
                self._current_smoothness_score = round(score, 1)
            return self._current_smoothness_score

    def update_balance(self, landmarks) -> float:
        try:
            sh_x = (landmarks[11].x + landmarks[12].x) / 2
        except Exception:
            return self._current_balance_score

        with self._balance_lock:
            self._shoulder_sway_buffer.append(sh_x)
            if len(self._shoulder_sway_buffer) >= 5:
                sway = statistics.pstdev(self._shoulder_sway_buffer)
                score = max(0.0, min(100.0, 100.0 - sway * 5000))
                self._current_balance_score = round(score, 1)
            return self._current_balance_score

    def _record_rep_quality(self, peak_angle: Optional[float], target_rom: float):
        if peak_angle is None or target_rom <= 0:
            return
        quality = max(0.0, min(100.0, (peak_angle / target_rom) * 100))
        with self._fatigue_lock:
            self._rep_quality_buffer.append(quality)
            n = len(self._rep_quality_buffer)
            if n >= 4:
                half   = n // 2
                buf    = list(self._rep_quality_buffer)
                early  = buf[:half]
                recent = buf[half:]
                early_avg  = sum(early)  / len(early)
                recent_avg = sum(recent) / len(recent)
                decline = max(0.0, early_avg - recent_avg)
                self._current_fatigue_score = round(min(100.0, decline * 2.5), 1)

    def maybe_record_rep_quality(self, reps: int, primary_angle: Optional[float], target_rom: float) -> float:
        with self._fatigue_lock:
            already_recorded = reps <= self._last_fatigue_rep_count
        if not already_recorded:
            self._record_rep_quality(primary_angle, target_rom)
            with self._fatigue_lock:
                self._last_fatigue_rep_count = reps
        return self.get_current_fatigue()

    def reset(self):
        """Call right before a session starts so rep count + stability/
        smoothness/balance/fatigue buffers don't carry over stale data from
        a previous session on this same connection."""
        with self._rep_lock:
            self._rep_count = 0
            self._rep_stage = None
        with self._stability_lock:
            self._landmark_jitter_buffer.clear()
            self._current_stability_score = 100.0
        with self._smoothness_lock:
            self._angle_velocity_buffer.clear()
            self._last_primary_angle = None
            self._current_smoothness_score = 100.0
        with self._balance_lock:
            self._shoulder_sway_buffer.clear()
            self._current_balance_score = 100.0
        with self._fatigue_lock:
            self._rep_quality_buffer.clear()
            self._current_fatigue_score = 0.0
            self._last_fatigue_rep_count = 0


def reset_state():
    """Call right before a session starts so rep count + stability/smoothness/
    balance/fatigue buffers don't carry over stale data from a previous
    session/patient."""
    global _rep_count, _rep_stage, _current_stability_score
    global _current_smoothness_score, _current_balance_score
    global _current_fatigue_score, _last_primary_angle, _last_fatigue_rep_count
    with _rep_lock:
        _rep_count = 0
        _rep_stage = None
    with _stability_lock:
        _landmark_jitter_buffer.clear()
        _current_stability_score = 100.0
    with _smoothness_lock:
        _angle_velocity_buffer.clear()
        _last_primary_angle = None
        _current_smoothness_score = 100.0
    with _balance_lock:
        _shoulder_sway_buffer.clear()
        _current_balance_score = 100.0
    with _fatigue_lock:
        _rep_quality_buffer.clear()
        _current_fatigue_score = 0.0
        _last_fatigue_rep_count = 0