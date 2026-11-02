"""
Exercise metrics (rep count, per-rep scores, stability, smoothness, balance,
fatigue, accuracy, average ROM).

Logic is ported 1:1 from NovaMotion (services/metrics.py MetricsState) so a
Thero session produces the same numbers as the in-clinic NovaMotion app:

  * every joint angle is converted to ROM (how far the joint MOVED from its
    neutral pose) before counting reps / scoring - see angle_to_rom()
  * reps are counted with a 3-sample median filter and a down-threshold of
    55% of target (or a 35%-of-target drop from the peak)
  * every rep is scored  0.6*range + 0.2*stability + 0.2*smoothness
  * stability / balance use a 1-second time window, smoothness is measured
    per 1/30 s, so scores do not depend on the server's frame rate
  * accuracy = average of the per-rep scores, average ROM = average peak ROM
    of the correct reps

All live state lives on a SessionMetrics instance (one per telehealth room /
WebSocket connection) so concurrent sessions never mix. The module-level
functions at the bottom delegate to one shared default instance and are only
kept for the local MJPEG pipeline (services/mjpeg_camera.py).

NOTE on left/right: unlike NovaMotion, Thero's WebSocket pipeline does NOT
mirror the frame before MediaPipe, so MediaPipe's l_* landmarks are already
the patient's real left side. affected_side is therefore used as-is (no
left/right inversion).
"""
import threading
import collections
import statistics
import time
from typing import Any, Dict, List, Optional

_VALID_SIDES = ("left", "right", "both")


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------
def _code(exercise_type: Optional[str]) -> str:
    from services.exercise_defs import resolve_exercise_code
    return resolve_exercise_code(exercise_type)


def angle_to_rom(angle: Optional[float], exercise_type: str) -> Optional[float]:
    """Convert a raw joint angle into ROM = how far the joint has MOVED from
    its neutral pose, in degrees (0..180).

      * SHOULDER_FLEXION : hip-shoulder-elbow angle already grows as the arm
                           goes up (arm down ~0-20deg) -> used as-is.
      * ANKLE_DORSIFLEXION : neutral (foot flat) ~90deg -> |angle - 90|.
      * everything else (elbow / knee / hip / wrist / hand grip): the joint
        is ~180deg when straight/open and shrinks as it flexes -> 180 - angle.

    Using the raw angle for elbow/knee/hip/grip is what made reps never count
    and ROM read ~100% while the limb was resting straight.
    """
    if angle is None:
        return None
    code = _code(exercise_type)
    if code == "SHOULDER_FLEXION":
        rom = angle
    elif code == "ANKLE_DORSIFLEXION":
        rom = abs(angle - 90.0)
    else:
        rom = 180.0 - angle
    return max(0.0, min(180.0, rom))


def compute_primary_angle(angles: Dict[str, float], exercise_type: str,
                          side: Optional[str] = "both") -> Optional[float]:
    """Primary movement value (ROM in degrees, see angle_to_rom) for the
    active exercise. For side "left"/"right" only that limb is used instead
    of averaging both. Returns None for BALANCE (no joint-angle signal) or
    when the needed joint is not visible."""
    code = _code(exercise_type)
    s = (side or "both").strip().lower()

    def avg(*vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def pick(l_key: str, r_key: str):
        if s == "left":
            return angles.get(l_key)
        if s == "right":
            return angles.get(r_key)
        return avg(angles.get(l_key), angles.get(r_key))

    if code == "BALANCE":
        return None

    if code == "HAND_GRIP":
        # Thumb excluded on purpose (its joint geometry differs).
        raw = avg(angles.get("index"), angles.get("middle"),
                  angles.get("ring"), angles.get("pinky"))
    elif code == "SHOULDER_FLEXION":
        raw = pick("l_shoulder", "r_shoulder")
    elif code == "ELBOW_FLEXION" or code == "WRIST_REHAB":
        # Pose-only hand/wrist rehab is tracked through the elbow-wrist chain.
        raw = pick("l_elbow", "r_elbow")
    elif code == "KNEE_FLEXION":
        raw = pick("l_knee", "r_knee")
    elif code == "HIP_FLEXION":
        raw = pick("l_hip", "r_hip")
    elif code == "ANKLE_DORSIFLEXION":
        raw = pick("l_ankle", "r_ankle")
    else:
        raw = pick("l_elbow", "r_elbow")

    return angle_to_rom(raw, exercise_type)


def tracked_joints(exercise_type: str, side: Optional[str] = "both") -> Dict[str, str]:
    """{angles-dict key: report label} for the joints that matter to this
    exercise (used for the per-rep joint range in rep_details)."""
    code = _code(exercise_type)
    s = (side or "both").strip().lower()
    if code == "BALANCE":
        return {}
    if code == "HAND_GRIP":
        return {"index": "Index finger", "middle": "Middle finger", "ring": "Ring finger",
                "pinky": "Pinky finger", "thumb": "Thumb"}
    joint = {
        "SHOULDER_FLEXION": "shoulder",
        "ELBOW_FLEXION": "elbow",
        "WRIST_REHAB": "elbow",
        "KNEE_FLEXION": "knee",
        "HIP_FLEXION": "hip",
        "ANKLE_DORSIFLEXION": "ankle",
    }.get(code, "elbow")
    out: Dict[str, str] = {}
    if s in ("left", "both"):
        out[f"l_{joint}"] = f"Left {joint}"
    if s in ("right", "both"):
        out[f"r_{joint}"] = f"Right {joint}"
    return out


def compute_finger_curl_angles(hand_landmarks) -> Dict[str, float]:
    """Real per-finger curl angle from MediaPipe Hands landmarks (21 pts).
    ~170-180 deg = finger fully extended (open hand), ~40-70 deg = fully
    curled (closed fist)."""
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
    """Average curl across the 4 main fingers (thumb excluded)."""
    vals = [finger_angles.get(f) for f in ("index", "middle", "ring", "pinky")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# Stability / balance look at the last second of movement and smoothness is
# measured per 1/30 s, so the scores do not depend on the server frame rate.
_WINDOW_S = 1.0
_SMOOTH_REF_DT = 1.0 / 30.0

# Per-rep score = 0.6 * range + 0.2 * stability + 0.2 * smoothness (all 0-100).
_REP_W_RANGE  = 0.6
_REP_W_STAB   = 0.2
_REP_W_SMOOTH = 0.2
# A rep "window" opens when ROM rises this far above the resting value, and an
# attempt that falls back near rest without reaching 80% of target is dropped.
_WIN_START_FRAC   = 0.15
_WIN_REST_FRAC    = 0.08
_INCOMPLETE_FRAC  = 0.30     # such an attempt must have reached 30% of target to be reported


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------
class SessionMetrics:
    """All live metric state for ONE session / connection."""

    def __init__(self):
        # Active exercise / target ROM / affected side
        self._exercise_lock = threading.Lock()
        self._current_exercise_type = "SHOULDER_FLEXION"
        self._current_target_rom = 90.0
        self._affected_side_lock = threading.Lock()
        self._current_affected_side: Optional[str] = "both"

        # Rep counting
        self._rep_lock = threading.Lock()
        self._rep_count = 0
        self._rep_stage: Optional[str] = None      # "up" / "down"
        self._rep_peak = 0.0
        self._rom_window: "collections.deque" = collections.deque(maxlen=3)

        # Per-rep scoring (all guarded by _rep_lock)
        self._rest_min: Optional[float] = None
        self._win_active = False
        self._win_counted = False
        self._win_peak = 0.0
        self._win_stab_sum = 0.0
        self._win_smooth_sum = 0.0
        self._win_bal_sum = 0.0
        self._win_n = 0
        self._rep_scores: List[float] = []
        self._rep_details: List[Dict[str, Any]] = []
        self._joint_recent: "collections.deque" = collections.deque(maxlen=10)
        self._win_joint: Dict[str, List[float]] = {}
        self._win_t0 = 0.0
        self._live_rep_score: Optional[float] = None
        self._incomplete_reps = 0

        # Stability - hip-center landmark jitter over a rolling window
        self._stability_lock = threading.Lock()
        self._landmark_jitter_buffer: "collections.deque" = collections.deque(maxlen=200)
        self._current_stability_score = 100.0

        # Smoothness - frame-to-frame angular jerk variance
        self._smoothness_lock = threading.Lock()
        self._angle_velocity_buffer: "collections.deque" = collections.deque(maxlen=15)
        self._last_primary_angle: Optional[float] = None
        self._last_angle_t: Optional[float] = None
        self._current_smoothness_score = 100.0

        # Balance - shoulder-midpoint lateral sway
        self._balance_lock = threading.Lock()
        self._shoulder_sway_buffer: "collections.deque" = collections.deque(maxlen=200)
        self._current_balance_score = 100.0

        # Fatigue - rep-quality decline over the session
        self._fatigue_lock = threading.Lock()
        self._rep_quality_buffer: "collections.deque" = collections.deque(maxlen=30)
        self._current_fatigue_score = 0.0
        self._last_fatigue_rep_count = 0

    # -- Exercise state ------------------------------------------------------
    def set_exercise_state(self, exercise_type: Optional[str] = None, target_rom: Optional[float] = None):
        if exercise_type:
            with self._exercise_lock:
                # Canonical machine code (services/exercise_defs.py)
                self._current_exercise_type = _code(exercise_type)
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

    # -- Score getters (thread-safe reads) -----------------------------------
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

    def get_rep_count(self) -> int:
        with self._rep_lock:
            return self._rep_count

    def get_accuracy(self) -> float:
        """Session accuracy = average of the per-rep scores (same as
        NovaMotion). 0.0 until the first rep is completed."""
        with self._rep_lock:
            scores = list(self._rep_scores)
            # A rep that already reached the target but whose arm has not come
            # back down yet is counted in reps, yet its score is only frozen
            # on the way down. When the room auto-completes on target reps the
            # session is saved right at that moment, so include that rep's
            # live score, otherwise the last rep would be missing from accuracy.
            if self._win_active and self._win_counted and self._live_rep_score is not None:
                scores.append(self._live_rep_score)
            if not scores:
                return 0.0
            return round(sum(scores) / len(scores), 1)

    def get_average_rom(self) -> float:
        """Average peak ROM (degrees) of the correctly completed reps
        (same as NovaMotion). 0.0 until the first rep is completed."""
        with self._rep_lock:
            roms = [d["rom"] for d in self._rep_details if d.get("done")]
            if self._win_active and self._win_counted:
                roms.append(round(self._win_peak, 1))   # rep reached target, not yet closed (see get_accuracy)
            if not roms:
                return 0.0
            return round(sum(roms) / len(roms), 1)

    def get_rep_scoring(self) -> Dict[str, Any]:
        """Per-rep results for the live page: final score of each completed
        rep, the live score of the rep in progress, and how many attempts fell
        back to rest without reaching the target."""
        with self._rep_lock:
            return {
                "rep_scores": list(self._rep_scores),
                "rep_details": [dict(d) for d in self._rep_details],
                "rep_live": self._live_rep_score,
                "rep_live_rom": round(self._win_peak, 1) if self._win_active else None,
                "rep_incomplete": self._incomplete_reps,
            }

    def get_rep_debug(self, target_rom: float) -> str:
        with self._rep_lock:
            med = statistics.median(self._rom_window) if self._rom_window else None
            return (f"median={med} stage={self._rep_stage} peak={round(self._rep_peak, 1)} "
                    f"count={self._rep_count} high={round(target_rom * 0.80, 1)} low={round(target_rom * 0.55, 1)}")

    # -- Rep counting + per-rep scoring --------------------------------------
    def update_rep_count(self, primary_angle: Optional[float], target_rom: float) -> int:
        """A rep is counted when ROM reaches >= 80% of target, then the joint
        comes back down: either to <= 55% of target, or by at least 35% of
        target below the peak just reached. A 3-sample median filters
        single-frame landmark spikes. The same pass scores each rep (see
        _REP_W_*). Call update_stability()/update_smoothness() every frame
        too: the rep window averages their live values."""
        if primary_angle is None or target_rom <= 0:
            with self._rep_lock:
                return self._rep_count

        high_thresh = target_rom * 0.80
        low_thresh  = target_rom * 0.55
        drop        = target_rom * 0.35

        with self._rep_lock:
            self._rom_window.append(primary_angle)
            value = statistics.median(self._rom_window)
            if self._rep_stage is None:
                self._rep_stage = "up" if value >= high_thresh else "down"
                self._rep_peak = value
                self._rest_min = value
            elif self._rep_stage == "down":
                if value >= high_thresh:
                    self._rep_count += 1
                    self._rep_stage = "up"
                    self._rep_peak = value
                    if not self._win_active:
                        self._open_window(value)
                    self._win_counted = True
                    self._window_add(value, target_rom)
                else:
                    self._track_attempt(value, target_rom)
            else:
                if value > self._rep_peak:
                    self._rep_peak = value
                if value < high_thresh and (value <= low_thresh or value <= self._rep_peak - drop):
                    self._rep_stage = "down"
                    self._rep_peak = 0.0
                    self._close_rep(target_rom)
                    self._rest_min = value
                else:
                    self._window_add(value, target_rom)
            return self._rep_count

    # (helpers below are called with _rep_lock held)
    def _open_window(self, value: float):
        self._win_active = True
        self._win_counted = False
        self._win_t0 = time.time()
        # the window opens a few frames AFTER the joint starts moving, so seed
        # the min/max from the last few frames before it
        self._win_joint = {}
        for snap in self._joint_recent:
            for label, v in snap.items():
                cur = self._win_joint.get(label)
                if cur is None:
                    self._win_joint[label] = [v, v]
                else:
                    cur[0] = min(cur[0], v); cur[1] = max(cur[1], v)
        self._joint_recent.clear()
        self._win_peak = value
        self._win_stab_sum = 0.0
        self._win_smooth_sum = 0.0
        self._win_bal_sum = 0.0
        self._win_n = 0

    def _window_score(self, target_rom: float) -> Optional[float]:
        if self._win_n <= 0 or target_rom <= 0:
            return None
        range_pct = min(100.0, self._win_peak / target_rom * 100.0)
        stab      = self._win_stab_sum / self._win_n
        smooth    = self._win_smooth_sum / self._win_n
        score = _REP_W_RANGE * range_pct + _REP_W_STAB * stab + _REP_W_SMOOTH * smooth
        return round(max(0.0, min(100.0, score)), 1)

    def _window_add(self, value: float, target_rom: float):
        if not self._win_active:
            return
        if value > self._win_peak:
            self._win_peak = value
        self._win_stab_sum += self._current_stability_score
        self._win_smooth_sum += self._current_smoothness_score
        self._win_bal_sum += self._current_balance_score
        self._win_n += 1
        live = self._window_score(target_rom)
        if live is not None:
            self._live_rep_score = live

    def _track_attempt(self, value: float, target_rom: float):
        """Joint is below the rep threshold while waiting for the next rep:
        decide when a rep window opens, and when an attempt that never reached
        the target is dropped."""
        if self._rest_min is None:
            self._rest_min = value
        if self._win_active:
            self._window_add(value, target_rom)
            if value <= self._rest_min + _WIN_REST_FRAC * target_rom:
                # back at rest without reaching the target -> incomplete attempt
                if self._win_peak >= _INCOMPLETE_FRAC * target_rom:
                    self._incomplete_reps += 1
                    self._rep_details.append(self._window_details(target_rom, done=False))
                self._win_active = False
                self._win_counted = False
                self._live_rep_score = self._rep_scores[-1] if self._rep_scores else None
                self._rest_min = min(self._rest_min, value)
            return
        self._rest_min = min(self._rest_min, value)
        if value >= self._rest_min + _WIN_START_FRAC * target_rom:
            self._open_window(value)
            self._window_add(value, target_rom)

    def _window_details(self, target_rom: float, done: bool = True) -> Dict[str, Any]:
        """One row for the report: the attempt that just ended."""
        n = max(1, self._win_n)
        rom = round(self._win_peak, 1)
        secs = max(0.1, time.time() - self._win_t0)
        range_pct = (self._win_peak / target_rom * 100.0) if target_rom > 0 else 0.0
        stab = self._win_stab_sum / n
        smooth = self._win_smooth_sum / n
        score = self._window_score(target_rom)
        if not done:
            fb = f"Wrong rep - reached only {range_pct:.0f}% of target"
        elif range_pct < 90:
            fb = "Raise higher"
        elif stab < 70:
            fb = "Keep body steady"
        elif smooth < 70:
            fb = "Move smoother"
        else:
            fb = "Good rep"
        joints = {name: round(hi - lo, 1) for name, (lo, hi) in self._win_joint.items()}
        # Hands-only frame has no body landmarks -> no balance signal.
        bal = None if self._current_exercise_type == "HAND_GRIP" else round(self._win_bal_sum / n, 1)
        return {"done": done, "score": score if score is not None else 0.0, "rom": rom,
                "speed": round(rom / secs, 1), "duration": round(secs, 1), "feedback": fb,
                "stability": round(stab, 1), "smoothness": round(smooth, 1), "balance": bal,
                "joints": joints}

    def _close_rep(self, target_rom: float):
        """The joint came back down after a counted rep: freeze its score."""
        if self._win_active and self._win_counted:
            score = self._window_score(target_rom)
            if score is not None:
                self._rep_scores.append(score)
                self._live_rep_score = score
                self._rep_details.append(self._window_details(target_rom))
        self._win_active = False
        self._win_counted = False

    def update_joint_angles(self, angles: Dict[str, float], exercise_type: str, side: Optional[str] = "both"):
        """Call every frame AFTER update_rep_count(). While a rep/attempt is in
        progress it keeps the lowest and highest angle of each tracked joint,
        so the report can show how far every joint moved in that rep."""
        if not angles:
            return
        tracked = tracked_joints(exercise_type, side)
        with self._rep_lock:
            if not self._win_active:
                snap = {label: angles[key] for key, label in tracked.items() if angles.get(key) is not None}
                if snap:
                    self._joint_recent.append(snap)
                return
            for key, label in tracked.items():
                v = angles.get(key)
                if v is None:
                    continue
                cur = self._win_joint.get(label)
                if cur is None:
                    self._win_joint[label] = [v, v]
                else:
                    if v < cur[0]: cur[0] = v
                    if v > cur[1]: cur[1] = v

    # -- Stability / smoothness / balance ------------------------------------
    def update_stability(self, landmarks) -> float:
        """Stability from hip-center landmark jitter over a short rolling
        window - steadier patient = less x/y variance = higher score."""
        try:
            hip_x = (landmarks[23].x + landmarks[24].x) / 2
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
        except Exception:
            return self._current_stability_score
        return self.update_stability_point(hip_x, hip_y)

    def update_stability_point(self, x: float, y: float) -> float:
        """Same jitter score for any tracked point (hand-grip exercises use
        the wrist landmark because a hands-only frame has no hips)."""
        with self._stability_lock:
            now = time.monotonic()
            buf = self._landmark_jitter_buffer
            buf.append((now, x, y))
            while buf and now - buf[0][0] > _WINDOW_S:
                buf.popleft()
            if len(buf) >= 5:
                xs = [p[1] for p in buf]
                ys = [p[2] for p in buf]
                jitter = statistics.pstdev(xs) + statistics.pstdev(ys)
                score = max(0.0, min(100.0, 100.0 - jitter * 4000))
                self._current_stability_score = round(score, 1)
            return self._current_stability_score

    def update_smoothness(self, primary_angle: Optional[float]) -> float:
        if primary_angle is None:
            return self._current_smoothness_score

        with self._smoothness_lock:
            now = time.monotonic()
            if self._last_primary_angle is not None and self._last_angle_t is not None:
                dt = min(0.5, max(0.02, now - self._last_angle_t))
                delta = abs(primary_angle - self._last_primary_angle) * (_SMOOTH_REF_DT / dt)
                self._angle_velocity_buffer.append(delta)
            self._last_primary_angle = primary_angle
            self._last_angle_t = now

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
            now = time.monotonic()
            buf = self._shoulder_sway_buffer
            buf.append((now, sh_x))
            while buf and now - buf[0][0] > _WINDOW_S:
                buf.popleft()
            if len(buf) >= 5:
                sway = statistics.pstdev([p[1] for p in buf])
                score = max(0.0, min(100.0, 100.0 - sway * 5000))
                self._current_balance_score = round(score, 1)
            return self._current_balance_score

    # -- Fatigue --------------------------------------------------------------
    def _record_rep_quality(self, quality: float):
        """Called once per completed rep with that rep's final score (0-100).
        Fatigue = drop-off between the first and second half of the session's
        rep scores."""
        quality = max(0.0, min(100.0, float(quality)))
        with self._fatigue_lock:
            self._rep_quality_buffer.append(quality)
            n = len(self._rep_quality_buffer)
            if n >= 4:
                half   = n // 2
                buf    = list(self._rep_quality_buffer)
                early  = buf[:half]
                recent = buf[half:]
                decline = max(0.0, sum(early) / len(early) - sum(recent) / len(recent))
                self._current_fatigue_score = round(min(100.0, decline * 2.5), 1)

    def maybe_record_rep_quality(self, reps: int, primary_angle: Optional[float], target_rom: float) -> float:
        """Call once per frame. Records each newly COMPLETED rep's score
        exactly once, then returns the current fatigue score."""
        with self._rep_lock:
            done = len(self._rep_scores)
            last = self._rep_scores[-1] if done else None
        with self._fatigue_lock:
            fresh = done > self._last_fatigue_rep_count and last is not None
        if fresh:
            self._record_rep_quality(last)
            with self._fatigue_lock:
                self._last_fatigue_rep_count = done
        return self.get_current_fatigue()

    def reset(self):
        """Call right before a session starts so rep count and all score
        buffers don't carry over stale data from a previous session."""
        with self._rep_lock:
            self._rep_count = 0
            self._rep_stage = None
            self._rep_peak = 0.0
            self._rom_window.clear()
            self._rest_min = None
            self._win_active = False
            self._win_counted = False
            self._win_peak = 0.0
            self._win_stab_sum = 0.0
            self._win_smooth_sum = 0.0
            self._win_bal_sum = 0.0
            self._win_n = 0
            self._rep_scores = []
            self._rep_details = []
            self._joint_recent.clear()
            self._win_joint = {}
            self._live_rep_score = None
            self._incomplete_reps = 0
        with self._stability_lock:
            self._landmark_jitter_buffer.clear()
            self._current_stability_score = 100.0
        with self._smoothness_lock:
            self._angle_velocity_buffer.clear()
            self._last_primary_angle = None
            self._last_angle_t = None
            self._current_smoothness_score = 100.0
        with self._balance_lock:
            self._shoulder_sway_buffer.clear()
            self._current_balance_score = 100.0
        with self._fatigue_lock:
            self._rep_quality_buffer.clear()
            self._current_fatigue_score = 0.0
            self._last_fatigue_rep_count = 0


# ---------------------------------------------------------------------------
# Module-level API - kept ONLY for the local MJPEG pipeline
# (services/mjpeg_camera.py). It delegates to one shared default instance, so
# it now runs the exact same logic as the per-session SessionMetrics above.
# ---------------------------------------------------------------------------
_default = SessionMetrics()


def set_exercise_state(exercise_type: Optional[str] = None, target_rom: Optional[float] = None):
    _default.set_exercise_state(exercise_type, target_rom)


def get_exercise_state():
    return _default.get_exercise_state()


def set_affected_side(side: Optional[str]):
    _default.set_affected_side(side)


def get_affected_side() -> str:
    return _default.get_affected_side()


def get_stability() -> float:
    return _default.get_stability()


def get_smoothness() -> float:
    return _default.get_smoothness()


def get_balance() -> float:
    return _default.get_balance()


def get_current_fatigue() -> float:
    return _default.get_current_fatigue()


def get_accuracy() -> float:
    return _default.get_accuracy()


def update_rep_count(primary_angle: Optional[float], target_rom: float) -> int:
    return _default.update_rep_count(primary_angle, target_rom)


def update_stability(landmarks) -> float:
    return _default.update_stability(landmarks)


def update_smoothness(primary_angle: Optional[float]) -> float:
    return _default.update_smoothness(primary_angle)


def update_balance(landmarks) -> float:
    return _default.update_balance(landmarks)


def maybe_record_rep_quality(reps: int, primary_angle: Optional[float], target_rom: float) -> float:
    return _default.maybe_record_rep_quality(reps, primary_angle, target_rom)


def reset_state():
    _default.reset()
