import time
import uuid
from typing import Dict, List, Optional

import cv2
import numpy as np

from fastapi import WebSocket

from config import pose as _pose, mp_drawing as _mp_drawing
from config import mp_drawing_styles as _mp_drawing_styles
from config import POSE_CONNECTIONS as _POSE_CONNECTIONS
from config import KEY_LANDMARKS as _KEY_LANDMARKS
from config import get_angle as _get_angle
from config import hands as _hands, HAND_CONNECTIONS as _HAND_CONNECTIONS
from services import metrics as _metrics
from services.exercise_defs import is_hand_exercise as _is_hand_exercise
from services.helpers import stamp as _stamp


class CameraManager:
    # Target FPS for streaming — caps how often we send annotated frames back
    TARGET_FPS   = 10
    FRAME_BUDGET = 1.0 / TARGET_FPS   # seconds per frame

    # ── Real backpressure (item: "frame rate limiting is just frame-skip/
    # FPS-throttle, not real backpressure") ────────────────────────────
    # should_process()/fps_throttle() below are UNCHANGED — they still do
    # the cheap fixed frame-skip / FPS-cap. What's new is congestion
    # DETECTION and an explicit signal back to the browser, which is what
    # "real backpressure" actually means: the producer (browser) learning
    # it's overwhelming the consumer (this connection) and adapting,
    # instead of the server silently discarding frames it can't keep up
    # with while the browser keeps pushing at full rate regardless.
    MIN_SKIP = 2      # existing behaviour: process every 2nd frame
    MAX_SKIP = 8      # worst case under sustained overload: 1 in 8
    _EMA_ALPHA = 0.25
    BACKPRESSURE_SIGNAL_INTERVAL = 3.0  # don't spam the client more than this often

    def __init__(self, metrics: Optional[_metrics.SessionMetrics] = None):
        self.is_running  = False
        self._frame_count = 0           # for frame skipping
        self._last_sent   = 0.0         # for FPS throttling
        self._draw_landmarks = True     # can disable for more speed
        # Own metrics instance (rep count, stability/smoothness/balance/
        # fatigue buffers) — one per CameraManager instead of the shared
        # module-level globals, so each WS connection's session state is
        # isolated. See services/metrics.py: SessionMetrics.
        self.metrics = metrics or _metrics.SessionMetrics()

        # Backpressure state — see note above.
        self._skip_n = self.MIN_SKIP                 # adaptive; grows under load
        self._proc_time_ema = self.FRAME_BUDGET       # smoothed process_frame() cost
        self._last_arrival = None                     # for inbound-rate detection
        self._arrival_gap_ema = self.FRAME_BUDGET
        self._last_backpressure_sent = 0.0

    def note_arrival(self) -> None:
        """Call once per inbound 'frame' message, BEFORE should_process()
        decides whether to actually run it. Tracks how fast the browser is
        sending, independent of whether we choose to process each one —
        this is what lets us detect "browser sending faster than we asked"
        even on frames we're about to skip."""
        now = time.monotonic()
        if self._last_arrival is not None:
            gap = now - self._last_arrival
            self._arrival_gap_ema = (
                self._EMA_ALPHA * gap + (1 - self._EMA_ALPHA) * self._arrival_gap_ema
            )
        self._last_arrival = now

    def note_processed(self, duration: float) -> None:
        """Call after process_frame() returns, with how long it took.
        Smoothed cost feeds the adaptive skip ratio below."""
        self._proc_time_ema = (
            self._EMA_ALPHA * duration + (1 - self._EMA_ALPHA) * self._proc_time_ema
        )
        # Adapt: if processing is eating more than ~90% of the frame
        # budget, we can't sustain the current skip ratio — skip more
        # aggressively. If there's comfortable headroom, ease back toward
        # MIN_SKIP so quality/responsiveness recovers once load drops.
        if self._proc_time_ema > self.FRAME_BUDGET * 0.9 and self._skip_n < self.MAX_SKIP:
            self._skip_n += 1
        elif self._proc_time_ema < self.FRAME_BUDGET * 0.5 and self._skip_n > self.MIN_SKIP:
            self._skip_n -= 1

    def is_congested(self) -> bool:
        """True if either (a) processing itself can't keep up with the
        target budget, or (b) the browser is sending faster than we're
        willing/able to process — the two independent signals that
        distinguish real backpressure from a fixed skip ratio."""
        overloaded = self._proc_time_ema > self.FRAME_BUDGET
        sender_too_fast = self._arrival_gap_ema < (self.FRAME_BUDGET / self.MIN_SKIP) * 0.8
        return overloaded or (sender_too_fast and self._skip_n > self.MIN_SKIP)

    def recommended_client_fps(self) -> float:
        """What we're actually able to sustain right now, given the
        current adaptive skip ratio — the number the client should throttle
        its own capture/send loop to, if it chooses to honor the signal."""
        return round(self.TARGET_FPS / self._skip_n, 1)

    def should_send_backpressure_signal(self) -> bool:
        """Rate-limits the control message sent to the client — congestion
        state is checked every frame, but the client only needs to hear
        about it every few seconds, not every frame."""
        if not self.is_congested():
            return False
        now = time.monotonic()
        if now - self._last_backpressure_sent < self.BACKPRESSURE_SIGNAL_INTERVAL:
            return False
        self._last_backpressure_sent = now
        return True

    def start(self) -> bool:
        """Cloud refactor (Phase B): there's no local device to open anymore
        — frames arrive from the browser over the WS connection instead
        (see routers/ws.py). This just resets this connection's frame-skip/
        FPS-throttle state for a fresh session."""
        self.is_running    = True
        self._frame_count  = 0
        self._last_sent    = 0.0
        self._skip_n            = self.MIN_SKIP
        self._proc_time_ema     = self.FRAME_BUDGET
        self._last_arrival      = None
        self._arrival_gap_ema   = self.FRAME_BUDGET
        self._last_backpressure_sent = 0.0
        return True

    def stop(self) -> bool:
        """No device to release anymore — just marks the session inactive."""
        self.is_running = False
        return True

    @staticmethod
    def decode_frame(raw: bytes):
        """Decode a browser-sent JPEG (raw bytes, already base64-decoded by
        the caller) into an OpenCV BGR frame — the browser-push equivalent
        of the old `self.cap.read()`."""
        if not raw:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame

    def should_process(self) -> bool:
        """Process only every Nth frame — N is adaptive (see note_processed/
        note_arrival above), not a fixed 2. Starts at MIN_SKIP (same
        behaviour as before) and grows under sustained load, shrinks back
        once load clears."""
        self._frame_count += 1
        return self._frame_count % self._skip_n == 0

    def fps_throttle(self) -> bool:
        """Return True if enough time has passed to send next frame."""
        now = time.monotonic()
        if now - self._last_sent >= self.FRAME_BUDGET:
            self._last_sent = now
            return True
        return False

    def process_frame(self, frame):
        if frame is None:
            return frame, None

        active_exercise, target_rom = self.metrics.get_exercise_state()
        if _is_hand_exercise(active_exercise):
            return self._process_hand_frame(frame, active_exercise, target_rom)

        if _pose is None:
            return frame, None

        # Convert BGR → RGB (MediaPipe needs RGB)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # avoid unnecessary copy inside MP
        results = _pose.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()

        if not results.pose_landmarks:
            return annotated, None

        # Optional landmark drawing — disable if still slow
        if self._draw_landmarks and _mp_drawing:
            _mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                _POSE_CONNECTIONS,
                landmark_drawing_spec=(
                    _mp_drawing_styles.get_default_pose_landmarks_style()
                    if _mp_drawing_styles else None
                ),
            )

        lm = results.pose_landmarks.landmark

        # Only send 13 key landmarks (not all 33)
        pose_data = {}
        for name, idx in _KEY_LANDMARKS.items():
            if idx < len(lm) and lm[idx].visibility > 0.5:   # skip low-confidence
                pose_data[name] = {
                    "x": round(lm[idx].x, 4),
                    "y": round(lm[idx].y, 4),
                    "z": round(lm[idx].z, 4),
                    "v": round(lm[idx].visibility, 2),
                }

        # Joint angles
        angles = {}
        try:
            if len(lm) > 16:
                angles["l_shoulder"] = round(_get_angle(lm[23], lm[11], lm[13]), 1)
                angles["r_shoulder"] = round(_get_angle(lm[24], lm[12], lm[14]), 1)
                angles["l_elbow"]    = round(_get_angle(lm[11], lm[13], lm[15]), 1)
                angles["r_elbow"]    = round(_get_angle(lm[12], lm[14], lm[16]), 1)
            if len(lm) > 28:
                angles["l_knee"]     = round(_get_angle(lm[23], lm[25], lm[27]), 1)
                angles["r_knee"]     = round(_get_angle(lm[24], lm[26], lm[28]), 1)
            if len(lm) > 26:
                angles["l_hip"]      = round(_get_angle(lm[11], lm[23], lm[25]), 1)
                angles["r_hip"]      = round(_get_angle(lm[12], lm[24], lm[26]), 1)
        except Exception:
            pass

        pose_data["angles"] = angles

        # Real scoring, not just angles — this branch previously stopped
        # here and never called the metrics below, so /ws/pose sessions
        # never got rep counts / stability / smoothness / balance /
        # fatigue live (the MJPEG pipeline in mjpeg_camera.py already did
        # this; this branch just hadn't been wired up to match).
        #
        # affected_side (per Nada: "both sides scored and stored, with
        # affected_side treated as primary") drives which side's angle
        # counts reps, while `angles` above still carries both l_/r_
        # readings every frame regardless of side.
        affected_side = self.metrics.get_affected_side()
        primary_angle = _metrics.compute_primary_angle(angles, active_exercise, affected_side)
        reps       = self.metrics.update_rep_count(primary_angle, target_rom)
        stability  = self.metrics.update_stability(lm)
        smoothness = self.metrics.update_smoothness(primary_angle)
        balance    = self.metrics.update_balance(lm)
        fatigue    = self.metrics.maybe_record_rep_quality(reps, primary_angle, target_rom)
        # maybe_record_rep_quality() above is what appends into the
        # rep-quality buffer get_accuracy() reads — must run first so this
        # frame's accuracy reflects any rep that just completed.
        accuracy   = self.metrics.get_accuracy()

        pose_data["reps"]          = reps
        pose_data["primary_angle"] = primary_angle
        pose_data["affected_side"] = affected_side
        pose_data["stability"]     = stability
        pose_data["smoothness"]    = smoothness
        pose_data["balance"]       = balance
        pose_data["fatigue"]       = fatigue
        pose_data["accuracy"]      = accuracy
        # Aliases matching the SAVED-summary field names (routers/sessions.py
        # / models.SessionModel) — the live pose_data broadcast historically
        # used short names while the DB/summary side uses *_score / *_estimation
        # suffixes. Any consumer (doctor dashboard, analytics-in-progress
        # views, future clients) that expects the summary naming can now read
        # pose_data without needing to know both conventions. Kept alongside
        # the short keys rather than replacing them, since templates/patient.html
        # already reads the short keys for its own live overlay.
        pose_data["stability_score"]     = stability
        pose_data["movement_smoothness"] = smoothness
        pose_data["balance_score"]       = balance
        pose_data["fatigue_estimation"]  = fatigue
        pose_data["accuracy_percentage"] = accuracy
        return annotated, pose_data

    def _process_hand_frame(self, frame, active_exercise: str, target_rom: float):
        """Hand Grip / Finger Flexion path — MediaPipe Hands instead of Pose,
        mirrors the MJPEG pipeline's hand branch so both delivery modes
        (WebSocket relay + MJPEG <img>) behave identically."""
        if _hands is None:
            return frame, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = _hands.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()

        if not results.multi_hand_landmarks:
            return annotated, None

        if self._draw_landmarks and _mp_drawing and _HAND_CONNECTIONS:
            for hand_landmarks in results.multi_hand_landmarks:
                _mp_drawing.draw_landmarks(annotated, hand_landmarks, _HAND_CONNECTIONS)

        lm0 = results.multi_hand_landmarks[0].landmark
        # compute_finger_curl_angles / compute_primary_angle are pure
        # functions (no state) — stay as module-level calls. update_rep_count
        # is stateful (rep counter), so it goes through this connection's
        # own metrics instance.
        finger_angles = _metrics.compute_finger_curl_angles(lm0)
        primary_angle = _metrics.compute_primary_angle(finger_angles, active_exercise)
        reps = self.metrics.update_rep_count(primary_angle, target_rom)
        # Hand-grip path never called maybe_record_rep_quality() before —
        # so the rep-quality buffer (fatigue's source, and now accuracy's
        # source too) was never populated for hand exercises at all.
        # Recording it here is what makes accuracy real for Hand Grip
        # instead of permanently reading 0.0.
        fatigue = self.metrics.maybe_record_rep_quality(reps, primary_angle, target_rom)
        accuracy = self.metrics.get_accuracy()

        pose_data = {}
        # Send finger points the same way body key-landmarks are sent, so
        # any downstream JS drawing/consumption code can treat them uniformly.
        for name, idx in _config_hand_landmarks().items():
            lmk = lm0[idx]
            pose_data[name] = {"x": round(lmk.x, 4), "y": round(lmk.y, 4), "z": round(lmk.z, 4)}

        pose_data["angles"] = finger_angles
        pose_data["reps"] = reps
        pose_data["primary_angle"] = primary_angle
        pose_data["affected_side"] = self.metrics.get_affected_side()
        pose_data["fatigue"] = fatigue
        pose_data["accuracy"] = accuracy
        # Same summary-naming aliases as the pose branch above (item: field
        # name mismatch). No stability/balance here — hand exercises never
        # compute those (no hip landmarks in a hands-only frame).
        pose_data["fatigue_estimation"]  = fatigue
        pose_data["accuracy_percentage"] = accuracy
        return annotated, pose_data


def _config_hand_landmarks():
    from config import HAND_LANDMARKS
    return HAND_LANDMARKS


class RoomConnection:
    """One WebSocket connection inside a Room. `camera` is only set for the
    patient side — a doctor connection has no CameraManager at all, so
    there is no code path by which it could process/send frames even if
    the router forgot to check the role. Read-only is structural, not just
    a router-level `if role == "doctor": continue`."""

    def __init__(self, ws: WebSocket, role: str, connection_id: str,
                 camera: Optional[CameraManager] = None):
        self.ws = ws
        self.role = role                # "patient" | "doctor"
        self.connection_id = connection_id
        self.camera = camera


class Room:
    """All connections for one session_id. Today this is at most one
    patient + one (or zero) doctor, but keyed/collection-based so a second
    observer connection doesn't require a structural change later."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.connections: Dict[str, RoomConnection] = {}

    def add(self, conn: RoomConnection) -> None:
        self.connections[conn.connection_id] = conn

    def remove(self, connection_id: str) -> None:
        self.connections.pop(connection_id, None)

    def is_empty(self) -> bool:
        return not self.connections


class RoomManager:
    """Cloud/concurrency refactor (Phase A) + Room model (Phase D):
    connections are grouped into Rooms keyed by session_id instead of each
    living in one flat connection_id -> CameraManager map. This is what
    lets a doctor connection join the *same* session as the patient and
    receive that patient's pose_data/scores as they're produced, while
    still keeping session A's state fully isolated from session B's (each
    Room only ever touches its own session_id).

    NOTE: `role` itself is still passed in from routers/ws.py as a plain
    string, but it's no longer trust-on-connect — both the patient and
    doctor paths in routers/ws.py now verify a signed session token
    (patient: auth.decode_patient_session_token, doctor:
    auth.decode_doctor_session_token) with a matching `role` claim
    *before* ws_pose() ever calls this connect(). A request that reaches
    here has already had its role authorized upstream.
    """

    def __init__(self):
        self.rooms: Dict[str, Room] = {}
        # ws -> (session_id, connection_id), so disconnect/get_connection
        # don't need the caller to remember which room a socket is in.
        self._ws_to_ref: Dict[WebSocket, tuple] = {}

    async def connect(self, ws: WebSocket, session_id: str, role: str) -> str:
        await ws.accept()
        connection_id = uuid.uuid4().hex[:12]
        camera = CameraManager() if role == "patient" else None
        conn = RoomConnection(ws, role, connection_id, camera)

        room = self.rooms.setdefault(session_id, Room(session_id))
        room.add(conn)
        self._ws_to_ref[ws] = (session_id, connection_id)
        return connection_id

    def get_connection(self, ws: WebSocket) -> Optional[RoomConnection]:
        ref = self._ws_to_ref.get(ws)
        if not ref:
            return None
        session_id, connection_id = ref
        room = self.rooms.get(session_id)
        if not room:
            return None
        return room.connections.get(connection_id)

    def disconnect(self, ws: WebSocket):
        ref = self._ws_to_ref.pop(ws, None)
        if not ref:
            return
        session_id, connection_id = ref
        room = self.rooms.get(session_id)
        if not room:
            return
        conn = room.connections.get(connection_id)
        if conn and conn.camera is not None:
            conn.camera.stop()
        room.remove(connection_id)
        if room.is_empty():
            # Drop the room entirely once both sides have left — nothing
            # should keep referencing a session_id after this point.
            self.rooms.pop(session_id, None)

    async def broadcast(self, session_id: str, data: dict) -> None:
        """Send to every connection currently in the room — patient's own
        client (so it renders its own overlay) and any doctor watching."""
        room = self.rooms.get(session_id)
        if not room:
            return
        for conn in list(room.connections.values()):
            await self.send(conn.ws, data)

    @property
    def connections(self) -> List[WebSocket]:
        """Kept for /api/health's connection-count display."""
        return list(self._ws_to_ref.keys())

    async def send(self, ws: WebSocket, data: dict):
        try:
            # Item 26: every server->client frame goes out through here or
            # through broadcast() (which itself calls this), so stamping
            # centrally here covers all /ws/pose messages in one place
            # instead of every call site remembering to do it.
            await ws.send_json(_stamp(data))
        except Exception:
            pass


ws_mgr = RoomManager()