import os
import sys
import types
import json as _json
import warnings

warnings.filterwarnings("ignore")

# Local-camera pipeline toggle (services/mjpeg_camera.py + /video_feed in
# routers/camera.py). That pipeline calls cv2.VideoCapture(0) directly on
# the machine running the server — it only makes sense when the server IS
# the doctor's in-clinic desktop, sitting next to the webcam. On a cloud
# deployment there's no camera attached to the server at all, so set
# LOCAL_CAMERA_ENABLED=false there: the app then skips even attempting to
# open a device and returns a clear "disabled" response instead of
# spending time probing for hardware that isn't there.
# (Refactor plan: "Cloud/Concurrency Refactor", Phase C.)
LOCAL_CAMERA_ENABLED = os.getenv("LOCAL_CAMERA_ENABLED", "true").lower() == "true"

# ── Block real `sounddevice` import before MediaPipe pulls it in ──────────
# `import mediapipe` unconditionally imports mediapipe.tasks.python, which
# imports its audio submodule, which imports `sounddevice` and calls
# PortAudio's Pa_Initialize() at import time. On some Windows machines
# (bad/absent audio drivers, sleeping Bluetooth audio devices, etc.) that
# call hangs forever — even though we never use MediaPipe's audio features,
# only Pose/Hands. Registering a fake `sounddevice` module in sys.modules
# BEFORE mediapipe is imported makes mediapipe's `import sounddevice`
# resolve instantly to this stub instead of loading the real PortAudio
# binding, so it never hangs. Safe because nothing in this app touches audio.
if "sounddevice" not in sys.modules:
    _fake_sd = types.ModuleType("sounddevice")
    _fake_sd.query_devices = lambda *a, **k: []
    _fake_sd.default = types.SimpleNamespace(device=(None, None))
    _fake_sd.InputStream = None
    _fake_sd.PortAudioError = Exception
    sys.modules["sounddevice"] = _fake_sd

import numpy as np
import mediapipe as mp
from fastapi.templating import Jinja2Templates

# Directories 
for d in ("data", "reports", "uploads", "static", "templates", "assets"):
    os.makedirs(d, exist_ok=True)

# PDF report palette — used by services/report_builder.py. Kept here (not
# in report_builder.py itself) so any future PDF-producing code shares the
# same look without redefining these, same reasoning as `templates` above.
from reportlab.lib import colors as _rl_colors

PDF_NAVY        = _rl_colors.HexColor("#1B2A4A")
PDF_GREY_BORDER = _rl_colors.HexColor("#D8DCE3")
PDF_GREY_BG     = _rl_colors.HexColor("#F5F6F8")
PDF_GREY_TEXT   = _rl_colors.HexColor("#5A6472")
PDF_ROW_ALT     = _rl_colors.HexColor("#FAFBFC")
PDF_BODY_TEXT   = _rl_colors.HexColor("#1A2332")

# Shared Jinja2 templates instance (import this everywhere instead of
# creating a new Jinja2Templates(...) so the `tojson` filter is available
# in every router that renders HTML) 
templates = Jinja2Templates(directory="templates")
templates.env.filters["tojson"] = lambda obj: _json.dumps(obj)

# MediaPipe init (optimized) 
# NOTE (fix — pose not detecting): min_detection_confidence / min_tracking_confidence
# lowered from 0.5 -> 0.3 so low-light / overexposed webcam frames still register
# a person. model_complexity bumped 0 -> 1 for better accuracy in poor lighting;
# smooth_landmarks turned back on to reduce jitter from the lower thresholds.
# If FPS drops too much on this machine, set model_complexity back to 0 and just
# keep the lower confidence values.
try:
    _mp_pose = mp.solutions.pose
    pose = _mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,           # ← was 0; 1 = more accurate, still real-time
        smooth_landmarks=True,        # ← was False; reduces jitter
        min_detection_confidence=0.3, # ← was 0.5; easier to detect in poor lighting
        min_tracking_confidence=0.3,  # ← was 0.5
    )
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    POSE_CONNECTIONS  = _mp_pose.POSE_CONNECTIONS
    print("MediaPipe Pose initialized (complexity=1, conf=0.3)")
except Exception as exc:
    print(f"MediaPipe init failed: {exc}")
    pose = mp_drawing = mp_drawing_styles = POSE_CONNECTIONS = None

# MediaPipe Hands init — separate model, needed for finger/grip tracking.
# Pose's 33 landmarks stop at the wrist, so a real finger-curl / hand-grip
# exercise needs this second model running alongside Pose.
try:
    _mp_hands = mp.solutions.hands
    hands = _mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,           # ← 0 = fastest, matches Pose setting
        min_detection_confidence=0.3, # ← was 0.5; matches Pose threshold change above
        min_tracking_confidence=0.3,  # ← was 0.5
    )
    HAND_CONNECTIONS = _mp_hands.HAND_CONNECTIONS
    print("MediaPipe Hands initialized (optimized: complexity=0, conf=0.3)")
except Exception as exc:
    print(f"MediaPipe Hands init failed: {exc}")
    hands = HAND_CONNECTIONS = None

# Hand landmark indices (21 points per hand) — MCP/PIP/DIP/TIP per finger,
# used for finger-curl angle calculation.
HAND_LANDMARKS = {
    "wrist": 0,
    "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
    "index_mcp": 5, "index_pip": 6, "index_dip": 7, "index_tip": 8,
    "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11, "middle_tip": 12,
    "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15, "ring_tip": 16,
    "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20,
}

# Key landmarks only (13 joints instead of 33) 
# Indices: nose=0, shoulders=11/12, elbows=13/14, wrists=15/16,
#          hips=23/24, knees=25/26, ankles=27/28
KEY_LANDMARKS = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def get_angle(p1, p2, p3) -> float:
    a = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    b = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))