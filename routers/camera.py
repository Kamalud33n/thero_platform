"""
LOCAL-ONLY router: /video_feed and friends serve the doctor's in-clinic
desktop camera pipeline (services/mjpeg_camera.py — cv2.VideoCapture(0) on
this machine). Not part of the cloud/telehealth path. Gated by
config.LOCAL_CAMERA_ENABLED; ops should set that to false on any cloud
deployment. (Refactor plan: "Cloud/Concurrency Refactor", Phase C.)
"""
from typing import Dict, Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse

from config import LOCAL_CAMERA_ENABLED
from services import mjpeg_camera, metrics

router = APIRouter()


@router.get("/video_feed")
async def video_feed():
    """Primary live camera feed with pose skeleton drawn server-side. <img src='/video_feed'>"""
    if not LOCAL_CAMERA_ENABLED:
        return JSONResponse(
            {"success": False, "message": "Local camera is disabled on this deployment"},
            status_code=503,
        )
    return StreamingResponse(
        mjpeg_camera.gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/pose_data")
async def api_pose_data():
    """Latest joint angles + detection flag, updated every frame by gen_frames()."""
    return JSONResponse(mjpeg_camera.latest_pose_data)


@router.post("/api/camera/stop")
async def api_camera_stop():
    """Explicitly release the MJPEG camera device (called on Stop / page unload)."""
    mjpeg_camera.stop_camera()
    return JSONResponse({"success": True, "message": "Camera stopped"})


@router.post("/api/exercise_type")
async def set_exercise_type(payload: Dict[str, Any]):
    """Frontend calls this whenever the exercise dropdown (or target ROM,
    or affected-side selector) changes, so the MJPEG stream draws the
    right joints, rep-counts against the right threshold, and scores the
    right side as primary."""
    ex  = payload.get("exercise_type")
    rom = payload.get("target_rom")
    side = payload.get("affected_side")
    metrics.set_exercise_state(exercise_type=ex, target_rom=rom)
    metrics.set_affected_side(side)
    current_ex, current_rom = metrics.get_exercise_state()
    return JSONResponse({
        "success": True,
        "exercise_type": current_ex,
        "target_rom": current_rom,
        "affected_side": metrics.get_affected_side(),
    })


@router.post("/api/session/reset")
async def api_session_reset():
    """Call this right before a session starts so rep count + stability
    buffer don't carry over stale data from a previous session/patient."""
    metrics.reset_state()
    return JSONResponse({"success": True})


@router.get("/api/camera/status")
async def api_camera_status():
    """Quick status check — useful for frontend polling / debugging."""
    return JSONResponse({
        "active": mjpeg_camera.is_active(),
        "local_camera_enabled": LOCAL_CAMERA_ENABLED,
    })