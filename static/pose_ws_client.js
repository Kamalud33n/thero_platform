/**
 * pose_ws_client.js — Cloud refactor Phase B: browser-pushed frames.
 *
 * Drop-in client for the /ws/pose engine. Captures the local webcam via
 * getUserMedia, draws frames to an offscreen canvas, JPEG-encodes them,
 * and pushes them over a WebSocket at a controlled rate (matches the
 * server's TARGET_FPS=10, see services/camera_ws.py). The server decodes
 * each frame, runs MediaPipe on it, and sends back an annotated frame +
 * pose_data (joint angles, reps, scores) on the same connection.
 *
 * This does NOT touch session.html's doctor-side MJPEG panel (that stays
 * local/in-clinic, see refactor plan Phase C) and does NOT touch
 * patient.html's existing WebRTC telehealth call or its separate
 * client-side MediaPipe Tasks overlay — this is a new, independent
 * capture path for solo/self-practice sessions where the server needs to
 * score the session itself (and persist it), not just show a live
 * skeleton to a doctor on the call.
 *
 * Usage:
 *   const client = new PoseWSClient({
 *     onPoseData: (msg) => { ... },        // {frame, pose_data, ts}
 *     onConnected: (connectionId) => { ... },
 *     onError: (err) => { ... },
 *   });
 *   await client.start();                  // asks for camera, opens WS
 *   client.setExercise('Knee Flexion', 100);
 *   client.resetSession();
 *   client.stop();                         // stops camera + closes WS
 */
class PoseWSClient {
    constructor({ wsPath = '/ws/pose', fps = 10, jpegQuality = 0.7,
                  onPoseData = null, onConnected = null, onError = null } = {}) {
        this.wsPath      = wsPath;
        this.fps         = fps;
        this.jpegQuality = jpegQuality;
        this.onPoseData  = onPoseData;
        this.onConnected = onConnected;
        this.onError     = onError;

        this.ws            = null;
        this.stream         = null;
        this.videoEl        = null;   // hidden <video>, source for canvas grabs
        this.canvas         = null;   // offscreen canvas for JPEG encoding
        this.ctx            = null;
        this.connectionId   = null;
        this._sendTimer     = null;
        this._running       = false;
    }

    async start() {
        // 1. Local webcam capture
        // Request a full-size capture (1280x720 ideal) instead of asking the
        // camera for 320x240 directly. Many webcams — especially Windows
        // laptop cameras — don't downscale their full field-of-view to a
        // small requested resolution; they center-crop instead, which is
        // what produced the "zoomed in" look. We grab the full frame here
        // and downscale it ourselves into the 320x240 canvas in _sendFrame().
        this.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: this.fps } },
            audio: false,
        });

        this.videoEl = document.createElement('video');
        this.videoEl.srcObject = this.stream;
        this.videoEl.muted = true;
        this.videoEl.playsInline = true;
        await this.videoEl.play();

        this.canvas = document.createElement('canvas');
        this.canvas.width  = 320;
        this.canvas.height = 240;
        this.ctx = this.canvas.getContext('2d');

        // 2. WebSocket connection
        const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
        this.ws = new WebSocket(`${proto}://${window.location.host}${this.wsPath}`);
        this.ws.onmessage = (evt) => this._handleMessage(evt);
        this.ws.onerror   = (evt) => { if (this.onError) this.onError(evt); };
        this.ws.onclose   = () => { this._running = false; this._stopSendLoop(); };

        await new Promise((resolve, reject) => {
            this.ws.addEventListener('open', resolve, { once: true });
            this.ws.addEventListener('error', reject, { once: true });
        });

        this._running = true;
        this._startSendLoop();
    }

    _handleMessage(evt) {
        let msg;
        try {
            msg = JSON.parse(evt.data);
        } catch (e) {
            return;
        }
        if (msg.type === 'connected') {
            this.connectionId = msg.connection_id;
            if (this.onConnected) this.onConnected(this.connectionId);
        } else if (msg.type === 'pose_data') {
            if (this.onPoseData) this.onPoseData(msg);
        }
    }

    _startSendLoop() {
        const intervalMs = 1000 / this.fps;
        this._sendTimer = setInterval(() => this._sendFrame(), intervalMs);
    }

    _stopSendLoop() {
        if (this._sendTimer) {
            clearInterval(this._sendTimer);
            this._sendTimer = null;
        }
    }

    _sendFrame() {
        if (!this._running || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (!this.videoEl || this.videoEl.readyState < 2) return;

        // Draw the FULL captured frame (source: actual videoWidth/videoHeight)
        // downscaled into the 320x240 destination canvas — this is what
        // avoids the center-crop/zoom effect described above.
        this.ctx.drawImage(
            this.videoEl,
            0, 0, this.videoEl.videoWidth, this.videoEl.videoHeight,   // source: full captured frame
            0, 0, this.canvas.width, this.canvas.height                // dest: downscale to 320x240
        );
        // toDataURL is simplest cross-browser path; strip the
        // "data:image/jpeg;base64," prefix before sending.
        const dataUrl = this.canvas.toDataURL('image/jpeg', this.jpegQuality);
        const base64  = dataUrl.substring(dataUrl.indexOf(',') + 1);

        this.ws.send(JSON.stringify({ type: 'frame', data: base64 }));
    }

    setExercise(exerciseType, targetRom) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        this.ws.send(JSON.stringify({
            type: 'set_exercise',
            exercise_type: exerciseType,
            target_rom: targetRom,
        }));
    }

    resetSession() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        this.ws.send(JSON.stringify({ type: 'reset_session' }));
    }

    stop() {
        this._running = false;
        this._stopSendLoop();
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}