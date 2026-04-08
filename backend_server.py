import asyncio
import json
import threading
import time
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Any

import cv2
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pose_pipeline import landmarks_to_pixels, pose_embedding, pose_similarity, draw_landmarks
from scoring_engine import load_choreography, ScoringEngine, DEFAULT_THRESHOLDS

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

CHOREO_JSON_PATH = "charts/ymca_extra.json"
SCORING_JSON_PATH = "charts/ymca.json"
WINDOW_HALF_MS = 150
MODEL_PATH = "models/pose_landmarker_full.task"
START_COUNTDOWN_MS = 3000


def _load_chart_metadata(path: str) -> Dict[str, Any]:
    chart = json.loads(Path(path).read_text())
    return {
        "path": path,
        "title": chart.get("title"),
        "video_path": chart.get("video_path"),
        "bpm": chart.get("bpm"),
        "offset_ms": chart.get("offset_ms"),
        "move_count": len(chart.get("moves", [])),
    }


class OptionUpdate(BaseModel):
    webcam_enabled: bool | None = None
    show_landmarks: bool | None = None
    reference_video_enabled: bool | None = None
    reference_audio_enabled: bool | None = None


class DanceRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"
        self.game_ts_ms = 0
        self.countdown_deadline = None
        self.running_zero_monotonic = None

        self.total_points = 0
        self.combo = 0
        self.best_combo = 0
        self.last_feedback = None

        self.options = {
            "webcam_enabled": True,
            "show_landmarks": True,
            "reference_video_enabled": True,
            "reference_audio_enabled": True,
        }

        self.current_move = None
        self.upcoming_moves = []
        self.latest_frame = None

        self.scoring_moves = load_choreography(SCORING_JSON_PATH)
        self.display_moves = load_choreography(CHOREO_JSON_PATH)
        self.display_start_times = [m.start_ms for m in self.display_moves]
        self.scorer = ScoringEngine(self.scoring_moves, window_half_ms=WINDOW_HALF_MS, thresholds=DEFAULT_THRESHOLDS)

        self.chart_meta = {
            "scoring": _load_chart_metadata(SCORING_JSON_PATH),
            "display": _load_chart_metadata(CHOREO_JSON_PATH),
        }

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def start_session(self):
        with self.lock:
            self.state = "countdown"
            self.countdown_deadline = time.monotonic() + START_COUNTDOWN_MS / 1000.0

    def pause_toggle(self):
        with self.lock:
            if self.state == "running":
                self.state = "paused"
            elif self.state == "paused":
                self.running_zero_monotonic = time.monotonic() - (self.game_ts_ms / 1000.0)
                self.state = "running"

    def reset_session(self):
        with self.lock:
            self.scorer = ScoringEngine(self.scoring_moves, window_half_ms=WINDOW_HALF_MS, thresholds=DEFAULT_THRESHOLDS)
            self.state = "idle"
            self.game_ts_ms = 0
            self.countdown_deadline = None
            self.running_zero_monotonic = None
            self.total_points = 0
            self.combo = 0
            self.last_feedback = None

    def update_options(self, data: OptionUpdate):
        with self.lock:
            for key, value in data.model_dump().items():
                if value is not None:
                    self.options[key] = value

    def _find_current_and_upcoming(self, max_upcoming=4):
        if not self.display_moves:
            return None, []
        idx = bisect_right(self.display_start_times, self.game_ts_ms) - 1
        if idx < 0:
            idx = 0
        current = self.display_moves[idx]
        return current, self.display_moves[idx + 1 : idx + 1 + max_upcoming]

    def _run_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )

        mp_ts_ms = 0
        with PoseLandmarker.create_from_options(options) as landmarker:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                with self.lock:
                    webcam_enabled = self.options["webcam_enabled"]
                    show_landmarks = self.options["show_landmarks"]
                frame_to_process = frame if webcam_enabled else frame * 0
                rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_ts_ms += 33
                result = landmarker.detect_for_video(mp_image, mp_ts_ms)

                with self.lock:
                    if self.state == "countdown":
                        ms_left = int((self.countdown_deadline - time.monotonic()) * 1000)
                        if ms_left <= 0:
                            self.state = "running"
                            self.running_zero_monotonic = time.monotonic()
                            self.game_ts_ms = 0
                    elif self.state == "running":
                        self.game_ts_ms = int((time.monotonic() - self.running_zero_monotonic) * 1000)

                    live_emb = None
                    if result and result.pose_landmarks and webcam_enabled:
                        h, w = frame.shape[:2]
                        live_pts = landmarks_to_pixels(result.pose_landmarks[0], w, h)
                        live_emb = pose_embedding(live_pts)

                    if self.state == "running" and live_emb is not None and self.scoring_moves:
                        finalized = self.scorer.update(self.game_ts_ms, live_emb, pose_similarity)
                        for res in finalized:
                            tier = (res.tier or "miss").lower()
                            if tier != "miss":
                                gained = max(0, int(round(res.best_score)))
                                self.total_points += gained
                                self.combo += 1
                                self.best_combo = max(self.best_combo, self.combo)
                            else:
                                gained = 0
                                self.combo = 0
                            self.last_feedback = {
                                "tier": tier,
                                "score": float(res.best_score),
                                "gained": gained,
                                "move_name": res.name,
                                "show_until_ms": self.game_ts_ms + 1000,
                            }

                    current, upcoming = self._find_current_and_upcoming()
                    self.current_move = current
                    self.upcoming_moves = upcoming

                frame_out = frame.copy()
                if show_landmarks and result and result.pose_landmarks:
                    draw_landmarks(frame_out, result.pose_landmarks[0])

                ok_jpg, jpg = cv2.imencode('.jpg', frame_out)
                if ok_jpg:
                    with self.lock:
                        self.latest_frame = jpg.tobytes()

                time.sleep(0.01)

        cap.release()

    def state_payload(self):
        with self.lock:
            feedback = self.last_feedback
            if feedback and self.game_ts_ms > feedback["show_until_ms"]:
                feedback = None

            def move_json(move):
                if not move:
                    return None
                pose = (move.raw.get("pose", {}) or {}) if move.raw else {}
                return {
                    "name": move.name,
                    "start_ms": move.start_ms,
                    "norm_xy": pose.get("norm_xy"),
                }

            return {
                "status": self.state,
                "game_ts_ms": self.game_ts_ms,
                "score": self.total_points,
                "combo": self.combo,
                "best_combo": self.best_combo,
                "feedback": feedback,
                "current_move": move_json(self.current_move),
                "upcoming_moves": [move_json(m) for m in self.upcoming_moves],
                "options": dict(self.options),
                "chart_meta": self.chart_meta,
            }

    def mjpeg_stream(self):
        while True:
            with self.lock:
                frame = self.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)


runtime = DanceRuntime()
app = FastAPI(title="JustDanceOpenCV Frontend Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

frontend_dir = Path("frontend")
app.mount("/assets", StaticFiles(directory=frontend_dir), name="assets")
app.mount("/media", StaticFiles(directory="."), name="media")


@app.get("/")
def root():
    return FileResponse(frontend_dir / "index.html")


@app.get("/api/config")
def api_config():
    return JSONResponse(runtime.state_payload())


@app.post("/api/session/start")
def start_session():
    runtime.start_session()
    return {"ok": True}


@app.post("/api/session/pause-toggle")
def pause_toggle():
    runtime.pause_toggle()
    return {"ok": True}


@app.post("/api/session/reset")
def reset_session():
    runtime.reset_session()
    return {"ok": True}


@app.post("/api/options")
def set_options(options: OptionUpdate):
    runtime.update_options(options)
    return {"ok": True}


@app.get("/video/feed")
def video_feed():
    return StreamingResponse(runtime.mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(runtime.state_payload())
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return
