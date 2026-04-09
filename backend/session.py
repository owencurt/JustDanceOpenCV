import base64
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from scoring_engine import DEFAULT_THRESHOLDS, ScoringEngine, load_choreography
from backend.game_logic import (
    find_current_and_upcoming,
    landmarks_to_pixels,
    move_to_payload,
    pose_embedding,
    pose_similarity,
)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

MODEL_PATH = "models/pose_landmarker_full.task"
WINDOW_HALF_MS = 150
START_COUNTDOWN_MS = 3000


@dataclass
class Feedback:
    tier: str
    move_name: str
    move_score: float
    show_until_ms: int


class GameSession:
    def __init__(self, scoring_chart: str, choreo_chart: str):
        self.scoring_chart_path = scoring_chart
        self.choreo_chart_path = choreo_chart

        self.scoring_moves = load_choreography(scoring_chart)
        self.display_moves = load_choreography(choreo_chart)
        self.display_start_times = [m.start_ms for m in self.display_moves]

        self.scorer = ScoringEngine(
            moves=self.scoring_moves,
            window_half_ms=WINDOW_HALF_MS,
            thresholds=DEFAULT_THRESHOLDS,
            tie_breaker="closest",
        )

        self.state = "idle"
        self.game_ts_ms = 0
        self.countdown_deadline: Optional[float] = None
        self.running_zero_monotonic: Optional[float] = None
        self.total_points = 0
        self.feedback: Optional[Feedback] = None

        self._mp_ts_ms = 0
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def close(self):
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def _reset_scorer(self):
        self.scorer = ScoringEngine(
            moves=self.scoring_moves,
            window_half_ms=WINDOW_HALF_MS,
            thresholds=DEFAULT_THRESHOLDS,
            tie_breaker="closest",
        )

    def command(self, action: str):
        if action == "start":
            self._reset_scorer()
            self.feedback = None
            self.total_points = 0
            self.state = "countdown"
            self.countdown_deadline = time.monotonic() + START_COUNTDOWN_MS / 1000.0
            self.game_ts_ms = 0
        elif action == "pause" and self.state == "running":
            self.state = "paused"
        elif action == "resume" and self.state == "paused":
            self.running_zero_monotonic = time.monotonic() - (self.game_ts_ms / 1000.0)
            self.state = "running"
        elif action == "reset":
            self._reset_scorer()
            self.feedback = None
            self.total_points = 0
            self.state = "idle"
            self.game_ts_ms = 0
            self.countdown_deadline = None
            self.running_zero_monotonic = None

    @staticmethod
    def _decode_image(image_b64: str):
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _tick_state(self):
        if self.state == "countdown":
            ms_left = int((self.countdown_deadline - time.monotonic()) * 1000)
            if ms_left <= 0:
                self.state = "running"
                self.running_zero_monotonic = time.monotonic()
                self.game_ts_ms = 0
        elif self.state == "running":
            self.game_ts_ms = int((time.monotonic() - self.running_zero_monotonic) * 1000)

    def process_frame(self, image_b64: str):
        frame_bgr = self._decode_image(image_b64)
        if frame_bgr is None:
            return {"error": "invalid_frame"}

        self._tick_state()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._mp_ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._mp_ts_ms)

        landmarks_payload = None
        live_emb = None
        if result and result.pose_landmarks:
            lm = result.pose_landmarks[0]
            landmarks_payload = [{"x": p.x, "y": p.y, "z": p.z, "v": p.visibility} for p in lm]
            h, w = frame_bgr.shape[:2]
            live_pts = landmarks_to_pixels(lm, w, h)
            live_emb = pose_embedding(live_pts)

        if self.state == "running" and live_emb is not None and self.scoring_moves:
            finalized = self.scorer.update(
                ts_ms=self.game_ts_ms,
                live_emb=live_emb,
                similarity_fn=pose_similarity,
            )
            for res in finalized:
                tier_lower = (res.tier or "").lower()
                if tier_lower != "miss":
                    gained = max(0, int(round(res.best_score)))
                    self.total_points += gained
                    self.feedback = Feedback(
                        tier=tier_lower,
                        move_name=res.name,
                        move_score=res.best_score,
                        show_until_ms=self.game_ts_ms + 1000,
                    )

        if self.feedback and self.game_ts_ms > self.feedback.show_until_ms:
            self.feedback = None

        if self.display_moves:
            if self.state in ("running", "paused"):
                current_move, upcoming = find_current_and_upcoming(
                    self.display_moves,
                    self.game_ts_ms,
                    max_upcoming=4,
                    start_times=self.display_start_times,
                )
            else:
                current_move = self.display_moves[0]
                upcoming = self.display_moves[1:5]
        else:
            current_move = None
            upcoming = []

        countdown_ms_left = None
        if self.state == "countdown" and self.countdown_deadline is not None:
            countdown_ms_left = max(0, int((self.countdown_deadline - time.monotonic()) * 1000))

        return {
            "state": self.state,
            "game_ts_ms": self.game_ts_ms,
            "countdown_ms_left": countdown_ms_left,
            "total_points": self.total_points,
            "feedback": None
            if not self.feedback
            else {
                "tier": self.feedback.tier,
                "move_name": self.feedback.move_name,
                "move_score": self.feedback.move_score,
            },
            "pose_landmarks": landmarks_payload,
            "current_move": move_to_payload(current_move),
            "upcoming_moves": [move_to_payload(m) for m in upcoming],
        }
