# Architecture

## Overview

The app now follows a backend/frontend split:

1. **Python backend loop (`backend_server.py`)**
   - Captures webcam frames via OpenCV.
   - Runs MediaPipe BlazePose per frame.
   - Converts landmarks to angle embeddings.
   - Scores against chart moves with `ScoringEngine`.
   - Publishes game state and webcam feed.

2. **Browser frontend (`frontend/`)**
   - Renders game HUD and controls.
   - Subscribes to live state over WebSocket.
   - Displays webcam via MJPEG endpoint.
   - Draws current/upcoming move silhouettes from `norm_xy`.

## Pose detection and scoring ownership

- Pose detection: backend only.
- Scoring/comparison: backend only (`scoring_engine.py` + `pose_pipeline.py`).
- Frontend: visualization + input/control only.

## Communication layer

- **WebSocket**: realtime state snapshots (`/ws/state`).
- **REST**: control endpoints (`/api/session/*`, `/api/options`).
- **MJPEG HTTP stream**: webcam feed (`/video/feed`).

## Why this architecture

- Reuses existing stable Python scoring logic.
- Avoids moving heavy CV logic into frontend.
- Supports cleaner UX iteration independently from detection pipeline.
- Keeps migration incremental (legacy OpenCV UI still present).

## Main modules

- `backend_server.py`: orchestration runtime and service layer.
- `pose_pipeline.py`: shared embedding/similarity helper functions.
- `scoring_engine.py`: move-window classification and tier output.
- `frontend/app.js`: state binding + pose canvas rendering.
