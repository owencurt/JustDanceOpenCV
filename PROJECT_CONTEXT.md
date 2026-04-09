# PROJECT_CONTEXT

## Purpose
This project is a camera-based Just Dance prototype. It now includes both:
- a **browser frontend gameplay UI** backed by Python APIs, and
- the legacy **OpenCV-only gameplay window** for compatibility.

## Current architecture
- `backend_server.py`: FastAPI server + realtime runtime loop.
- `pose_pipeline.py`: shared pose embedding/similarity/drawing helpers.
- `scoring_engine.py`: move-window evaluation engine.
- `frontend/`: browser HUD/control UI.
- `blazepose_webcam.py`: legacy monolithic OpenCV UI runtime.
- `pose_author_editor.py`: PyQt6 chart authoring tool.

## Runtime flow (new frontend path)
1. Backend loads scoring/display charts.
2. Webcam frame → BlazePose landmarks.
3. Landmarks → angle embedding (`pose_pipeline.pose_embedding`).
4. Embedding + game timestamp → `ScoringEngine.update(...)`.
5. Backend emits state snapshots over WebSocket.
6. Frontend renders score/combo/feedback/current+upcoming moves.
7. Frontend controls session/options via REST.
8. Health/debug info available at `/api/health` and in `runtime_error` field from `/api/config` + `/ws/state`.

## JSON chart model
- `pose.angles`: reference scoring embedding.
- `pose.norm_xy`: normalized landmark layout for move visualization.
- `start_ms`: move timing anchor.

## Constraints
- Model file required at `models/pose_landmarker_full.task`.
- Browser reference playback uses resolved `video_url` from backend metadata; absolute authoring paths may require placing media under `media/` or `reference_poses/`.

## Known limitations
- MJPEG webcam transport (not WebRTC).
- Single-camera/single-player runtime.
