# JustDanceOpenCV

A Just Dance-style prototype with a **Python pose/scoring backend** and a new **browser frontend game UI**.

## What changed

The project now supports a frontend-driven UX while preserving the existing scoring pipeline:

- ✅ MediaPipe BlazePose detection still runs in Python.
- ✅ Existing JSON choreography + `ScoringEngine` move evaluation is reused.
- ✅ A FastAPI server now exposes:
  - MJPEG webcam feed (`/video/feed`)
  - realtime game state over WebSocket (`/ws/state`)
  - session/options APIs (`/api/*`)
- ✅ A web UI (in `frontend/`) shows:
  - webcam view
  - current move card
  - upcoming move queue
  - live score + combo + feedback
  - session controls (start/pause/reset)
  - options/settings toggles
  - reference media controls

The original OpenCV-only gameplay app (`blazepose_webcam.py`) is kept for compatibility.

---

## Architecture

See [architecture.md](architecture.md) for details.

At a glance:

- `backend_server.py`: backend runtime loop + API/WebSocket/stream endpoints.
- `pose_pipeline.py`: shared pose embedding/similarity/drawing helpers.
- `scoring_engine.py`: timing-window move scoring and tiering.
- `frontend/`: browser UI (`index.html`, `app.js`, `styles.css`).
- `charts/*.json`: choreography source-of-truth.

---

## Requirements

- Python 3.10+
- Webcam
- Model file: `models/pose_landmarker_full.task`

Install backend dependencies:

```bash
pip install -r requirements.txt
```

---

## Run

### New frontend + backend flow (recommended)

```bash
uvicorn backend_server:app --reload --host 0.0.0.0 --port 8000
```

Then open: `http://localhost:8000`

### Legacy OpenCV window runtime

```bash
python blazepose_webcam.py
```

---

## Data contract (frontend/backend)

### WebSocket: `ws://localhost:8000/ws/state`

Payload includes:

- `status` (`idle`, `countdown`, `running`, `paused`)
- `game_ts_ms`
- `score`, `combo`, `best_combo`
- `feedback` (`tier`, `gained`, `move_name`, timeout)
- `current_move` (`name`, `start_ms`, `norm_xy`)
- `upcoming_moves[]`
- `options`
- `chart_meta`
  - includes `video_url` (resolved browser-playable URL if source media is found locally)

### REST

- `POST /api/session/start`
- `POST /api/session/pause-toggle`
- `POST /api/session/reset`
- `POST /api/options`
- `GET /api/config`
- `GET /api/health`

### Video stream

- `GET /video/feed` (MJPEG)

---

## JSON choreography format

Charts remain unchanged and should provide:

- top-level metadata (`title`, `video_path`, `bpm`, `offset_ms`)
- `moves[]` with:
  - `start_ms`
  - `pose.angles` (used for scoring)
  - `pose.norm_xy` (used for frontend pose drawing)

---

## Configuration

Current defaults in `backend_server.py`:

- `CHOREO_JSON_PATH = charts/ymca_extra.json`
- `SCORING_JSON_PATH = charts/ymca.json`
- `MODEL_PATH = models/pose_landmarker_full.task`

---

## Known limitations

- Reference media playback now uses resolved `chart_meta.display.video_url`. If chart `video_path` is an absolute authoring path, backend tries local fallbacks (`media/<basename>` then `reference_poses/<basename>`).
- MJPEG stream is simple and broadly compatible, but not as bandwidth-efficient as WebRTC.
- Single-player / single-webcam runtime.

---

## Troubleshooting start button / session not advancing

If clicking **Start** keeps the app in idle/countdown or doesn’t score:

1. Check backend health endpoint:
   - `http://localhost:8000/api/health`
2. Confirm model exists at:
   - `models/pose_landmarker_full.task`
3. Confirm webcam is not blocked by another app.
4. Open browser devtools and verify `POST /api/session/start` returns `200`.
5. Verify `GET /api/config` shows `game_ts_ms` increasing while running. If it is flat, check webcam access and backend logs.

The frontend now shows backend runtime errors directly in the feedback area, and it falls back to polling when WebSocket is unavailable.

---

## Extension points

- Replace MJPEG with WebRTC/WebCodecs for lower-latency camera transport.
- Add chart/song selector endpoint and frontend screen.
- Add multiplayer leaderboard state in backend.
- Add richer judgement timing bars in frontend.
