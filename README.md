# JustDanceOpenCV

A lightweight **Just Dance-style prototype** powered by **MediaPipe BlazePose**.

This repository now supports two runtimes:

1. **Legacy desktop gameplay** (`python blazepose_webcam.py`) for OpenCV-window playback.
2. **New browser runtime** (frontend + Python backend), where the full game experience is used in the browser.

---

## What changed architecturally

The project was refactored from a single OpenCV desktop loop into a **web-friendly split architecture**:

- **Frontend (React + Vite)**
  - Runs in the browser.
  - Owns webcam capture (`getUserMedia`) and UI rendering.
  - Sends compressed webcam frames to backend over WebSocket.
  - Draws gameplay HUD (score, tiers, current pose, upcoming poses, webcam PiP).

- **Backend (FastAPI + MediaPipe + existing scoring engine)**
  - Keeps pose detection and scoring in Python.
  - Reuses the existing `scoring_engine.py` behavior and thresholds.
  - Maintains game state (idle/countdown/running/paused), move finalization, and point accumulation.
  - Streams per-frame results back to frontend.

This preserves core gameplay logic while making the UI easier to style and iterate going forward.

---

## New project structure overview

```text
.
├── backend/
│   ├── __init__.py
│   ├── game_logic.py          # Shared pose embedding/similarity helpers
│   ├── main.py                # FastAPI app + API/WS endpoints
│   └── session.py             # Per-WebSocket game session state
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx            # Browser gameplay UI + webcam loop
│       ├── main.jsx
│       └── styles.css
├── charts/
│   ├── ymca.json
│   └── ymca_extra.json
├── blazepose_webcam.py        # Legacy desktop runtime (kept)
├── pose_author_editor.py      # Existing chart authoring editor (kept)
├── scoring_engine.py          # Existing scoring logic (reused)
├── run_browser_dev.sh         # Convenience script to run backend + frontend
└── requirements.txt
```

---

## Dependencies

### Python dependencies
Install from `requirements.txt`:

- `opencv-python`
- `numpy`
- `mediapipe`
- `PyQt6` (needed for existing desktop authoring editor)
- `fastapi`
- `uvicorn[standard]`

### Frontend dependencies
Defined in `frontend/package.json`:

- `react`
- `react-dom`
- `vite`
- `@vitejs/plugin-react`

### Model asset (required)
You still need:

```text
models/pose_landmarker_full.task
```

---

## Backend setup instructions

### 1) Create/activate Python virtual environment

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run backend

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

API sanity check:

```bash
curl http://127.0.0.1:8000/api/health
```

---

## Frontend setup instructions

In a second terminal:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Then open:

```text
http://127.0.0.1:5173
```

---

## How to run locally now (recommended flow)

### Option A (two terminals)
1. Terminal 1: run backend (`uvicorn ...`).
2. Terminal 2: run frontend (`npm run dev ...`).
3. Open `http://127.0.0.1:5173`.
4. Allow camera access in browser.
5. Use on-screen buttons: **Start / Pause / Resume / Reset**.

### Option B (single command helper)

```bash
./run_browser_dev.sh
```

This starts backend first, then frontend dev server.

---

## Environment variables / config

No required environment variables were introduced for default local use.

Current defaults:
- Backend port: `8000`
- Frontend dev port: `5173`
- Default scoring chart: `charts/ymca.json`
- Default display chart: `charts/ymca_extra.json`
- Pose model path: `models/pose_landmarker_full.task`

If you need different charts, use the chart selectors in the browser UI.

---

## Webcam access in browser version

- Webcam permission is requested by the browser via `navigator.mediaDevices.getUserMedia`.
- Browser captures frames and sends JPEG-compressed images over WebSocket to backend.
- Backend runs MediaPipe pose detection + scoring and returns per-frame state.
- Frontend renders:
  - target pose,
  - upcoming poses,
  - live mirrored PiP webcam + detected skeleton,
  - score/tier/status overlays.

---

## Preserved functionality

The browser version keeps the core gameplay behavior:

- Webcam-driven pose detection.
- Pose-angle embedding and similarity scoring.
- Existing `ScoringEngine` move-window logic and thresholds.
- Tier results (`perfect/great/good/ok/miss`) and score accumulation.
- Reference move handling via chart JSON (`pose.norm_xy` + `pose.angles`).
- Gameplay state flow (`idle → countdown → running → paused/reset`).

Legacy scripts are still present for compatibility:
- `blazepose_webcam.py`
- `pose_author_editor.py`

---

## Chart media (new)

Browser gameplay now supports synchronized audio playback from a chart-associated MP4 file.

### Chart JSON field

Add a media field at the chart root:

```json
{
  "schema_version": 2,
  "title": "YMCA",
  "media_path": "charts/video_path.mp4",
  "moves": [ ... ]
}
```

- `media_path` is the preferred field (source of truth).
- `video_path` is still read as a backward-compatible fallback.
- If a chart has no media field, gameplay still runs silently.

### Where to place media files

- Put media files under `charts/` (for example: `charts/video_path.mp4`).
- Backend serves that folder at `/charts/*`, and frontend uses that URL during gameplay.
- In local dev, Vite proxies `/charts` to the backend.

### Audio controls

- Browser UI now includes an **Audio: On/Off** toggle in the toolbar.
- Default is **Audio On**.
- Turning audio off pauses chart media immediately while gameplay/scoring continues.
- Turning audio on seeks to the current game timestamp and resumes playback so timing stays aligned.

### Timing + autoplay behavior

- Gameplay timing remains driven by backend game clock (`game_ts_ms`).
- Frontend continuously syncs media playback position to that clock to avoid drift.
- Frame-upload flow now resets its in-flight send lock on WebSocket reconnect/close so countdown can always progress into running state after reconnects or chart changes.
- If autoplay is blocked by browser policy, use a user interaction (for example Audio toggle, Start/Resume) and then resume; gameplay still runs even if audio cannot start.

---

## Known limitations / follow-up recommendations

1. **Network hop in local loop:**
   Webcam frames now cross browser → backend WebSocket, so latency depends on frame size/FPS and CPU.
   - Current frontend sends ~15 FPS at 640x360 JPEG for practical responsiveness.

2. **Pose rendering parity:**
   The browser UI reproduces layout/functionality but not pixel-identical OpenCV visuals.

3. **Production deployment hardening:**
   For production, add:
   - auth/session limits,
   - structured logging,
   - configurable CORS,
   - HTTPS and secure WSS.

---

## Legacy desktop usage (still available)

```bash
python blazepose_webcam.py
```

And pose authoring editor:

```bash
python pose_author_editor.py
```

---

## Quick validation commands

```bash
python -m py_compile backend/main.py backend/session.py backend/game_logic.py scoring_engine.py blazepose_webcam.py pose_author_editor.py
python -m compileall backend
cd frontend && npm run build
```

---

## License

No license file is currently present in this repository.
