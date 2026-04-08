# JustDanceOpenCV

A lightweight **Just Dance-style prototype** built with **OpenCV** and **MediaPipe BlazePose**.

It includes:
- A **live gameplay app** (`blazepose_webcam.py`) that scores your webcam pose against a choreography chart.
- A **pose authoring editor** (`pose_author_editor.py`) to create/edit choreography JSON from video.
- A reusable **scoring engine** (`scoring_engine.py`) that evaluates timing + pose quality.
- A **browser runtime scaffold** (`webapp/backend.py` + `webapp/static/`) for modern UI + reliable media playback.

---

## Features

- Real-time webcam pose tracking with MediaPipe Tasks BlazePose.
- Pose similarity scoring using joint angles (elbows/shoulders/hips).
- Tiered move results (`perfect`, `great`, `good`, `ok`, `miss`) and cumulative points.
- Countdown/start/pause/reset controls for gameplay.
- Authoring tool to:
  - open a source video,
  - initialize poses from BlazePose AI detection,
  - drag/edit joints manually,
  - add/copy/paste/delete timed moves,
  - export chart JSON.

---

## Repository structure

```text
.
├── blazepose_webcam.py      # Main webcam gameplay runtime
├── pose_author_editor.py    # PyQt6 chart authoring tool
├── scoring_engine.py        # Move-window scoring logic
├── charts/
│   ├── ymca.json            # Example scoring chart
│   └── ymca_extra.json      # Example display chart
├── reference_poses/
│   └── yoga.webp            # Reference media
└── PROJECT_CONTEXT.md       # Concise maintainer context for future AI/dev work
```

---

## Requirements

- Python **3.10+** (3.11 recommended)
- Webcam (for gameplay)
- OS audio/video support for PyQt6 multimedia (for editor playback)

Python packages used:
- `opencv-python`
- `numpy`
- `mediapipe`
- `PyQt6`

> Note: This project expects a MediaPipe pose landmarker model file at:
> `models/pose_landmarker_full.task`

---

## Setup from scratch

### 1) Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd JustDanceOpenCV
```

### 2) Create and activate a virtual environment

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

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install opencv-python numpy mediapipe PyQt6
```

### 4) Download BlazePose model

Create a `models/` folder and place the model file as:

```text
models/pose_landmarker_full.task
```

You can use the official MediaPipe Tasks pose landmarker model assets from Google/MediaPipe documentation.

---

## Run the apps

### Gameplay runtime (webcam scoring)

```bash
python blazepose_webcam.py
```

Default chart config is in the top of the file:
- `SCORING_JSON_PATH` → scoring target chart
- `CHOREO_JSON_PATH` → displayed target poses

### Browser runtime (new UI scaffold)

```bash
pip install fastapi uvicorn
uvicorn webapp.backend:app --reload
```

Then open `http://127.0.0.1:8000`.

This runtime:
- renders choreography timing in a browser UI,
- plays reference media with native browser `<video>` playback (audio + video),
- includes separate audio/video toggles,
- is designed as the migration path for moving UX out of raw OpenCV windows while keeping Python as backend.

### Pose authoring editor

```bash
python pose_author_editor.py
```

Use this flow:
1. Open video.
2. Set BPM/offset.
3. Init pose from frame (AI) or blank pose.
4. Adjust joints manually if needed.
5. Add moves with time stamps.
6. Export JSON into `charts/`.

---

## Gameplay controls

In the gameplay window:
- `SPACE`: start countdown / restart run
- `P`: pause/resume
- `R`: reset to idle
- `Q`: quit

---

## Chart format (overview)

Each chart JSON includes metadata and a list of moves:
- `start_ms`: move timing
- `pose.angles`: scoring embedding reference
- `pose.norm_xy`: normalized keypoint layout for rendering target skeletons

The scoring engine loads and sorts `moves` by `start_ms`.

---

## Common troubleshooting

### Webcam fails to open
- Ensure camera permissions are enabled for your terminal/app.
- Close other apps that may be using the camera.

### No pose detected
- Improve lighting.
- Keep full upper body visible.
- Move farther from camera to include torso and arms.

### Editor has no audio/video sync
- Some codec/container combinations can drift.
- Re-encode source video to a common format (e.g., H.264 MP4).

### Model file errors
- Verify `models/pose_landmarker_full.task` exists.
- Confirm file name matches exactly.

---

## Development notes

- Keep scoring key names consistent across exporter/runtime:
  - `l_elbow`, `r_elbow`, `l_shoulder`, `r_shoulder`, `l_hip`, `r_hip`.
- If you change chart schema, update both:
  - exporter (`pose_author_editor.py`) and
  - loader (`scoring_engine.py`).
- Use `PROJECT_CONTEXT.md` for a concise architectural brief before making changes.

---

## Quick validation

```bash
python -m py_compile blazepose_webcam.py scoring_engine.py pose_author_editor.py
```

---

## License

No license file is currently present in this repository.
Add a license if you plan to distribute or open-source this project.
