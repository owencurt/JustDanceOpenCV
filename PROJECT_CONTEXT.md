# PROJECT_CONTEXT

## Purpose
This project is a **camera-based Just Dance prototype** that uses MediaPipe BlazePose + OpenCV to compare a player's live pose against authored choreography and award per-move scores.

## High-level architecture
- **`pose_author_editor.py`**: PyQt6 tool for authoring choreography JSON from a source video.
- **`blazepose_webcam.py`**: real-time gameplay app (webcam input, target-pose rendering, scoring HUD).
- **`scoring_engine.py`**: move-window evaluation engine (hit/hold classification, weighted aggregation, tiering).
- **`charts/*.json`**: authored move charts (timing + pose angles + normalized keypoint layout).

## Runtime flow (gameplay app)
1. Load two charts:
   - `SCORING_JSON_PATH`: sparse chart used for score evaluation.
   - `CHOREO_JSON_PATH`: display chart used for target pose UI.
2. Open webcam and run BlazePose in `VIDEO` mode each frame.
3. Convert live landmarks to angle embedding (elbows/shoulders/hips).
4. Feed embedding + game timestamp into `ScoringEngine.update(...)` while running.
5. Render composed UI:
   - centered current target pose,
   - right-side upcoming poses,
   - bottom-left PiP live camera with skeleton,
   - score + move-tier feedback overlays.
6. Controls: `SPACE` start/restart countdown, `P` pause/resume, `R` reset, `Q` quit.

## How pose detection is used
- BlazePose returns normalized landmarks.
- Runtime converts landmarks to pixel coordinates, then to joint-angle embedding.
- Authoring tool can initialize editable pose points from AI detection on a selected video frame.
- Export stores both:
  - `pose.norm_xy` (for drawing target skeletons),
  - `pose.angles` (for scoring reference embedding).

## Scoring/gameplay logic
- Each move has a time window derived from move spacing:
  - **hit** if next gap `< HIT_GAP_MS`, else **hold**.
- Frame scores in the window are aggregated via weighted percentile.
- Timing penalty is applied based on peak-response frame offset.
- Stability requires consecutive frames at/above tier thresholds.
- Critical-joint error cap can downgrade high tiers to `good`.
- Final tier is one of: `perfect`, `great`, `good`, `ok`, `miss`.

## Dependencies
- **OpenCV (`cv2`)**: camera/video IO + rendering.
- **MediaPipe Tasks Vision**: BlazePose detection/landmarks.
- **NumPy**: lightweight frame/image helpers and beat jitter sampling.
- **PyQt6 (+ QtMultimedia)**: desktop authoring editor + audio/video timeline control.

## Assumptions / constraints
- Pose model file is expected at `models/pose_landmarker_full.task`.
- Charts are expected to contain hips-up angle keys used by scorer.
- Main gameplay targets low-latency per-frame processing; avoid heavy allocations inside the loop.
- Current choreography aspect is assumed 16:9 for target-pose rendering.

## Known issues / future improvement candidates
- `SCORING_JSON_PATH` and `CHOREO_JSON_PATH` can diverge; keep intentional when editing.
- Authoring auto-fill samples many frames and can be slow on long videos.
- Timing/smoothing constants are tuned heuristically and may need song-specific tuning.
- Mirroring flags are currently preserved in chart data but not heavily used in runtime logic.

## Intentionally left alone
- Scoring thresholds/penalties and move classification constants in `scoring_engine.py` were not changed to avoid gameplay feel regressions.
- MediaPipe confidence parameters were preserved for stability with existing charts.

## Safe modification guide for future AI agents
1. Prefer small edits and validate with `python -m py_compile ...` before larger refactors.
2. Keep scoring embedding keys consistent across editor/export/runtime (`l_elbow`, `r_elbow`, etc.).
3. If changing per-frame logic in `blazepose_webcam.py`, measure impact on responsiveness.
4. When changing chart schema, update both exporter (`pose_author_editor.py`) and loader (`scoring_engine.py`).
5. Avoid deleting “unused-looking” chart fields unless all producers/consumers are audited.
