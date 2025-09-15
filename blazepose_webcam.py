# blazepose_webcam.py
import time
import cv2
import numpy as np
import mediapipe as mp
import math

from scoring_engine import (
    load_choreography, ScoringEngine, DEFAULT_THRESHOLDS
)

# --- MediaPipe short aliases ---
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

# --- Config: JSON choreo + scoring window + thresholds ---
CHOREO_JSON_PATH = "charts/test.json"     # <-- set this to your JSON file
WINDOW_HALF_MS = 150                      # default ±250ms; tweak as desired
TIER_THRESHOLDS = DEFAULT_THRESHOLDS      # or override: {"perfect": 90, "great": 78, ...}

# --- Start/Control ---
START_COUNTDOWN_MS = 3000                 # 3-2-1 before the timer starts
INSTRUCTIONS = "SPACE to start • P to pause/resume • R to reset • Q to quit"

# --- Drawing params ---
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
MODEL_PATH = "models/pose_landmarker_full.task"  # _lite / _full / _heavy

LANDMARK_RADIUS = 3
LINE_THICKNESS = 2
KP_COLOR = (0, 255, 0)
BONE_COLOR = (240, 240, 240)
TEXT_COLOR = (255, 255, 255)

# --- Landmark indices ---
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28

# --- Weights (as in your similarity code) ---
REGION_WEIGHTS = {"upper": 1.0, "core": 0.6, "lower": 0}
PER_JOINT_WEIGHTS = {
    "l_elbow": 1.0, "r_elbow": 1.0,
    "l_shoulder": 0.9, "r_shoulder": 0.9,
    "l_hip": 0.6, "r_hip": 0.6,
    "l_knee": 0, "r_knee": 0,
}

def _angle(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = (ax - bx, ay - by); v2 = (cx - bx, cy - by)
    v1_len = math.hypot(*v1); v2_len = math.hypot(*v2)
    if v1_len == 0 or v2_len == 0:
        return None
    dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (v1_len * v2_len)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

def _joint_weight(name):
    region = ("upper" if "elbow" in name or "shoulder" in name else
              "lower" if "knee" in name or "ankle" in name else
              "core")
    base = REGION_WEIGHTS.get(region, 1.0)
    return PER_JOINT_WEIGHTS.get(name, base)

def landmarks_to_pixels(landmarks_norm_list, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm_list]

def pose_embedding(pts):
    """
    Upper-body-only embedding (hips & up). Excludes knees/ankles entirely.
    """
    emb = {
        "l_elbow":    _angle(pts[L_SHOULDER], pts[L_ELBOW],  pts[L_WRIST]),
        "r_elbow":    _angle(pts[R_SHOULDER], pts[R_ELBOW],  pts[R_WRIST]),
        "l_shoulder": _angle(pts[L_HIP],      pts[L_SHOULDER], pts[L_ELBOW]),
        "r_shoulder": _angle(pts[R_HIP],      pts[R_SHOULDER], pts[R_ELBOW]),
        "l_hip":      _angle(pts[L_SHOULDER], pts[L_HIP],    pts[L_KNEE]),
        "r_hip":      _angle(pts[R_SHOULDER], pts[R_HIP],    pts[R_KNEE]),
    }
    return emb

# --- Similarity (your existing tolerant score, 0..100 higher=better) ---
JOINT_TOLERANCE_DEG = {
    "l_elbow": 16.0, "r_elbow": 16.0,
    "l_shoulder": 18.0, "r_shoulder": 18.0,
    "l_hip": 20.0, "r_hip": 20.0,
}
SCORE_DEG_SCALE = 100.0  # average post-tolerance error of 120° → ~0 score

def pose_similarity(emb_a, emb_b):
    total_w = 0.0
    weighted_err = 0.0
    per_joint_err = {}

    for k, a in emb_a.items():
        b = emb_b.get(k)
        if a is None or b is None:
            continue
        diff = abs(a - b)
        if diff > 180:
            diff = 360 - diff
        tol = JOINT_TOLERANCE_DEG.get(k, 0.0)
        diff = max(0.0, diff - tol)
        w = _joint_weight(k)
        weighted_err += w * diff
        total_w += w
        per_joint_err[k] = (diff, w)

    if total_w == 0:
        return 0.0, {"reason": "no comparable joints"}

    avg_err = weighted_err / total_w
    score = max(0.0, 100.0 * (1.0 - (avg_err / SCORE_DEG_SCALE)))
    return score, {"avg_err_deg": avg_err, "per_joint_err": per_joint_err}

def draw_landmarks(frame_bgr, landmarks_norm_list, kp_color=KP_COLOR, bone_color=BONE_COLOR,
                   radius=LANDMARK_RADIUS, thickness=LINE_THICKNESS):
    if not landmarks_norm_list:
        return
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm_list]
    for a, b in POSE_CONNECTIONS:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], bone_color, thickness, cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(frame_bgr, (x, y), radius, kp_color, -1, lineType=cv2.LINE_AA)

def draw_norm_xy_pose_scaled(frame_bgr, norm_xy_dict, top_left, size, 
                             kp_color=(255, 200, 0), bone_color=(220, 220, 220),
                             radius=2, thickness=1):
    """
    Draw a mini stick-figure using the JSON's pose.norm_xy (dict: str(index)->[x,y], 0..1).
    top_left: (x0, y0) pixel of the miniature.
    size: (w, h) pixel box to draw in (preserves norm positions within this box).
    Only draws connections where both endpoints exist in norm_xy_dict.
    """
    if not norm_xy_dict:
        return
    x0, y0 = top_left
    box_w, box_h = size

    # Convert keys to ints and points to px
    pts = {}
    for k, (nx, ny) in norm_xy_dict.items():
        try:
            idx = int(k)
        except Exception:
            continue
        px = int(x0 + nx * box_w)
        py = int(y0 + ny * box_h)
        pts[idx] = (px, py)

    # Bones
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame_bgr, pts[a], pts[b], bone_color, thickness, cv2.LINE_AA)
    # Keypoints (draw only the ones we have)
    for (px, py) in pts.values():
        cv2.circle(frame_bgr, (px, py), radius, kp_color, -1, lineType=cv2.LINE_AA)

# --- Overlay helpers ---
def draw_text_shadow(img, text, org, scale=0.7, color=(255,255,255), thickness=2):
    x, y = org
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     cv2.FONT_HERSHEY_SIMPLEX, scale, color,          thickness, cv2.LINE_AA)

def draw_countdown_overlay(frame_bgr, ms_left):
    secs = max(0, int((ms_left + 999) // 1000))
    text = f"Starting in {secs}..."
    draw_text_shadow(frame_bgr, text, (10, 52), scale=0.9, color=(60, 240, 120), thickness=2)

def draw_idle_overlay(frame_bgr, moves, window_half_ms):
    draw_text_shadow(frame_bgr, "Ready.", (10, 28), scale=0.9, color=(200, 200, 255))
    draw_text_shadow(frame_bgr, INSTRUCTIONS, (10, 56), scale=0.7, color=(200, 200, 200))
    if moves:
        first = moves[0]
        draw_text_shadow(frame_bgr, f"First move '{first.name}' at {first.start_ms} ms (window ±{window_half_ms} ms)",
                         (10, 84), scale=0.7, color=(200, 200, 200))

def draw_upcoming_overlay(frame_bgr, moves, game_ts_ms, max_items=6):
    """Top-right panel: next move ETA + progress bar + tiny ghost of the next pose, then the following moves."""
    H, W = frame_bgr.shape[:2]
    pad = 10
    line_h = 22
    x0 = W - 360
    y0 = 20
    panel_w = 350

    # Filter upcoming moves
    upcoming = [m for m in moves if m.start_ms >= game_ts_ms][:max_items]

    # Estimate panel height (header + next line + bar + ghost + others)
    ghost_h = 110     # height in px for the tiny ghost
    ghost_gap = 8
    base_h = line_h * 2 + pad*2                   # header + one line spacer
    bar_h  = 10 + 12                               # bar + gap below it
    if upcoming:
        others = max(0, len(upcoming) - 1)
        panel_h = base_h + bar_h + ghost_h + ghost_gap + (others * line_h)
    else:
        panel_h = base_h

    # Translucent panel bg
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+panel_h), (30, 30, 30), -1)
    frame_bgr[:] = cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0)

    # Header
    draw_text_shadow(frame_bgr, "UPCOMING", (x0+pad, y0+pad+16), scale=0.7, color=(180, 220, 255))

    y = y0 + pad + 16 + 8

    if not upcoming:
        draw_text_shadow(frame_bgr, "No more moves.", (x0+pad, y+line_h), scale=0.7, color=(200,200,200))
        return

    # First (next) move with progress bar
    next_move = upcoming[0]
    dt_ms = max(0, next_move.start_ms - game_ts_ms)
    draw_text_shadow(frame_bgr, f"Next: {next_move.name}  in {dt_ms/1000:.2f}s",
                     (x0+pad, y+line_h), scale=0.75, color=(60,240,120))

    # Progress bar: 0 at 3s out → full at 0s
    BAR_MAX_MS = 3000
    frac = 1.0 - min(1.0, dt_ms / BAR_MAX_MS)
    bar_x = x0+pad
    bar_y = y + line_h + 8
    bar_w = panel_w - 2*pad
    bar_h_px = 10
    cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h_px), (80,80,80), 1, cv2.LINE_AA)
    cv2.rectangle(frame_bgr, (bar_x+1, bar_y+1), (bar_x+1+int((bar_w-2)*frac), bar_y+bar_h_px-1), (60,240,120), -1, cv2.LINE_AA)

    # Tiny ghost of the upcoming pose, using JSON pose.norm_xy
    ghost_top = bar_y + bar_h_px + 8
    ghost_left = x0 + pad
    ghost_w = panel_w - 2*pad
    ghost_h = 110  # set above
    norm_xy = None
    try:
        norm_xy = (next_move.raw.get("pose", {}) or {}).get("norm_xy", None)
    except Exception:
        norm_xy = None
    if norm_xy:
        draw_norm_xy_pose_scaled(
            frame_bgr,
            norm_xy_dict=norm_xy,
            top_left=(ghost_left, ghost_top),
            size=(ghost_w, ghost_h),
            kp_color=(255, 200, 0),     # gold-ish
            bone_color=(220, 220, 220),
            radius=2,
            thickness=1
        )

    # The rest of the upcoming list
    y2 = ghost_top + ghost_h + 12
    for m in upcoming[1:]:
        d = max(0, m.start_ms - game_ts_ms)
        draw_text_shadow(frame_bgr, f"{m.name}  in {d/1000:.2f}s", (x0+pad, y2), scale=0.7, color=(220,220,220))
        y2 += line_h


def main():
    # Webcam init
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open webcam. Check OS camera permissions.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Load choreography from JSON
    try:
        moves = load_choreography(CHOREO_JSON_PATH)
        print(f"[i] Loaded {len(moves)} moves from {CHOREO_JSON_PATH}")
    except Exception as e:
        print(f"[!] Could not load choreography: {e}")
        moves = []

    # Create scoring engine (JSON-based)
    scorer = ScoringEngine(
        moves=moves,
        window_half_ms=WINDOW_HALF_MS,
        thresholds=TIER_THRESHOLDS,
        tie_breaker="closest"
    )

    last_move_feedback = None  # (tier, score, name, show_until_ms)

    # MediaPipe Pose
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    # --- State machine for game time (separate from MediaPipe time) ---
    state = "idle"  # "idle" -> "countdown" -> "running" (also "paused")
    game_ts_ms = 0
    countdown_deadline = None
    running_zero_monotonic = None  # wall clock when we started running

    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_ts_ms = 0  # monotonically increasing for MediaPipe only
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    print("[!] Frame grab failed.")
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # advance MediaPipe clock ~30fps regardless of game state
                mp_ts_ms += 33
                result = landmarker.detect_for_video(mp_image, mp_ts_ms)

                # Draw live skeleton if present
                if result and result.pose_landmarks:
                    draw_landmarks(frame_bgr, result.pose_landmarks[0])
                    cv2.putText(frame_bgr, "BlazePose (Full) — VIDEO mode", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)

                # --- Update GAME clock based on state ---
                if state == "idle":
                    # show instructions + first move info
                    draw_idle_overlay(frame_bgr, moves, WINDOW_HALF_MS)

                elif state == "countdown":
                    # count down in real time
                    ms_left = int((countdown_deadline - time.monotonic()) * 1000)
                    if ms_left <= 0:
                        state = "running"
                        running_zero_monotonic = time.monotonic()
                        game_ts_ms = 0
                    else:
                        draw_countdown_overlay(frame_bgr, ms_left)

                elif state == "running":
                    # set game ts from real time since start
                    game_ts_ms = int((time.monotonic() - running_zero_monotonic) * 1000)

                elif state == "paused":
                    # game_ts_ms frozen; show pause label
                    draw_text_shadow(frame_bgr, "PAUSED", (10, 52), scale=0.9, color=(200, 200, 80))

                # --- JSON scoring only when running ---
                live_emb = None
                if result and result.pose_landmarks:
                    # Build live embedding (angles)
                    h, w = frame_bgr.shape[:2]
                    live_pts = landmarks_to_pixels(result.pose_landmarks[0], w, h)
                    live_emb = pose_embedding(live_pts)

                if state == "running" and live_emb is not None and moves:
                    finalized = scorer.update(
                        ts_ms=game_ts_ms,        # <-- game clock
                        live_emb=live_emb,
                        similarity_fn=pose_similarity
                    )
                    # Any moves that just finalized?
                    for res in finalized:
                        print(f"[json] Move {res.name} @ {res.start_ms} ms → {res.tier.upper()} "
                              f"({res.best_score:.1f}) | best frame ts: {res.best_ts_ms}")
                        # show on-screen feedback for ~1 second
                        last_move_feedback = (res.tier, res.best_score, res.name, game_ts_ms + 1000)

                # HUD for last finalized move
                if last_move_feedback is not None:
                    tier_txt, tier_score, move_name, show_until = last_move_feedback
                    if game_ts_ms <= show_until:
                        txt = f"Move {move_name}: {tier_txt.upper()} ({tier_score:.0f})"
                        cv2.putText(frame_bgr, txt, (10, 86),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 240, 120), 2, cv2.LINE_AA)
                    else:
                        last_move_feedback = None

                # Upcoming overlay (works in running/paused; in idle it will show "No more" until start)
                if moves and state in ("running", "paused"):
                    draw_upcoming_overlay(frame_bgr, moves, game_ts_ms, max_items=6)

                # Controls overlay (always)
                draw_text_shadow(frame_bgr, INSTRUCTIONS, (10, frame_bgr.shape[0]-12), scale=0.6, color=(200,200,200))

                cv2.imshow("BlazePose Webcam", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # SPACE to start from idle or restart countdown
                    if state in ("idle", "paused"):
                        state = "countdown"
                        countdown_deadline = time.monotonic() + START_COUNTDOWN_MS / 1000.0
                    elif state == "running":
                        # restart countdown mid-run: treat as reset+start
                        scorer = ScoringEngine(moves=moves, window_half_ms=WINDOW_HALF_MS,
                                               thresholds=TIER_THRESHOLDS, tie_breaker="closest")
                        last_move_feedback = None
                        state = "countdown"
                        countdown_deadline = time.monotonic() + START_COUNTDOWN_MS / 1000.0
                elif key == ord('p'):
                    if state == "running":
                        state = "paused"
                    elif state == "paused":
                        # resume: keep same game_ts_ms baseline by shifting zero time
                        running_zero_monotonic = time.monotonic() - (game_ts_ms / 1000.0)
                        state = "running"
                elif key == ord('r'):
                    # hard reset
                    scorer = ScoringEngine(moves=moves, window_half_ms=WINDOW_HALF_MS,
                                           thresholds=TIER_THRESHOLDS, tie_breaker="closest")
                    last_move_feedback = None
                    state = "idle"
                    game_ts_ms = 0
                    countdown_deadline = None
                    running_zero_monotonic = None

        except KeyboardInterrupt:
            print("\n[i] Stopping (Ctrl+C).")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if moves:
                leftovers = scorer.finalize_all()
                for res in leftovers:
                    print(f"[finalize] {res.name} → {res.tier.upper()} ({res.best_score:.1f})")

if __name__ == "__main__":
    main()
