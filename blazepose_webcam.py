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
CHOREO_JSON_PATH = "charts/ymca_extra.json"     # <-- set this to your JSON file
WINDOW_HALF_MS = 150                      # default ±250ms; tweak as desired
TIER_THRESHOLDS = DEFAULT_THRESHOLDS      # or override: {"perfect": 90, "great": 78, ...}

# --- Start/Control ---
START_COUNTDOWN_MS = 3000                 # 3-2-1 before the timer starts
INSTRUCTIONS = "SPACE to start - P to pause/resume - R to reset - Q to quit"

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

# --- Pose zoom constants ---
UPCOMING_POSE_ZOOM = 2.00   # >1.0 = bigger, <1.0 = smaller
CURRENT_POSE_ZOOM = 1.5

# --- Tier UI ---
TIER_COLORS = {
    "perfect": (80, 255, 120),
    "great":   (80, 200, 255),
    "good":    (255, 200, 80),
    "ok":      (240, 180, 80),
    "miss":    (90, 90, 90),
}

# --- Pose tweening (center target) ---
SMOOTH_TARGET_ENABLED = False   # ← flip to False to disable
EASE_MODE = "linear"            # "linear" | "cubic" | "quint"
CLAMP_TO_INTERVAL = True       # keep alpha in [0,1] strictly per beat interval
MIN_INTERVAL_MS = 120         # guard: if beat gap is tiny, skip tween to avoid jitter



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
    draw_text_shadow(frame_bgr, text, (10, 75), scale=0.9, color=(60, 240, 120), thickness=2)

def draw_idle_overlay(frame_bgr, moves, window_half_ms):
    draw_text_shadow(frame_bgr, "Ready.", (10, 75), scale=0.9, color=(200, 200, 255))

# ---------- Carousel/ghost helpers (used for left list) ----------
def _render_ghost_from_norm_xy(norm_xy, size=(96, 120)):
    """
    Render a tiny stick-figure image from pose.norm_xy onto a BGR image.
    """
    w, h = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if not norm_xy:
        return canvas

    # Convert to px inside this mini-canvas with a small margin
    margin = 6
    box_w = w - 2*margin
    box_h = h - 2*margin
    pts = {}
    for k, (nx, ny) in norm_xy.items():
        try:
            idx = int(k)
        except Exception:
            continue
        px = int(margin + nx * box_w)
        py = int(margin + ny * box_h)
        pts[idx] = (px, py)

    # Bones
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(canvas, pts[a], pts[b], (220, 220, 220), 1, cv2.LINE_AA)
    # Keypoints
    for (px, py) in pts.values():
        cv2.circle(canvas, (px, py), 2, (255, 200, 0), -1, lineType=cv2.LINE_AA)

    return canvas

def build_ghost_cache(moves, ghost_size=(96, 120)):
    """
    Pre-render per-move ghosts once to avoid per-frame work.
    Keyed by move.start_ms (or any unique field).
    """
    cache = {}
    for m in moves:
        norm_xy = None
        try:
            norm_xy = (m.raw.get("pose", {}) or {}).get("norm_xy", None)
        except Exception:
            pass
        cache[m.start_ms] = _render_ghost_from_norm_xy(norm_xy, ghost_size)
    return cache

def _truncate(text, max_chars):
    return (text[:max_chars-1] + "…") if len(text) > max_chars else text

# ---------- NEW LAYOUT HELPERS ----------
def find_current_and_upcoming(moves, game_ts_ms, max_upcoming=4):
    """
    Returns:
      current_move: the move whose start_ms is the latest <= game_ts_ms,
      upcoming: next max_upcoming moves after current_move
    """
    if not moves:
        return None, []
    past = [m for m in moves if m.start_ms <= game_ts_ms]
    if past:
        current = max(past, key=lambda m: m.start_ms)
        after = [m for m in moves if m.start_ms > current.start_ms]
    else:
        current = moves[0]
        after = moves[1:]
    return current, after[:max_upcoming]

def scale_norm_xy(norm_xy_dict, zoom=1.2):
    """Scale normalized [0..1] coords around centroid; clamp to [0,1]."""
    if not norm_xy_dict:
        return norm_xy_dict
    xs, ys = zip(*norm_xy_dict.values())
    cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
    out = {}
    for k, (nx, ny) in norm_xy_dict.items():
        sx = cx + (nx - cx) * zoom
        sy = cy + (ny - cy) * zoom
        out[k] = (min(1.0, max(0.0, sx)), min(1.0, max(0.0, sy)))
    return out

def draw_current_json_pose_center(frame_bgr, current_move, next_move=None, game_ts_ms=None):
    """
    Center-pane large target pose from JSON (pose.norm_xy), optionally tweened toward next pose.
    """
    if current_move is None:
        return
    H, W = frame_bgr.shape[:2]

    # Big centered box (uses whole canvas; you can keep your panel overlay if you like)
    box_w = int(W)
    box_h = int(H)
    x0 = (W - box_w) // 2
    y0 = (H - box_h) // 2

    # # Title
    # draw_text_shadow(frame_bgr, f"NOW: {current_move.name}", (x0 + 12, y0 + 36),
    #                  scale=0.8, color=(60, 240, 120), thickness=2)

    # --- pick base and (optionally) blended pose ---
    norm_xy_a = _get_pose_norm_xy(current_move)
    norm_xy_draw = norm_xy_a

    if (SMOOTH_TARGET_ENABLED 
        and game_ts_ms is not None 
        and next_move is not None 
        and hasattr(current_move, "start_ms") 
        and hasattr(next_move, "start_ms")):

        dt = next_move.start_ms - current_move.start_ms
        if dt >= MIN_INTERVAL_MS:
            # progress inside [current, next)
            p = (game_ts_ms - current_move.start_ms) / float(dt)
            if CLAMP_TO_INTERVAL:
                p = max(0.0, min(1.0, p))
            # Ease
            alpha = _ease(p, EASE_MODE)

            norm_xy_b = _get_pose_norm_xy(next_move)
            norm_xy_draw = _blend_norm_xy(norm_xy_a, norm_xy_b, alpha)

    if norm_xy_draw:
        # Optional size zoom, then aspect-preserving draw
        norm_xy_draw = scale_norm_xy(norm_xy_draw, zoom=CURRENT_POSE_ZOOM)
        draw_norm_xy_pose_fit(
            frame_bgr, norm_xy_dict=norm_xy_draw, top_left=(x0, y0), size=(box_w, box_h),
            target_aspect=CHOREO_ASPECT,
            kp_color=(255, 200, 0), bone_color=(230, 230, 230), radius=4, thickness=3
        )
    else:
        draw_text_shadow(frame_bgr, "(no pose data in JSON)", (x0+12, y0+64),
                         scale=0.7, color=(200,200,200), thickness=2)


def draw_upcoming_right_list(frame_bgr, upcoming, game_ts_ms, ghost_cache, max_items=4):
    """
    Right-side vertical list of the next N moves.
    Pose-only cards (no text). Top card gets a 'NEXT' tag (top-right).
    Designed for a white background.
    """
    if not upcoming:
        return

    H, W = frame_bgr.shape[:2]
    pad = 12
    item_w, item_h = 220, 140   # a bit taller since no text
    x = W - item_w - 16          # right edge
    y = 60

    for i, m in enumerate(upcoming[:max_items]):
        iy = y + i*(item_h + pad)

        # Card (white) with soft shadow/border
        bg = (255, 255, 255)
        border = (140, 140, 140) if i != 0 else (60, 180, 120)  # emphasize first card border
        cv2.rectangle(frame_bgr, (x, iy), (x+item_w, iy+item_h), bg, -1, cv2.LINE_AA)
        cv2.rectangle(frame_bgr, (x, iy), (x+item_w, iy+item_h), border, 2, cv2.LINE_AA)

        # Center the ghost in the card
        # Draw pose directly with aspect-preserving fit (no raster stretch)
        norm_xy = None
        try:
            norm_xy = (m.raw.get("pose", {}) or {}).get("norm_xy", None)
        except Exception:
            norm_xy = None

        # Apply zoom if we have a pose
        if norm_xy:
            norm_xy = scale_norm_xy(norm_xy, zoom=UPCOMING_POSE_ZOOM)

        inner_pad = 12
        draw_box_top_left = (x + inner_pad, iy + inner_pad)
        draw_box_size = (item_w - 2*inner_pad, item_h - 2*inner_pad)

        draw_norm_xy_pose_fit(
            frame_bgr,
            norm_xy_dict=norm_xy,
            top_left=draw_box_top_left,
            size=draw_box_size,
            target_aspect=CHOREO_ASPECT,
            kp_color=(255, 200, 0),
            bone_color=(180, 180, 180),
            radius=2,
            thickness=1
        )


        # “NEXT” tag on the very first card (top-right)
        if i == 0:
            tag_w, tag_h = 48, 18
            tx = x + item_w - tag_w - 8
            ty = iy + 8
            cv2.rectangle(frame_bgr, (tx, ty), (tx+tag_w, ty+tag_h), (60, 180, 120), -1, cv2.LINE_AA)
            cv2.putText(frame_bgr, "NEXT", (tx+6, ty+13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_pip_live_skeleton(base_canvas_bgr, live_frame_bgr):
    """
    Bottom-left PiP of the *already-annotated* live camera view.
    """
    H, W = base_canvas_bgr.shape[:2]
    pip_w = int(W * 0.28)        # ~28% width
    pip_h = int(pip_w * 9/16)    # keep 16:9
    x = 16
    y = H - pip_h - 16

    # Background panel
    overlay = base_canvas_bgr.copy()
    cv2.rectangle(overlay, (x-8, y-8), (x+pip_w+8, y+pip_h+8), (25,25,25), -1)
    base_canvas_bgr[:] = cv2.addWeighted(overlay, 0.35, base_canvas_bgr, 0.65, 0)

    # Resize and paste
    pip = cv2.resize(live_frame_bgr, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
    base_canvas_bgr[y:y+pip_h, x:x+pip_w] = pip

    draw_text_shadow(base_canvas_bgr, "LIVE", (x, y-10),
                     scale=0.6, color=(255, 200, 0), thickness=2)
    
# Assume your choreography poses were authored against a 16:9 frame.
# If your JSON poses were normalized against a square, change to 1.0.
CHOREO_ASPECT = 16.0 / 9.0  # width / height

def _fit_rect(container_w, container_h, target_aspect):
    """
    Return (x_off, y_off, draw_w, draw_h) that fits a target aspect inside container,
    centered, with letter/pillar boxing (no stretch).
    """
    cont_aspect = container_w / float(container_h)
    if cont_aspect > target_aspect:
        # container is wider -> pillarbox
        draw_h = container_h
        draw_w = int(round(draw_h * target_aspect))
        x_off = (container_w - draw_w) // 2
        y_off = 0
    else:
        # container is taller -> letterbox
        draw_w = container_w
        draw_h = int(round(draw_w / target_aspect))
        x_off = 0
        y_off = (container_h - draw_h) // 2
    return x_off, y_off, draw_w, draw_h

def draw_norm_xy_pose_fit(frame_bgr, norm_xy_dict, top_left, size,
                          target_aspect=CHOREO_ASPECT,
                          kp_color=(255, 200, 0), bone_color=(220, 220, 220),
                          radius=2, thickness=1):
    """
    Aspect-preserving renderer: draws pose into a letterboxed sub-rect
    inside `size`, so the skeleton isn't squished.
    """
    if not norm_xy_dict:
        return
    x0, y0 = top_left
    box_w, box_h = size

    # Compute aspect-preserving draw area INSIDE the provided box
    off_x, off_y, draw_w, draw_h = _fit_rect(box_w, box_h, target_aspect)
    base_x = x0 + off_x
    base_y = y0 + off_y

    # Map norm coords (0..1) into the fitted draw rect
    pts = {}
    for k, (nx, ny) in norm_xy_dict.items():
        try:
            idx = int(k)
        except Exception:
            continue
        px = int(base_x + nx * draw_w)
        py = int(base_y + ny * draw_h)
        pts[idx] = (px, py)

    # Bones
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame_bgr, pts[a], pts[b], bone_color, thickness, cv2.LINE_AA)
    # Keypoints
    for (px, py) in pts.values():
        cv2.circle(frame_bgr, (px, py), radius, kp_color, -1, lineType=cv2.LINE_AA)

def draw_points_hud(img, points, pos=(10, 38)):
    draw_text_shadow(img, f"Score: {int(points)}", pos, scale=1.0, color=(255,255,255), thickness=2)

def draw_tier_banner(img, tier, score, move_name, y_px=120):
    t = (tier or "miss").lower()
    color = TIER_COLORS.get(t, (255,255,255))
    text = f"{t.upper()}  +{int(score)}"
    H, W = img.shape[:2]
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    x = max(12, (W - text_size[0]) // 2)
    draw_text_shadow(img, text, (x, y_px), scale=1.2, color=color, thickness=3)

# --- Pose tweening Helpers ---

def _ease(t, mode="cubic"):
    """t in [0,1] → eased t."""
    t = max(0.0, min(1.0, t))
    if mode == "linear":
        return t
    if mode == "quint":  # smoother ends than cubic
        return 16*t**5 - 40*t**4 + 40*t**3 - 20*t**2 + 5*t
    # default cubic in-out
    if t < 0.5:
        return 4*t*t*t
    return 1 - pow(-2*t + 2, 3) / 2

def _blend_norm_xy(a, b, alpha):
    """
    LERP two norm_xy dicts: {str(idx): [x,y]}, alpha in [0,1].
    Missing keys fall back to whichever is present.
    """
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va is None: 
            out[k] = vb
            continue
        if vb is None:
            out[k] = va
            continue
        ax, ay = va
        bx, by = vb
        out[k] = (ax + (bx - ax) * alpha, ay + (by - ay) * alpha)
    return out

def _get_pose_norm_xy(move):
    try:
        return (move.raw.get("pose", {}) or {}).get("norm_xy", None)
    except Exception:
        return None



# ---------- Main ----------
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
    ghost_cache = build_ghost_cache(moves, ghost_size=(96, 120))

    # Create scoring engine (JSON-based)
    scorer = ScoringEngine(
        moves=moves,
        window_half_ms=WINDOW_HALF_MS,
        thresholds=TIER_THRESHOLDS,
        tie_breaker="closest"
    )

    last_move_feedback = None  # (tier, score, name, show_until_ms)
    total_points = 0

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

                # --- Update GAME clock based on state ---
                if state == "idle":
                    pass  # idle overlay drawn later on canvas
                elif state == "countdown":
                    ms_left = int((countdown_deadline - time.monotonic()) * 1000)
                    if ms_left <= 0:
                        state = "running"
                        running_zero_monotonic = time.monotonic()
                        game_ts_ms = 0
                elif state == "running":
                    game_ts_ms = int((time.monotonic() - running_zero_monotonic) * 1000)
                elif state == "paused":
                    pass  # paused label drawn later

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

                        # Points accumulate
                        tier_lower = (res.tier or "").lower()
                        gained = 0 if tier_lower == "miss" else max(0, int(round(res.best_score)))
                        total_points += gained

                        # Store banner state for ~1 second
                        last_move_feedback = (res.tier, res.best_score, res.name, game_ts_ms + 1000)




                # ---------- Build our new layout ----------
                H, W = frame_bgr.shape[:2]

                # 1) Prepare a live view with landmarks (annotate on a copy)
                live_view = frame_bgr.copy()
                if result and result.pose_landmarks:
                    draw_landmarks(live_view, result.pose_landmarks[0])
                    cv2.putText(live_view, "BlazePose (Full) — VIDEO mode", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)

                # 2) Fresh canvas for the composed scene
                canvas = np.zeros_like(frame_bgr)

                # 3) State overlays
                if state == "idle":
                    draw_idle_overlay(canvas, moves, WINDOW_HALF_MS)
                elif state == "countdown":
                    ms_left = int((countdown_deadline - time.monotonic()) * 1000)
                    if ms_left > 0:
                        draw_countdown_overlay(canvas, ms_left)
                elif state == "paused":
                    draw_text_shadow(canvas, "PAUSED", (10, 75), scale=0.9, color=(200, 200, 80))

                # 4) Last-move feedback (big banner) + score HUD
                if last_move_feedback is not None:
                    tier_txt, tier_score, move_name, show_until = last_move_feedback
                    if state in ("running", "paused") and game_ts_ms <= show_until:
                        txt = f"{tier_txt.upper()}  +{int(tier_score)}"
                        cv2.putText(canvas, txt, (10, 66),   # directly under the score
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TIER_COLORS.get(tier_txt.lower(), (255,255,255)), 2, cv2.LINE_AA)
                    elif state in ("running", "paused") and game_ts_ms > show_until:
                        last_move_feedback = None

                # Always-on score HUD (top-left)
                draw_points_hud(canvas, total_points, pos=(10, 38))


                # 5) Center: large target pose for current beat (or first in idle)
                current_move, upcoming = (None, [])
                if moves:
                    if state in ("running", "paused"):
                        current_move, upcoming = find_current_and_upcoming(moves, game_ts_ms, max_upcoming=4)
                    else:
                        current_move = moves[0]
                        upcoming = moves[1:5]

                next_move = upcoming[0] if upcoming else None
                draw_current_json_pose_center(canvas, current_move, next_move=next_move, game_ts_ms=game_ts_ms)

                # 6) Left: vertical upcoming list (next 4)
                draw_upcoming_right_list(canvas, upcoming, game_ts_ms, ghost_cache, max_items=4)

                # 7) Bottom-left: PiP live camera with skeleton
                draw_pip_live_skeleton(canvas, live_view)

                # 8) Controls overlay (always)
                draw_text_shadow(canvas, INSTRUCTIONS, (650, canvas.shape[0] - 12), scale=0.6, color=(200,200,200), thickness=2)

                # 9) Present
                frame_bgr = canvas
                cv2.imshow("BlazePose Webcam", frame_bgr)

                # --- input handling ---
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
                        total_points = 0
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
                    total_points = 0
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
