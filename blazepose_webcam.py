# blazepose_compare.py
import time
import cv2
import numpy as np
import mediapipe as mp
import math

# Short aliases
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

# Connections for drawing bones
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

MODEL_PATH = "models/pose_landmarker_full.task"  # _lite / _full / _heavy
REF_IMAGE_PATH = "reference_poses/jumping_jacks.jpg"  # <-- set this

# Drawing params
LANDMARK_RADIUS = 3
LINE_THICKNESS = 2
KP_COLOR = (0, 255, 0)
BONE_COLOR = (240, 240, 240)
TEXT_COLOR = (255, 255, 255)
REF_KP_COLOR = (255, 0, 255)

# Landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28

# Weights
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


JOINT_TOLERANCE_DEG = {
    "l_elbow": 16.0, "r_elbow": 16.0,      # elbows: fairly precise
    "l_shoulder": 18.0, "r_shoulder": 18.0,# shoulders: a bit more leeway
    "l_hip": 20.0, "r_hip": 20.0,          # hips: most lenient of upper body
}
SCORE_DEG_SCALE = 120.0  # average post-tolerance error of 120° → ~0 score



def pose_similarity(emb_a, emb_b):
    """
    Tolerant, upper-body-only similarity:
    - Per-joint tolerance (no error within ±tol)
    - Weighted average of remaining error
    - Softer mapping to 0..100 via SCORE_DEG_SCALE
    """
    total_w = 0.0
    weighted_err = 0.0
    per_joint_err = {}

    for k, a in emb_a.items():
        b = emb_b.get(k)
        if a is None or b is None:
            continue

        # shortest angular difference
        diff = abs(a - b)
        if diff > 180:
            diff = 360 - diff

        # apply per-joint tolerance (free zone)
        tol = JOINT_TOLERANCE_DEG.get(k, 0.0)
        diff = max(0.0, diff - tol)

        # weight
        w = _joint_weight(k)
        weighted_err += w * diff
        total_w += w
        per_joint_err[k] = (diff, w)

    if total_w == 0:
        return 0.0, {"reason": "no comparable joints"}

    avg_err = weighted_err / total_w

    # Softer score curve: 0 error → 100. If avg_err == SCORE_DEG_SCALE → ~0
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

def draw_norm_landmarks_scaled(frame_bgr, landmarks_norm_list, scale=0.55, offset=(20, 80),
                               kp_color=(255, 0, 255), bone_color=(200, 120, 200),
                               radius=2, thickness=1):
    """
    Draw normalized landmarks scaled down into a sub-rectangle on the frame.
    scale: fraction of frame height for ghost height (0..1). offset: (x0, y0) in px.
    """
    if not landmarks_norm_list:
        return
    H, W = frame_bgr.shape[:2]
    ghost_h = int(H * scale)
    ghost_w = int(ghost_h * 9 / 16)  # assume 16:9-ish; adjust if you like
    x0, y0 = offset
    pts = []
    for lm in landmarks_norm_list:
        x = int(x0 + lm.x * ghost_w)
        y = int(y0 + lm.y * ghost_h)
        pts.append((x, y))
    for a, b in POSE_CONNECTIONS:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], bone_color, thickness, cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(frame_bgr, (x, y), radius, kp_color, -1, lineType=cv2.LINE_AA)

def load_reference_embedding(image_path, model_path=MODEL_PATH):
    ref_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    ref_bgr = cv2.imread(image_path)
    if ref_bgr is None:
        raise FileNotFoundError(f"Could not read reference image: {image_path}")

    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=ref_rgb)

    with PoseLandmarker.create_from_options(ref_options) as ref_landmarker:
        ref_result = ref_landmarker.detect(mp_img)

    if not ref_result or not ref_result.pose_landmarks:
        raise RuntimeError("No pose detected in reference image.")

    ref_norm = ref_result.pose_landmarks[0]  # normalized landmarks (0..1)
    h, w = ref_bgr.shape[:2]
    ref_pts = [(int(lm.x * w), int(lm.y * h)) for lm in ref_norm]
    ref_emb = pose_embedding(ref_pts)

    preview = ref_bgr.copy()
    draw_landmarks(preview, ref_norm, kp_color=REF_KP_COLOR, bone_color=(200, 120, 200))
    return ref_emb, ref_norm, preview

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open webcam. Check OS camera permissions.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ghost_on = True     # 'g' to toggle ghost overlay
    split_on = False    # 's' to toggle side-by-side view

    # Load reference once
    ref_emb = None
    ref_norm = None
    ref_preview = None
    try:
        ref_emb, ref_norm, ref_preview = load_reference_embedding(REF_IMAGE_PATH, MODEL_PATH)
        cv2.imshow("Reference Pose (once)", ref_preview)
        cv2.waitKey(1)
    except Exception as e:
        print(f"[!] Reference load failed: {e}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        ts_ms = 0
        score = None  # will hold latest score
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[!] Frame grab failed.")
                break

            # Optional selfie-mirror for dancing:
            # frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            ts_ms += 33
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result and result.pose_landmarks:
                # Draw your live skeleton
                draw_landmarks(frame_bgr, result.pose_landmarks[0])
                cv2.putText(frame_bgr, "BlazePose (Full) — VIDEO mode", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)

                # Compute similarity (but don't draw text yet)
                if ref_emb is not None:
                    h, w = frame_bgr.shape[:2]
                    live_pts = landmarks_to_pixels(result.pose_landmarks[0], w, h)
                    live_emb = pose_embedding(live_pts)
                    score, _info = pose_similarity(live_emb, ref_emb)

            # --- Ghost overlay (smaller + more opaque) ---
            if ref_emb is not None and ref_norm is not None and ghost_on:
                overlay = frame_bgr.copy()
                # Tweak these to taste:
                GHOST_SCALE  = 0.45    # 0.3..0.8 (smaller/bigger ghost)
                GHOST_OFFSET = (20, 80)  # (x,y) from top-left
                draw_norm_landmarks_scaled(
                    overlay, ref_norm,
                    scale=GHOST_SCALE, offset=GHOST_OFFSET,
                    kp_color=(255, 0, 255), bone_color=(200, 120, 200),
                    radius=2, thickness=1
                )
                alpha = 0.75  # higher = more opaque ghost
                frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

            # Draw score last so it sits on top of overlays
            if score is not None:
                cv2.putText(frame_bgr, f"Similarity: {score:5.1f}",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

            # --- Split view window (optional) ---
            if split_on and ref_preview is not None:
                h_live, w_live = frame_bgr.shape[:2]
                ref_scaled = cv2.resize(ref_preview, (int(w_live * 0.6), h_live))
                side_by_side = np.hstack([ref_scaled, frame_bgr])
                cv2.imshow("Ref ⟷ Live", side_by_side)
            else:
                cv2.destroyWindow("Ref ⟷ Live")

            cv2.imshow("BlazePose Webcam", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                ghost_on = not ghost_on
            elif key == ord('s'):
                split_on = not split_on

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
