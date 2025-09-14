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
REF_IMAGE_PATH = "reference_poses/jumping_jacks.jpg"      # <-- set this

# Drawing params
LANDMARK_RADIUS = 3
LINE_THICKNESS = 2
KP_COLOR = (0, 255, 0)
BONE_COLOR = (240, 240, 240)
TEXT_COLOR = (255, 255, 255)

# Landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26
L_ANKLE,    R_ANKLE    = 27, 28

# Weights
REGION_WEIGHTS = {"upper": 1.0, "core": 0.6, "lower": 0.3}
PER_JOINT_WEIGHTS = {
    "l_elbow":1.0, "r_elbow":1.0,
    "l_shoulder":0.9, "r_shoulder":0.9,
    "l_hip":0.6, "r_hip":0.6,
    "l_knee":0.3, "r_knee":0.3,
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
    emb = {
        "l_elbow":   _angle(pts[L_SHOULDER], pts[L_ELBOW],  pts[L_WRIST]),
        "r_elbow":   _angle(pts[R_SHOULDER], pts[R_ELBOW],  pts[R_WRIST]),
        "l_shoulder":_angle(pts[L_HIP],      pts[L_SHOULDER], pts[L_ELBOW]),
        "r_shoulder":_angle(pts[R_HIP],      pts[R_SHOULDER], pts[R_ELBOW]),
        "l_hip":     _angle(pts[L_SHOULDER], pts[L_HIP],    pts[L_KNEE]),
        "r_hip":     _angle(pts[R_SHOULDER], pts[R_HIP],    pts[R_KNEE]),
        "l_knee":    _angle(pts[L_HIP],      pts[L_KNEE],   pts[L_ANKLE]),
        "r_knee":    _angle(pts[R_HIP],      pts[R_KNEE],   pts[R_ANKLE]),
    }
    return emb

def pose_similarity(emb_a, emb_b):
    total_w = 0.0
    weighted_err = 0.0
    for k in emb_a.keys():
        a = emb_a[k]; b = emb_b.get(k)
        if a is None or b is None:
            continue
        diff = abs(a - b)
        if diff > 180:
            diff = 360 - diff
        w = _joint_weight(k)
        weighted_err += w * diff
        total_w += w
    if total_w == 0:
        return 0.0, {"reason": "no comparable joints"}
    avg_err = weighted_err / total_w
    score = max(0.0, 100.0 * (1.0 - (avg_err / 90.0)))
    return score, {"avg_err_deg": avg_err}

def draw_landmarks(frame_bgr, landmarks_norm_list):
    if not landmarks_norm_list:
        return
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm_list]
    for a, b in POSE_CONNECTIONS:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], BONE_COLOR, LINE_THICKNESS, cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(frame_bgr, (x, y), LANDMARK_RADIUS, KP_COLOR, -1, lineType=cv2.LINE_AA)

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
    h, w = ref_bgr.shape[:2]
    ref_pts = landmarks_to_pixels(ref_result.pose_landmarks[0], w, h)
    ref_emb = pose_embedding(ref_pts)
    preview = ref_bgr.copy()
    draw_landmarks(preview, ref_result.pose_landmarks[0])
    return ref_emb, preview

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open webcam. Check OS camera permissions.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Load reference once
    try:
        ref_emb, ref_preview = load_reference_embedding(REF_IMAGE_PATH, MODEL_PATH)
        cv2.imshow("Reference Pose (once)", ref_preview)
        cv2.waitKey(1)
    except Exception as e:
        print(f"[!] Reference load failed: {e}")
        ref_emb = None

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
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[!] Frame grab failed.")
                break

            # Optional selfie-mirror (often nicer for dancing):
            # frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            ts_ms += 33
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result and result.pose_landmarks:
                draw_landmarks(frame_bgr, result.pose_landmarks[0])
                cv2.putText(frame_bgr, "BlazePose (Full) — VIDEO mode", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)

                if ref_emb is not None:
                    h, w = frame_bgr.shape[:2]
                    live_pts = landmarks_to_pixels(result.pose_landmarks[0], w, h)
                    live_emb = pose_embedding(live_pts)
                    score, info = pose_similarity(live_emb, ref_emb)
                    cv2.putText(frame_bgr, f"Similarity: {score:5.1f}",
                                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow("BlazePose Webcam", frame_bgr)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
