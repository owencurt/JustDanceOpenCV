import math
from typing import Dict, Optional, Tuple, Any

import cv2
import mediapipe as mp

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26

REGION_WEIGHTS = {"upper": 1.0, "core": 0.6, "lower": 0}
PER_JOINT_WEIGHTS = {
    "l_elbow": 1.0,
    "r_elbow": 1.0,
    "l_shoulder": 0.9,
    "r_shoulder": 0.9,
    "l_hip": 0.6,
    "r_hip": 0.6,
    "l_knee": 0,
    "r_knee": 0,
}

JOINT_TOLERANCE_DEG = {
    "l_elbow": 16.0,
    "r_elbow": 16.0,
    "l_shoulder": 18.0,
    "r_shoulder": 18.0,
    "l_hip": 20.0,
    "r_hip": 20.0,
}
SCORE_DEG_SCALE = 100.0


def _angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    v1_len = math.hypot(*v1)
    v2_len = math.hypot(*v2)
    if v1_len == 0 or v2_len == 0:
        return None
    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (v1_len * v2_len)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _joint_weight(name: str):
    region = (
        "upper"
        if "elbow" in name or "shoulder" in name
        else "lower" if "knee" in name or "ankle" in name else "core"
    )
    base = REGION_WEIGHTS.get(region, 1.0)
    return PER_JOINT_WEIGHTS.get(name, base)


def landmarks_to_pixels(landmarks_norm_list, w: int, h: int):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm_list]


def pose_embedding(pts):
    return {
        "l_elbow": _angle(pts[L_SHOULDER], pts[L_ELBOW], pts[L_WRIST]),
        "r_elbow": _angle(pts[R_SHOULDER], pts[R_ELBOW], pts[R_WRIST]),
        "l_shoulder": _angle(pts[L_HIP], pts[L_SHOULDER], pts[L_ELBOW]),
        "r_shoulder": _angle(pts[R_HIP], pts[R_SHOULDER], pts[R_ELBOW]),
        "l_hip": _angle(pts[L_SHOULDER], pts[L_HIP], pts[L_KNEE]),
        "r_hip": _angle(pts[R_SHOULDER], pts[R_HIP], pts[R_KNEE]),
    }


def pose_similarity(emb_a, emb_b) -> Tuple[float, Dict[str, Any]]:
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


def draw_landmarks(frame_bgr, landmarks_norm_list, kp_color=(0, 255, 0), bone_color=(240, 240, 240), radius=3, thickness=2):
    if not landmarks_norm_list:
        return
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm_list]
    for a, b in POSE_CONNECTIONS:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], bone_color, thickness, cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(frame_bgr, (x, y), radius, kp_color, -1, lineType=cv2.LINE_AA)
