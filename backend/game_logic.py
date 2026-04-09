import math
from bisect import bisect_right
from typing import Dict, Optional

# Landmark indices (MediaPipe BlazePose)
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26

REGION_WEIGHTS = {"upper": 1.0, "core": 0.6, "lower": 0.0}
PER_JOINT_WEIGHTS = {
    "l_elbow": 1.0,
    "r_elbow": 1.0,
    "l_shoulder": 0.9,
    "r_shoulder": 0.9,
    "l_hip": 0.6,
    "r_hip": 0.6,
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


def _angle(a, b, c) -> Optional[float]:
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


def landmarks_to_pixels(landmarks_norm_list, w, h):
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


def _joint_weight(name):
    region = (
        "upper"
        if "elbow" in name or "shoulder" in name
        else "lower"
        if "knee" in name or "ankle" in name
        else "core"
    )
    base = REGION_WEIGHTS.get(region, 1.0)
    return PER_JOINT_WEIGHTS.get(name, base)


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


def find_current_and_upcoming(moves, game_ts_ms, max_upcoming=4, start_times=None):
    if not moves:
        return None, []

    starts = start_times if start_times is not None else [m.start_ms for m in moves]
    idx = bisect_right(starts, game_ts_ms) - 1
    if idx < 0:
        idx = 0
    current = moves[idx]
    upcoming_start = idx + 1
    return current, moves[upcoming_start : upcoming_start + max_upcoming]


def move_to_payload(move):
    if move is None:
        return None
    norm_xy = ((move.raw or {}).get("pose", {}) or {}).get("norm_xy", {}) or {}
    return {
        "name": move.name,
        "start_ms": move.start_ms,
        "norm_xy": norm_xy,
    }
