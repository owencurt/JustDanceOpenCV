# blazepose_webcam.py
import time
import cv2
import numpy as np
import mediapipe as mp

# Short aliases
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

# Use MediaPipe's standard pose connections for drawing bones
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

MODEL_PATH = "models/pose_landmarker_full.task"  # change to _lite or _heavy if desired

# Drawing params
LANDMARK_RADIUS = 3
LINE_THICKNESS = 2
KP_COLOR = (0, 255, 0)
BONE_COLOR = (240, 240, 240)
TEXT_COLOR = (255, 255, 255)


def draw_landmarks(frame_bgr, landmarks_norm_list):
    """Draw 33 keypoints and bones from a list of NormalizedLandmark (x,y in [0..1])."""
    if not landmarks_norm_list:
        return

    h, w = frame_bgr.shape[:2]
    # Convert normalized coords to pixel coords
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm_list]

    # Bones
    for a, b in POSE_CONNECTIONS:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], BONE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

    # Keypoints
    for (x, y) in pts:
        cv2.circle(frame_bgr, (x, y), LANDMARK_RADIUS, KP_COLOR, -1, lineType=cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open webcam. Check macOS Camera permissions for Terminal/VS Code.")
        return

    # Try 1280x720 for a good quality/speed balance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,   # synchronous per-frame inference
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        ts_ms = 0  # monotonically increasing timestamp for VIDEO mode

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("[!] Frame grab failed.")
                break

            # Convert BGR (OpenCV) -> SRGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            ts_ms += 33  # ~30 FPS pacing; only needs to be strictly increasing
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result and result.pose_landmarks:
                # Single person: index 0
                draw_landmarks(frame_bgr, result.pose_landmarks[0])
                cv2.putText(frame_bgr, "BlazePose (Full) — VIDEO mode", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)

            cv2.imshow("BlazePose Webcam", frame_bgr)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
