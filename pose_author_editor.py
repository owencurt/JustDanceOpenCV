import sys, os, json, time, math
os.environ["GLOG_minloglevel"] = "2"      # 0=INFO,1=WARNING,2=ERROR,3=FATAL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence TensorFlow/TFLite C++ INFO/WARN

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

import mediapipe as mp

from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput


# ------------ Config ------------
MODEL_PATH = "models/pose_landmarker_full.task"  # _lite/_full/_heavy
DEFAULT_BPM = 120.0
DEFAULT_OFFSET_MS = 0
SAVE_DIR = "charts"
os.makedirs(SAVE_DIR, exist_ok=True)

# BlazePose helpers
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# Indices (MediaPipe BlazePose)
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24
L_KNEE,     R_KNEE     = 25, 26  # used for hip angle calc only (we still store hips-up)
UPPER_INDICES = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE]

# Global default scoring knobs (hips-up)
GLOBAL_TOL = {"l_elbow":16.0, "r_elbow":16.0, "l_shoulder":18.0, "r_shoulder":18.0, "l_hip":20.0, "r_hip":20.0}
GLOBAL_WEIGHTS = {"upper":1.0, "core":0.6, "lower":0.0}
GLOBAL_SCALE_DEG = 120.0

# ------------ Data Model ------------
@dataclass
class PoseData:
    """Pose stored as normalized XY (0..1) for editor/overlay, plus derived angles for runtime."""
    norm_xy: Dict[str, List[float]]  # key = str(index), value = [x,y]
    angles: Dict[str, float]         # elbows/shoulders/hips

@dataclass
class Move:
    # Explicit time cues: hold this pose from start_ms until the next move's start_ms
    name: str
    start_ms: int
    mirror: bool
    weights: Dict[str, float]           # per move (defaults populated from global)
    tolerance_deg: Dict[str, float]
    score_scale_deg: float
    pose: PoseData

@dataclass
class Chart:
    title: str
    video_path: str
    bpm: float
    offset_ms: int
    moves: List[Move]

# ------------ Geometry / Angles ------------
def _angle(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = (ax - bx, ay - by); v2 = (cx - bx, cy - by)
    l1 = math.hypot(*v1); l2 = math.hypot(*v2)
    if l1 == 0 or l2 == 0: return None
    dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (l1 * l2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

def compute_angles_from_xy(px: Dict[int, Tuple[int,int]]):
    """Upper-body angles (hips-up)."""
    ang = {
        "l_elbow":    _angle(px[L_SHOULDER], px[L_ELBOW],  px[L_WRIST]),
        "r_elbow":    _angle(px[R_SHOULDER], px[R_ELBOW],  px[R_WRIST]),
        "l_shoulder": _angle(px[L_HIP],      px[L_SHOULDER], px[L_ELBOW]),
        "r_shoulder": _angle(px[R_HIP],      px[R_SHOULDER], px[R_ELBOW]),
        "l_hip":      _angle(px[L_SHOULDER], px[L_HIP],    px[L_KNEE]),
        "r_hip":      _angle(px[R_SHOULDER], px[R_HIP],    px[R_KNEE]),
    }
    return ang

# ------------ MediaPipe (single-image) ------------
def detect_pose_on_frame_bgr(frame_bgr):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with PoseLandmarker.create_from_options(options) as lm:
        res = lm.detect(mp_img)
    if not res or not res.pose_landmarks:
        return None
    return res.pose_landmarks[0]

# ------------ PyQt Widgets ------------
class VideoCanvas(QtWidgets.QWidget):
    frameChanged = QtCore.pyqtSignal()
    requestStatus = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.playing = False
        self.frame_bgr = None
        self.frame_size = (1280, 720)
        self.current_ms = 0
        self.video_fps = 30
        self.video_path = ""

        # Pose edit state
        self.norm_xy: Dict[int, Tuple[float,float]] = {}  # index -> (nx, ny) in 0..1
        self.drag_index = None
        self.drag_radius = 14  # px
        self.show_grid = True

        # BPM/offset for grid
        self.bpm = DEFAULT_BPM
        self.offset_ms = DEFAULT_OFFSET_MS

    def load_video(self, path: str):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.requestStatus.emit("Failed to open video.")
            return
        self.video_path = path
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_ms = 0
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        ok, frame = self.cap.read()
        if ok:
            self.frame_bgr = frame
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.update()
            self.requestStatus.emit(f"Loaded {os.path.basename(path)} ({self.frame_size[0]}x{self.frame_size[1]} @ {self.video_fps:.1f} fps)")
        else:
            self.requestStatus.emit("Could not read first frame.")

    def toggle_play(self):
        if not self.cap: return
        self.playing = not self.playing
        if self.playing:
            self.timer.start(int(1000 / max(1, self.video_fps)))
        else:
            self.timer.stop()

    def set_pose_from_norm(self, norm_xy_str: Dict[str, List[float]]):
        """Accepts pose.norm_xy (strings -> [x,y]) and applies to the overlay."""
        self.norm_xy = {int(k): (float(v[0]), float(v[1])) for k, v in norm_xy_str.items()}
        self.update()

    def _tick(self):
        if not self.cap: return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            self.timer.stop()
            return
        self.frame_bgr = frame
        self.current_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        self.update()
        self.frameChanged.emit()

    def seek_ms(self, ms: int):
        if not self.cap: return
        ms = max(0, ms)
        self.cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        ok, frame = self.cap.read()
        if ok:
            self.frame_bgr = frame
            self.current_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.update()
            self.frameChanged.emit()

    def sizeHint(self):
        return QtCore.QSize(720, 405)

    def set_grid_params(self, bpm: float, offset_ms: int):
        self.bpm = bpm
        self.offset_ms = offset_ms
        self.update()

    def init_pose_blank(self):
        """Start with a simple T-ish pose centered."""
        if self.frame_bgr is None: return
        H, W = self.frame_bgr.shape[:2]
        cx, cy = 0.5, 0.45
        # Rough T pose
        init = {
            L_SHOULDER:(cx-0.06, cy), R_SHOULDER:(cx+0.06, cy),
            L_ELBOW:(cx-0.12, cy),   R_ELBOW:(cx+0.12, cy),
            L_WRIST:(cx-0.18, cy),   R_WRIST:(cx+0.18, cy),
            L_HIP:(cx-0.04, cy+0.18), R_HIP:(cx+0.04, cy+0.18),
            L_KNEE:(cx-0.04, cy+0.34), R_KNEE:(cx+0.04, cy+0.34),
        }
        self.norm_xy = {k:(max(0,min(1,x)), max(0,min(1,y))) for k,(x,y) in init.items()}
        self.update()

    def init_pose_from_frame(self):
        if self.frame_bgr is None: return
        res = detect_pose_on_frame_bgr(self.frame_bgr)
        if res is None:
            self.requestStatus.emit("No pose detected on this frame.")
            return
        self.norm_xy = {}
        for idx in UPPER_INDICES:
            lm = res[idx]
            self.norm_xy[idx] = (float(lm.x), float(lm.y))
        self.update()
        self.requestStatus.emit("Initialized pose from frame.")

    def _is_near_beat(self, ms: int, tol_ms: int = 80) -> bool:
        if self.bpm <= 0: return False
        beat_ms = 60000.0 / self.bpm
        phase = (ms - self.offset_ms) % beat_ms
        return (phase <= tol_ms) or (beat_ms - phase <= tol_ms)

    def paintEvent(self, e: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing |
            QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        # Draw frame
        if self.frame_bgr is not None:
            rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QtGui.QImage(rgb.data, w, h, w*3, QtGui.QImage.Format.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            pix = pix.scaled(self.width(), self.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            painter.drawPixmap(0, 0, pix)

        # Beat flash (visual metronome)
        if self.show_grid and self._is_near_beat(self.current_ms):
            pen = QtGui.QPen(QtGui.QColor(255, 255, 0, 180), 8)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(4, 4, self.width()-8, self.height()-8)

        # Current time text
        painter.setPen(QtGui.QPen(QtGui.QColor(255,255,255,220), 1))
        painter.drawText(10, 20, f"{self.current_ms} ms")

        # Pose overlay
        if self.norm_xy:
            # map norm_xy -> widget coords
            Ww, Hw = self.width(), self.height()
            pts = {i:(int(self.norm_xy[i][0]*Ww), int(self.norm_xy[i][1]*Hw)) for i in self.norm_xy}

            # bones
            pen_bone = QtGui.QPen(QtGui.QColor(220,220,220), 2)
            painter.setPen(pen_bone)
            for a,b in POSE_CONNECTIONS:
                if a in pts and b in pts:
                    painter.drawLine(pts[a][0], pts[a][1], pts[b][0], pts[b][1])

            # joints
            brush = QtGui.QBrush(QtGui.QColor(0,255,0))
            painter.setBrush(brush)
            pen_kp = QtGui.QPen(QtGui.QColor(0,255,0))
            painter.setPen(pen_kp)
            for i,(x,y) in pts.items():
                painter.drawEllipse(QtCore.QPoint(x,y), 6, 6)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if not self.norm_xy: return
        x, y = ev.position().x(), ev.position().y()
        Ww, Hw = self.width(), self.height()
        # find nearest joint within radius
        best = None; best_d2 = self.drag_radius**2
        for i,(nx,ny) in self.norm_xy.items():
            px, py = nx*Ww, ny*Hw
            d2 = (px - x)**2 + (py - y)**2
            if d2 <= best_d2:
                best = i; best_d2 = d2
        self.drag_index = best

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self.drag_index is None or not self.norm_xy: return
        Ww, Hw = self.width(), self.height()
        nx = max(0.0, min(1.0, ev.position().x()/Ww))
        ny = max(0.0, min(1.0, ev.position().y()/Hw))
        self.norm_xy[self.drag_index] = (nx, ny)
        self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        self.drag_index = None

    def export_pose(self) -> Optional[PoseData]:
        if self.frame_bgr is None or not self.norm_xy: return None
        H, W = self.frame_bgr.shape[:2]
        # build pixel dict for angle calc (include knees for hip angle)
        need = {L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE}
        px = {}
        for idx in need:
            if idx in self.norm_xy:
                nx, ny = self.norm_xy[idx]
                px[idx] = (int(nx*W), int(ny*H))
            else:
                # if missing knee because we didn't drag it, fake a knee below hip (keeps hip angle sane)
                if idx == L_KNEE and L_HIP in self.norm_xy:
                    nx, ny = self.norm_xy[L_HIP]
                    px[idx] = (int(nx*W), int(min(1.0, ny+0.18)*H))
                elif idx == R_KNEE and R_HIP in self.norm_xy:
                    nx, ny = self.norm_xy[R_HIP]
                    px[idx] = (int(nx*W), int(min(1.0, ny+0.18)*H))
                else:
                    return None
        ang = compute_angles_from_xy(px)
        if any(v is None for v in ang.values()):
            return None
        norm_xy_str = {str(i): [float(xy[0]), float(xy[1])] for i,xy in self.norm_xy.items()}
        return PoseData(norm_xy=norm_xy_str, angles=ang)

class AddMoveDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, default_start_ms=0):
        super().__init__(parent)
        self.setWindowTitle("Add Move")
        self.setModal(True)
        layout = QtWidgets.QFormLayout(self)

        self.name = QtWidgets.QLineEdit("New Move")
        self.start = QtWidgets.QSpinBox(); self.start.setRange(0, 10_000_000); self.start.setValue(default_start_ms)
        self.mirror = QtWidgets.QCheckBox(); self.mirror.setChecked(True)

        # override boxes (optional)
        self.override = QtWidgets.QCheckBox("Override defaults (tolerance/scale)")
        self.scale = QtWidgets.QDoubleSpinBox(); self.scale.setRange(10.0, 360.0); self.scale.setValue(GLOBAL_SCALE_DEG)
        self.scale.setEnabled(False)
        self.override.toggled.connect(lambda v: self.scale.setEnabled(v))

        layout.addRow("Name", self.name)
        layout.addRow("Start (ms)", self.start)
        layout.addRow("Mirror", self.mirror)
        layout.addRow(self.override)
        layout.addRow("Score scale (deg)", self.scale)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get(self):
        return {
            "name": self.name.text().strip(),
            "start_ms": int(self.start.value()),
            "mirror": bool(self.mirror.isChecked()),
            "override": bool(self.override.isChecked()),
            "scale": float(self.scale.value())
        }

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Authoring Editor")

        # Widgets
        self.canvas = VideoCanvas()
        self.setCentralWidget(self.canvas)

        self._move_clipboard: List[Move] = []
        
        # --- Auto-fill (AI) ---
        np.random.seed(1234)

        self.autofill_step = QtWidgets.QSpinBox(); self.autofill_step.setRange(1, 16); self.autofill_step.setValue(1)
        self.autofill_jitter = QtWidgets.QSpinBox(); self.autofill_jitter.setRange(0, 80); self.autofill_jitter.setValue(0)  # ms sampled around each beat
        self.autofill_clear_chk = QtWidgets.QCheckBox("Clear existing moves first"); self.autofill_clear_chk.setChecked(False)

        # Audio player
        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.75)  # 75% volume to start

        # Right panel: controls
        dock = QtWidgets.QDockWidget("Controls", self); dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        panel = QtWidgets.QWidget(); dock.setWidget(panel); self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        form = QtWidgets.QFormLayout(panel)

        self.title_edit = QtWidgets.QLineEdit("My Dance")
        self.bpm_spin = QtWidgets.QDoubleSpinBox(); self.bpm_spin.setRange(1.0, 10000.0); self.bpm_spin.setValue(DEFAULT_BPM)
        self.offset_spin = QtWidgets.QSpinBox(); self.offset_spin.setRange(0, 100000); self.offset_spin.setValue(DEFAULT_OFFSET_MS)
        self.grid_chk = QtWidgets.QCheckBox("Show beat flash"); self.grid_chk.setChecked(True)
        self.grid_chk.toggled.connect(lambda v: setattr(self.canvas, "show_grid", v))

        # Pose init buttons
        self.btn_init_blank = QtWidgets.QPushButton("Blank Pose")
        self.btn_init_frame = QtWidgets.QPushButton("Init From Frame (AI)")
        self.btn_init_blank.clicked.connect(self.canvas.init_pose_blank)
        self.btn_init_frame.clicked.connect(self.canvas.init_pose_from_frame)

        self.btn_autofill = QtWidgets.QPushButton("Auto-Fill Beats (AI)")
        self.btn_autofill.clicked.connect(self._auto_fill_beats)

        

        # Transport
        self.btn_open = QtWidgets.QPushButton("Open Video")
        self.btn_play = QtWidgets.QPushButton("Play/Pause")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)

        self.btn_next_beat = QtWidgets.QPushButton("Next Beat →")
        self.btn_next_beat.clicked.connect(self._jump_to_next_beat)

        self.btn_prev_beat = QtWidgets.QPushButton("← Prev Beat")
        self.btn_prev_beat.clicked.connect(self._jump_to_prev_beat)

        self.btn_next_beat = QtWidgets.QPushButton("Next Beat →")
        self.btn_next_beat.clicked.connect(self._jump_to_next_beat)

        self._last_jump_ms: Optional[int] = None  # remembers last seek/jump target



        # Moves Preview
        self.btn_preview_moves = QtWidgets.QPushButton("▶ Play Moves Preview")
        self.btn_preview_moves.clicked.connect(self._toggle_moves_preview)

        # Moves preview state
        self._moves_previewing = False
        self._moves_timer = QtCore.QTimer(self)
        self._moves_timer.timeout.connect(self._on_moves_tick)
        self._moves_schedule = []  # list of (start_ms, end_ms, move)
        self._preview_t0_ms = 0    # wallclock ms when preview started
        self._sequence_start_ms = 0
        self._saved_overlay = None  # to restore user's editing pose afterwards

        # scrubbing state
        self._user_scrubbing = False
        self.slider.sliderPressed.connect(self._slider_pressed)
        self.slider.sliderMoved.connect(self._slider_moved)
        self.slider.sliderReleased.connect(self._slider_released)
        self.slider.valueChanged.connect(self._slider_changed)  # for clicks on the bar
        self.slider.setRange(0, 1000)
        self.btn_open.clicked.connect(self._open_video)
        self.btn_play.clicked.connect(self._toggle_play)
        self.canvas.frameChanged.connect(self._sync_slider)

        # Moves table
        self.moves: List[Move] = []
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.table.setHorizontalHeaderLabels(["Name","Start(ms)","Mirror"])
        self.table.horizontalHeader().setStretchLastSection(True)

        # Allow inline editing
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked |
            QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked |
            QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.itemChanged.connect(self._on_table_item_changed)

        # Guard to avoid recursion while refreshing
        self._refreshing_table = False


        self.btn_add_move = QtWidgets.QPushButton("Add Move")
        self.btn_add_move.clicked.connect(self._add_move)

        self.btn_copy = QtWidgets.QPushButton("Copy Selected")
        self.btn_paste = QtWidgets.QPushButton("Paste After Selection")
        self.btn_copy.clicked.connect(self._copy_moves)
        self.btn_paste.clicked.connect(self._paste_moves)
        
        self.btn_delete = QtWidgets.QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self._delete_moves)
        form.addRow(self.btn_delete)


        # Layout order
        form.addRow("Title", self.title_edit)
        form.addRow("BPM", self.bpm_spin)
        form.addRow("Offset (ms)", self.offset_spin)
        form.addRow(self.grid_chk)
        form.addRow(self.btn_open, self.btn_play)
        form.addRow(self.slider)
        form.addRow(self.btn_prev_beat, self.btn_next_beat)
        form.addRow(self.btn_preview_moves)
        form.addRow(self.btn_init_blank, self.btn_init_frame)
        form.addRow(QtWidgets.QLabel("Moves:"))
        form.addRow(self.table)
        form.addRow(self.btn_add_move)
        form.addRow(self.btn_copy, self.btn_paste)

        self.btn_export = QtWidgets.QPushButton("Export JSON")
        self.btn_export.clicked.connect(self._export_chart)
        form.addRow(self.btn_export)

        form.addRow(QtWidgets.QLabel("Auto-Fill every Nth beat"), self.autofill_step)
        form.addRow(QtWidgets.QLabel("Sample jitter (±ms)"), self.autofill_jitter)
        form.addRow(self.autofill_clear_chk)
        form.addRow(self.btn_autofill)

        # Status bar
        self.status = self.statusBar()
        self.canvas.requestStatus.connect(self.status.showMessage)

        # Wiring
        self.bpm_spin.valueChanged.connect(self._update_grid)
        self.offset_spin.valueChanged.connect(self._update_grid)

        self.resize(1100, 720)

        # Optional small drift corrector to resync audio to video every second
        self._drift_timer = QtCore.QTimer(self)
        self._drift_timer.timeout.connect(self._correct_av_drift)
        self._drift_timer.start(1000)

    # ----- Transport / Keys -----
    def _toggle_play(self):
        if not self.canvas.cap:
            return
        self.canvas.toggle_play()
        if self.canvas.playing:
            self.player.play()
        else:
            self.player.pause()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == QtCore.Qt.Key.Key_Space:
            self._toggle_play()
        else:
            super().keyPressEvent(e)

    # ----- Preview schedule (explicit times) -----
    def _build_moves_schedule(self) -> List[Tuple[int,int,Move]]:
        """Expand to [(start,end,move)] using next move's start as end; last ends at video end."""
        if not self.moves: return []
        ms = sorted(self.moves, key=lambda m: m.start_ms)
        blocks = []
        video_end = self._video_duration_ms()
        for i, mv in enumerate(ms):
            s = mv.start_ms
            e = (ms[i+1].start_ms if i+1 < len(ms) else video_end)
            e = max(e, s + 1)
            blocks.append((s, e, mv))
        return blocks
    
    def _current_clock_ms(self) -> int:
        """Best-effort current media time in ms (prefers audio, falls back to video, then last jump)."""
        vid = getattr(self.canvas, "current_ms", 0) or 0
        aud = self.player.position() if self.player is not None else -1
        # Prefer audio clock if valid; it usually tracks setPosition precisely.
        if aud >= 0:
            now = aud
        else:
            now = vid
        # If we recently jumped and clocks haven't caught up, trust our last jump if it's later.
        if self._last_jump_ms is not None and self._last_jump_ms > now - 2:
            now = self._last_jump_ms
        return int(max(0, now))


    def _all_beat_times_ms(self) -> list[int]:
        """Return all beat times (ms) in [first_beat, video_end)."""
        bpm = float(self.bpm_spin.value())
        off = int(self.offset_spin.value())
        dur = self._video_duration_ms()
        if bpm <= 0 or dur <= 0:
            return []
        beat_ms = 60000.0 / bpm
        # first k such that off + k*beat_ms >= 0
        k0 = math.ceil((0 - off) / beat_ms) if off < 0 else 0
        times = []
        t = int(off + k0 * beat_ms)
        while t < max(0, dur - 1):
            times.append(int(t))
            k0 += 1
            t = int(off + k0 * beat_ms)
        return times

    def _auto_fill_beats(self):
        if not self.canvas.cap:
            self.status.showMessage("Open a video first.")
            return
        bpm = float(self.bpm_spin.value())
        if bpm <= 0:
            self.status.showMessage("Set a valid BPM first.")
            return

        # Prep
        app = QtWidgets.QApplication.instance()
        was_playing = self.canvas.playing
        if was_playing:
            self._toggle_play()  # pause both audio & video

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.btn_autofill.setEnabled(False)

        try:
            # Optionally clear existing moves
            if self.autofill_clear_chk.isChecked():
                self.moves.clear()

            beat_times = self._all_beat_times_ms()
            step = max(1, int(self.autofill_step.value()))
            jitter = int(self.autofill_jitter.value())
            added = 0

            # Small local helper: try to detect at t and a couple tiny fallbacks
            def try_detect_at(t_ms: int) -> Optional[PoseData]:
                probe_times = [t_ms]
                # slight fallbacks if detection fails at exact beat
                for d in (-30, +30, -60, +60):
                    probe_times.append(max(0, t_ms + d))
                for probe in probe_times:
                    self.canvas.seek_ms(probe)
                    if self.player is not None:
                        self.player.setPosition(probe)
                    # let the frame settle
                    if app: app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 5)
                    self.canvas.init_pose_from_frame()
                    pose = self.canvas.export_pose()
                    if pose is not None:
                        return pose
                return None

            for i, t in enumerate(beat_times):
                if i % step != 0:
                    continue
                # optional jitter around the beat (uniform in ±jitter)
                if jitter > 0:
                    delta = np.random.randint(-jitter, jitter + 1)
                    t = max(0, t + int(delta))

                pose = try_detect_at(t)
                if pose is None:
                    # Skip silently; you can always hand-place later
                    continue

                mv = Move(
                    name=f"Beat {i}",
                    start_ms=int(t),
                    mirror=True,
                    weights=dict(GLOBAL_WEIGHTS),
                    tolerance_deg=dict(GLOBAL_TOL),
                    score_scale_deg=GLOBAL_SCALE_DEG,
                    pose=pose
                )
                self.moves.append(mv)
                added += 1

                # keep UI alive and show progress
                if app: app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)
                if added % 8 == 0:
                    self.status.showMessage(f"Auto-fill: added {added} moves...")

            # Sort & refresh table once at the end
            self._refresh_moves_table()
            self.status.showMessage(f"Auto-fill complete: added {added} move(s).")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.btn_autofill.setEnabled(True)
            if was_playing:
                self._toggle_play()


        # ----- Beat jumping -----
    def _next_beat_ms(self, current_ms: int) -> int:
        """Compute the next beat strictly after current_ms."""
        bpm = float(self.bpm_spin.value())
        off = int(self.offset_spin.value())
        dur = self._video_duration_ms()
        if not self.canvas.cap or dur <= 0 or bpm <= 0:
            return current_ms
        beat_ms = 60000.0 / bpm
        # index of the beat just before/at current_ms
        k = math.floor((current_ms - off) / beat_ms)
        # strictly next beat
        next_ms = int(off + (k + 1) * beat_ms)
        # clamp to just before the end (so a seek can still render a frame)
        return max(0, min(next_ms, max(0, dur - 1)))

    def _jump_to_next_beat(self):
        if not self.canvas.cap:
            self.status.showMessage("Open a video first.")
            return
        # +1ms so if we're exactly on-beat, we still move forward
        now = self._current_clock_ms() + 1
        target_ms = self._next_beat_ms(now)
        self.canvas.seek_ms(target_ms)
        if self.player is not None:
            self.player.setPosition(target_ms)
        self._last_jump_ms = target_ms
        self.status.showMessage(f"Jumped to next beat: {target_ms} ms")

        # ----- Beat jumping -----
    def _prev_beat_ms(self, current_ms: int) -> int:
        """Compute the previous beat strictly before current_ms."""
        bpm = float(self.bpm_spin.value())
        off = int(self.offset_spin.value())
        dur = self._video_duration_ms()
        if not self.canvas.cap or dur <= 0 or bpm <= 0:
            return current_ms
        beat_ms = 60000.0 / bpm
        # index of the beat just after current_ms, then step back one
        k = math.ceil((current_ms - off) / beat_ms) - 1
        prev_ms = int(off + k * beat_ms)
        return max(0, min(prev_ms, max(0, dur - 1)))

    def _jump_to_prev_beat(self):
        if not self.canvas.cap:
            self.status.showMessage("Open a video first.")
            return
        # -1ms so if we're exactly on-beat, we still move backward
        now = self._current_clock_ms() - 1
        target_ms = self._prev_beat_ms(now)
        self.canvas.seek_ms(target_ms)
        if self.player is not None:
            self.player.setPosition(target_ms)
        self._last_jump_ms = target_ms
        self.status.showMessage(f"Jumped to previous beat: {target_ms} ms")



    def _sequence_bounds(self, blocks) -> Tuple[int,int]:
        if not blocks: return (0,0)
        return (blocks[0][0], blocks[-1][1])

    def _delete_moves(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            self.status.showMessage("Select one or more moves to delete.")
            return

        for r in rows:
            if 0 <= r < len(self.moves):
                self.moves.pop(r)
        self._refresh_moves_table()
        self.status.showMessage(f"Deleted {len(rows)} move(s).")

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == QtCore.Qt.Key.Key_Space:
            self._toggle_play()
        elif e.key() == QtCore.Qt.Key.Key_Delete:
            self._delete_moves()
        else:
            super().keyPressEvent(e)



    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if self._refreshing_table:
            return  # ignore our own writes

        r = item.row()
        c = item.column()
        if r < 0 or r >= len(self.moves):
            return

        mv = self.moves[r]
        text = item.text().strip()

        try:
            if c == 0:
                # name
                mv.name = text or mv.name
            elif c == 1:
                # start_ms
                new_ms = int(text)
                mv.start_ms = max(0, new_ms)
            elif c == 2:
                # mirror (accept yes/no/true/false/1/0)
                t = text.lower()
                mv.mirror = t in ("yes", "true", "1", "y")
            else:
                return
        except Exception as e:
            # Revert cell on parse error
            self.status.showMessage(f"Invalid value: {e}")
            self._refresh_moves_table()
            return

        # Re-sort after edits that affect ordering
        if c in (1,):
            self._refresh_moves_table()


    def _toggle_moves_preview(self):
        if self._moves_previewing:
            self._stop_moves_preview()
        else:
            self._start_moves_preview()

    def _start_moves_preview(self):
        if not self.canvas.cap or not self.moves:
            self.status.showMessage("Open a video and add at least one move.")
            return

        # Build schedule
        self._moves_schedule = self._build_moves_schedule()
        if not self._moves_schedule:
            self.status.showMessage("No move blocks to preview.")
            return

        # Save the current overlay so we can restore when done
        self._saved_overlay = dict(self.canvas.norm_xy) if self.canvas.norm_xy else None

        # Compute full sequence bounds and seek everything to start
        seq_start, _seq_end = self._sequence_bounds(self._moves_schedule)
        self._sequence_start_ms = seq_start
        self.canvas.seek_ms(seq_start)
        if self.player is not None:
            self.player.setPosition(seq_start)

        # Start video+audio if not already playing
        if not self.canvas.playing:
            self.canvas.toggle_play()
        if self.player is not None:
            self.player.play()

        # Start timing
        self._preview_t0_ms = int(time.time() * 1000)
        self._moves_previewing = True
        self.btn_preview_moves.setText("■ Stop Preview")
        self.status.showMessage("Playing moves preview...")
        self._moves_timer.start(16)  # ~60Hz

    def _stop_moves_preview(self):
        self._moves_timer.stop()
        self._moves_previewing = False
        self.btn_preview_moves.setText("▶ Play Moves Preview")
        self.status.showMessage("Preview stopped.")

        # Pause playback (leave where it ended)
        if self.canvas.playing:
            self.canvas.toggle_play()
        if self.player is not None:
            self.player.pause()

        # Restore editor overlay pose
        if self._saved_overlay is not None:
            self.canvas.norm_xy = dict(self._saved_overlay)
            self.canvas.update()
        self._saved_overlay = None

    def _on_moves_tick(self):
        """Called ~60Hz: figure out current time in video, pick active move, and show its pose."""
        if not self._moves_previewing or not self._moves_schedule:
            return

        # Derive current video time (prefer the actual video clock)
        vid_ms = self.canvas.current_ms
        if vid_ms <= 0 and self._sequence_start_ms:
            now = int(time.time() * 1000)
            vid_ms = self._sequence_start_ms + (now - self._preview_t0_ms)

        # Find active block
        active = None
        for (s, e, mv) in self._moves_schedule:
            if s <= vid_ms < e:
                active = (s, e, mv)
                break

        # Update overlay to the active move pose
        if active is not None:
            _s, _e, mv = active
            self.canvas.set_pose_from_norm(mv.pose.norm_xy)
            self.status.showMessage(f"Move: {mv.name}  |  {vid_ms - _s} / {_e - _s} ms")
        else:
            # Past the end?
            _seq_start, seq_end = self._sequence_bounds(self._moves_schedule)
            if vid_ms >= seq_end:
                self._stop_moves_preview()

    # ----- Copy / Paste (relative timing preserved) -----
    def _copy_moves(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            self.status.showMessage("Select one or more moves to copy.")
            return
        self._move_clipboard = [self.moves[r] for r in rows]
        # sort clipboard by time for sanity
        self._move_clipboard.sort(key=lambda m: m.start_ms)
        self.status.showMessage(f"Copied {len(self._move_clipboard)} move(s).")

    def _paste_moves(self):
        if not self._move_clipboard:
            self.status.showMessage("Clipboard is empty. Copy moves first.")
            return

        # Selection-aware insertion point
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if rows:
            sel_starts = [self.moves[r].start_ms for r in rows]
            sel_first = min(sel_starts)
            sel_last  = max(sel_starts)
            sel_span  = sel_last - sel_first
            # Start AFTER the selection's span (keeps the same delay you had)
            end_after = sel_last + sel_span if sel_span > 0 else sel_last
        elif self.moves:
            # No selection: infer a reasonable gap from the last two moves
            all_starts = sorted(m.start_ms for m in self.moves)
            last = all_starts[-1]
            prev = all_starts[-2] if len(all_starts) >= 2 else None
            last_gap = (last - prev) if prev is not None else 2000  # default 2s if no info
            end_after = last + max(1, last_gap)
            sel_first = self._move_clipboard[0].start_ms  # used below to compute relative offsets
        else:
            end_after = 0
            sel_first = self._move_clipboard[0].start_ms

        # Preserve relative offsets within the copied selection
        src_sorted = sorted(self._move_clipboard, key=lambda m: m.start_ms)
        base = src_sorted[0].start_ms
        new_moves = []
        for src in src_sorted:
            new_start = end_after + (src.start_ms - base)
            mv = Move(
                name=src.name,
                start_ms=new_start,
                mirror=src.mirror,
                weights=dict(src.weights),
                tolerance_deg=dict(src.tolerance_deg),
                score_scale_deg=src.score_scale_deg,
                pose=src.pose
            )
            new_moves.append(mv)

        self.moves.extend(new_moves)
        self._refresh_moves_table()
        self.status.showMessage(f"Pasted {len(new_moves)} move(s) starting at {end_after} ms.")


    # ----- UI helpers -----
    def _open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mov *.avi)")
        if not path: return
        self.canvas.load_video(path)
        # hook audio to same file
        self.player.setSource(QUrl.fromLocalFile(path))
        # keep position aligned on fresh load
        self.player.setPosition(0)
        self._update_grid()

    def _update_grid(self):
        self.canvas.set_grid_params(self.bpm_spin.value(), int(self.offset_spin.value()))

    def _sync_slider(self):
        if self._user_scrubbing:  # don't override while dragging
            return
        dur_ms = self._video_duration_ms()
        if dur_ms <= 0: return
        val = int(1000 * self.canvas.current_ms / dur_ms)
        self.slider.blockSignals(True)
        self.slider.setValue(max(0, min(1000, val)))
        self.slider.blockSignals(False)

    def _add_move(self):
        # ensure we have a pose
        pose = self.canvas.export_pose()
        if pose is None:
            self.status.showMessage("Create or init a pose first (drag joints or Init From Frame).")
            return
        dlg = AddMoveDialog(self, default_start_ms=self.canvas.current_ms)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        data = dlg.get()
        mv = Move(
            name=data["name"],
            start_ms=data["start_ms"],
            mirror=data["mirror"],
            weights=dict(GLOBAL_WEIGHTS),
            tolerance_deg=dict(GLOBAL_TOL),
            score_scale_deg=(data["scale"] if data["override"] else GLOBAL_SCALE_DEG),
            pose=pose
        )
        self.moves.append(mv)
        self._refresh_moves_table()

    def _refresh_moves_table(self):
        self._refreshing_table = True
        try:
            # Keep moves sorted by start time
            self.moves.sort(key=lambda m: m.start_ms)
            self.table.setRowCount(len(self.moves))
            for r, mv in enumerate(self.moves):
                # Name
                item0 = QtWidgets.QTableWidgetItem(mv.name)
                # Start (ms)
                item1 = QtWidgets.QTableWidgetItem(str(mv.start_ms))
                # Mirror (Yes/No)
                item2 = QtWidgets.QTableWidgetItem("Yes" if mv.mirror else "No")

                # Make them editable
                for it in (item0, item1, item2):
                    it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)

                self.table.setItem(r, 0, item0)
                self.table.setItem(r, 1, item1)
                self.table.setItem(r, 2, item2)
        finally:
            self._refreshing_table = False


    def _export_chart(self):
        title = self.title_edit.text().strip() or "My Dance"
        video_path = self.canvas.video_path
        if not video_path:
            self.status.showMessage("Open a video first.")
            return
        if not self.moves:
            self.status.showMessage("Add at least one move.")
            return
        chart = Chart(
            title=title,
            video_path=video_path,
            bpm=float(self.bpm_spin.value()),
            offset_ms=int(self.offset_spin.value()),
            moves=self.moves
        )
        out_path = os.path.join(SAVE_DIR, f"{title.replace(' ','_').lower()}.json")
        # serialize dataclasses -> dict
        def move_to_dict(m: Move):
            return {
                "name": m.name,
                "start_ms": m.start_ms,
                "mirror": m.mirror,
                "weights": m.weights,
                "tolerance_deg": m.tolerance_deg,
                "score_scale_deg": m.score_scale_deg,
                "pose": {
                    "angles": m.pose.angles,
                    "norm_xy": m.pose.norm_xy
                }
            }
        data = {
            "schema_version": 2,
            "title": chart.title,
            "video_path": chart.video_path,
            "bpm": chart.bpm,
            "offset_ms": chart.offset_ms,
            "moves": [move_to_dict(m) for m in chart.moves]
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        self.status.showMessage(f"Saved {out_path}")

    def _video_duration_ms(self) -> int:
        if not self.canvas.cap: return 0
        frames = int(self.canvas.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.canvas.video_fps or 30
        if fps <= 0: return 0
        return int(frames / fps * 1000)

    # ----- Slider scrubbing -----
    def _slider_pressed(self):
        self._user_scrubbing = True

    def _slider_moved(self, val: int):
        # preview seek while dragging
        dur = self._video_duration_ms()
        if dur <= 0: return
        target_ms = int(dur * (val / 1000.0))
        self.canvas.seek_ms(target_ms)
        if self.player is not None:
            self.player.setPosition(target_ms)
        self._last_jump_ms = target_ms


    def _slider_released(self):
        self._user_scrubbing = False

    def _slider_changed(self, val: int):
        # Handle direct clicks on slider track (when not dragging)
        if self._user_scrubbing: return
        dur = self._video_duration_ms()
        if dur <= 0: return
        target_ms = int(dur * (val / 1000.0))
        self.canvas.seek_ms(target_ms)
        if self.player is not None:
            self.player.setPosition(target_ms)

    # ----- Drift corrector (optional) -----
    def _correct_av_drift(self):
        if not self.canvas.cap: return
        vid_ms = self.canvas.current_ms
        aud_ms = self.player.position()
        if abs(vid_ms - aud_ms) > 80:  # >80 ms drift -> correct
            self.player.setPosition(vid_ms)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
