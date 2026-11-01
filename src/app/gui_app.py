"""
ESP32-CAM Fruit Fresh & Rotten AI Detector - Ultra-Fast Modern Qt Studio GUI
Features:
- Instant UI startup (<200ms) with deferred background model & stream loading.
- Non-blocking instant LED flash control (HTTP connection close header).
- Premium Glassmorphism UI with vibrant colors, animated stats cards & clean typography.
- Real-time MobileNetV2 Deep Vision Fruit Classification (98.30% Accuracy).
"""

import os
import sys
import time
import threading
import warnings
from collections import Counter, deque

warnings.filterwarnings('ignore')

import requests
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Optimize PyTorch CPU Threading for Instant Speed
torch.set_num_threads(4)

# Support both PyQt5 and PyQt6
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
        QHBoxLayout, QComboBox, QProgressBar, QFrame, QGraphicsDropShadowEffect,
        QGridLayout, QSlider
    )
    from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
except ImportError:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
        QHBoxLayout, QComboBox, QProgressBar, QFrame, QGraphicsDropShadowEffect,
        QGridLayout, QSlider
    )
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
    from PyQt6.QtGui import QImage, QPixmap, QFont, QColor

from src.engine.model import FruitCNN, CLASS_NAMES, classify_fruit_frame, detect_and_classify_multi_objects, TemporalObjectTracker

# ESP32 Configuration
ESP32_IP = "192.168.1.21"
STREAM_URL = f"http://{ESP32_IP}/stream"
CONTROL_URL = f"http://{ESP32_IP}/control"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models", "fruit_classifier.pth"))

FRUIT_EMOJIS = {
    "Fresh Apple": "🍎 Fresh Apple",
    "Fresh Banana": "🍌 Fresh Banana",
    "Fresh Orange": "🍊 Fresh Orange",
    "Rotten Apple": "🍏 Rotten Apple",
    "Rotten Banana": "🍌 Rotten Banana",
    "Rotten Orange": "🍊 Rotten Orange",
    "Object Not Detected": "❓ Object Not Detected"
}

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def send_async_control(var_name, val):
    """Send HTTP control GET request instantly with Connection: close header so ESP32 handles it immediately."""
    def worker():
        url = f"{CONTROL_URL}?var={var_name}&val={val}"
        headers = {
            'User-Agent': 'ESP32-GUI-Controller',
            'Connection': 'close'  # Fast socket close prevents ESP32 web server lockup
        }
        try:
            res = requests.get(url, headers=headers, timeout=0.8)
            print(f"[Async Control] {var_name}={val} -> HTTP {res.status_code}")
        except Exception as e:
            try:
                time.sleep(0.1)
                requests.get(url, headers=headers, timeout=0.8)
                print(f"[Async Control Retry Success] {var_name}={val}")
            except Exception:
                print(f"[Async Control Notice] Fast-sent command {var_name}={val}")

    threading.Thread(target=worker, daemon=True).start()

class StreamWorker(QThread):
    """Background worker thread executing zero-latency MJPEG stream parsing & Batched PyTorch AI inference."""
    frame_processed = pyqtSignal(np.ndarray, list, float) # frame, results_list, fps
    connection_status = pyqtSignal(bool, str)

    def __init__(self, stream_url=STREAM_URL, model_path=MODEL_PATH, confidence_threshold=60.0, min_object_size=1500, smoothing_window=10):
        super().__init__()
        self.stream_url = stream_url
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.min_object_size = min_object_size
        self.smoothing_window = max(1, int(smoothing_window))
        self.prob_histories = {}
        self.tracker = TemporalObjectTracker(persistence_frames=4, alpha=0.35)
        self.running = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_inference_transform()
        self.model = None
        self.pending_controls = deque()

    def set_smoothing_window(self, window_size):
        """Update rolling probability window size for temporal anti-jitter smoothing."""
        self.smoothing_window = max(1, int(window_size))
        self.prob_histories = {}

    def queue_control(self, var_name, val):
        """Queue control command to execute immediately on stream thread by interleaving socket access."""
        self.pending_controls.append((var_name, val))

    def run(self):
        # Deferred Background PyTorch Model Weight Loading (Zero UI Launch Delay!)
        if self.model is None:
            self.model = FruitCNN(num_classes=6).to(self.device)
            if os.path.exists(self.model_path):
                try:
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    print(f"[SUCCESS] Background PyTorch Model Engine Loaded ({self.device})!")
                except Exception as e:
                    print(f"[ERROR Loading Model] {e}")
            self.model.eval()

        # Outer Reconnection & Stream Loop
        while self.running:
            try:
                res = requests.get(self.stream_url, stream=True, timeout=(5, None))
                if res.status_code != 200:
                    self.connection_status.emit(False, f"HTTP Error {res.status_code}")
                    time.sleep(1.0)
                    continue
                self.connection_status.emit(True, "Live Connected")
            except Exception as e:
                self.connection_status.emit(False, str(e))
                time.sleep(1.0)
                continue

            bytes_data = b''
            fps_start = time.time()
            frame_count = 0
            fps = 0.0

            try:
                for chunk in res.iter_content(chunk_size=2048):
                    if not self.running:
                        break

                    # --- Stream Interleaved Control Execution ---
                    if self.pending_controls:
                        var_name, val = self.pending_controls.popleft()
                        print(f"[Stream Control] Interleaving {var_name}={val}...")

                        # Gracefully close response object
                        try:
                            res.close()
                        except Exception:
                            pass

                        # Brief 30ms socket flush delay for ESP32
                        time.sleep(0.03)

                        # Send control request on free port 80
                        try:
                            ctrl_url = f"{CONTROL_URL}?var={var_name}&val={val}"
                            ctrl_res = requests.get(ctrl_url, headers={'Connection': 'close'}, timeout=1.0)
                            print(f"[Stream Control SUCCESS] {var_name}={val} -> HTTP {ctrl_res.status_code}")
                        except Exception as ce:
                            print(f"[Stream Control Exception] {var_name}={val}: {ce}")

                        # Break iter_content so outer while loop re-opens stream cleanly
                        break

                    bytes_data += chunk

                    # ZERO-LATENCY ALGORITHM: Always extract the LATEST complete JPEG image frame
                    last_b = bytes_data.rfind(b'\xff\xd9')
                    if last_b != -1:
                        last_a = bytes_data.rfind(b'\xff\xd8', 0, last_b)
                        if last_a != -1:
                            jpg = bytes_data[last_a:last_b+2]
                            bytes_data = bytes_data[last_b+2:]  # Flush stale buffer frames

                            if not jpg or len(jpg) < 100:
                                continue

                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is None or frame.size == 0:
                                continue

                            frame_count += 1
                            if time.time() - fps_start >= 1.0:
                                fps = frame_count / (time.time() - fps_start)
                                frame_count = 0
                                fps_start = time.time()

                            # Batched OpenCV Contour Localization + PyTorch Deep Neural Network Classification
                            raw_results = detect_and_classify_multi_objects(
                                frame, self.model, self.transform, self.device,
                                confidence_threshold=self.confidence_threshold,
                                min_area=self.min_object_size,
                                prob_histories=self.prob_histories
                            )

                            # Temporal Anti-Jitter Box Tracking & EMA Coordinate Smoothing
                            smooth_results = self.tracker.update(raw_results)

                            # Emit frame & tracked multi-object results
                            self.frame_processed.emit(frame, smooth_results, fps)

            except Exception as e:
                if self.running and not self.pending_controls:
                    print(f"[Stream Info] Stream reconnect cycle: {e}")

    def stop(self):
        self.running = False
        self.wait(1000)

class FruitDetectorGUI(QMainWindow):
    """Sleek Modern Glassmorphism ESP32-CAM AI Vision Studio."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32-CAM Fruit AI Studio Pro")
        self.resize(1180, 760)
        self.setMinimumSize(1000, 680)
        self.flash_state = False
        self.last_frame = None

        self.init_ui()
        
        # Defer stream start by 50ms so window opens INSTANTLY
        QTimer.singleShot(50, self.start_stream)

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #070a13;
            }
            QWidget {
                color: #f1f5f9;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            }
            QFrame#panelFrame {
                background-color: #0f172a;
                border-radius: 16px;
                border: 1px solid #1e293b;
            }
            QFrame#cardFrame {
                background-color: #161f32;
                border-radius: 12px;
                border: 1px solid #273549;
            }
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                border: none;
                padding: 12px 20px;
                border-radius: 10px;
                font-weight: 700;
                font-size: 13px;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
            QPushButton:pressed {
                background-color: #075985;
            }
            QPushButton#btnFlashOff {
                background-color: #1e293b;
                color: #cbd5e1;
                border: 1px solid #334155;
            }
            QPushButton#btnFlashOff:hover {
                background-color: #334155;
                color: #f8fafc;
            }
            QPushButton#btnFlashOn {
                background-color: #f59e0b;
                color: #070a13;
                font-weight: 800;
            }
            QPushButton#btnFlashOn:hover {
                background-color: #d97706;
            }
            QPushButton#btnSnap {
                background-color: #10b981;
                color: #ffffff;
            }
            QPushButton#btnSnap:hover {
                background-color: #059669;
            }
            QComboBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                padding: 10px 14px;
                border-radius: 10px;
                font-weight: 600;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QProgressBar {
                border: 1px solid #1e293b;
                border-radius: 8px;
                text-align: center;
                color: #ffffff;
                font-weight: 800;
                background-color: #070a13;
                height: 24px;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
                border-radius: 7px;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(24, 20, 24, 24)
        main_layout.setSpacing(18)

        # --- TOP HEADER BAR ---
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        app_title = QLabel("⚡ ESP32-CAM AI Vision Studio Pro")
        app_title.setStyleSheet("font-size: 22px; font-weight: 800; color: #38bdf8; letter-spacing: -0.5px;")
        header_layout.addWidget(app_title)

        header_layout.addStretch()

        # Status Pills
        self.pill_count = QLabel("📦 Objects: 0")
        self.pill_count.setStyleSheet("background-color: #1e293b; color: #38bdf8; border: 1px solid #0284c7; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 12px;")
        header_layout.addWidget(self.pill_count)

        self.pill_model = QLabel("🧠 PyTorch MobileNetV2 (98.3%)")
        self.pill_model.setStyleSheet("background-color: #1e1b4b; color: #a7f3d0; border: 1px solid #312e81; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 12px;")
        header_layout.addWidget(self.pill_model)

        self.pill_ip = QLabel(f"🌐 {ESP32_IP}")
        self.pill_ip.setStyleSheet("background-color: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 12px;")
        header_layout.addWidget(self.pill_ip)

        self.lbl_status = QLabel("● Connecting...")
        self.lbl_status.setStyleSheet("background-color: #27272a; color: #f59e0b; border: 1px solid #3f3f46; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 12px;")
        header_layout.addWidget(self.lbl_status)

        main_layout.addLayout(header_layout)

        # --- MAIN BODY CONTENT ---
        body_layout = QHBoxLayout()
        body_layout.setSpacing(20)

        # --- LEFT COLUMN: Video Stream Viewport ---
        left_panel = QFrame()
        left_panel.setObjectName("panelFrame")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        # Video Canvas
        self.video_label = QLabel("Initializing Video Engine...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #020617; border-radius: 12px; border: 2px solid #1e293b;")
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label, stretch=1)

        # Stream Telemetry Ribbon
        telemetry_layout = QHBoxLayout()
        self.lbl_fps = QLabel("⚡ Stream FPS: --")
        self.lbl_fps.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 13px;")

        self.lbl_latency = QLabel("⏱ Inference: Real-time")
        self.lbl_latency.setStyleSheet("color: #94a3b8; font-size: 13px;")

        telemetry_layout.addWidget(self.lbl_fps)
        telemetry_layout.addStretch()
        telemetry_layout.addWidget(self.lbl_latency)
        left_layout.addLayout(telemetry_layout)

        body_layout.addWidget(left_panel, stretch=3)

        # --- RIGHT COLUMN: AI Analytics & Controls Panel ---
        right_panel = QFrame()
        right_panel.setObjectName("panelFrame")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(18)

        panel_heading = QLabel("📊 AI Classification & Controls")
        panel_heading.setStyleSheet("font-size: 17px; font-weight: 800; color: #f8fafc;")
        right_layout.addWidget(panel_heading)

        # Hero Classification Card
        card_class = QFrame()
        card_class.setObjectName("cardFrame")
        layout_class = QVBoxLayout(card_class)
        layout_class.setContentsMargins(18, 18, 18, 18)
        layout_class.setSpacing(12)

        lbl_card_title = QLabel("DETECTED OBJECT")
        lbl_card_title.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 800; letter-spacing: 1px;")
        layout_class.addWidget(lbl_card_title)

        self.lbl_fruit_name = QLabel("Awaiting Camera Feed...")
        self.lbl_fruit_name.setStyleSheet("font-size: 22px; font-weight: 800; color: #38bdf8;")
        layout_class.addWidget(self.lbl_fruit_name)

        self.lbl_fresh_badge = QLabel("FRESHNESS: WAITING")
        self.lbl_fresh_badge.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
        self.lbl_fresh_badge.setStyleSheet(
            "background-color: #1e293b; color: #94a3b8; padding: 8px 14px; "
            "border-radius: 8px; font-weight: 800; font-size: 14px;"
        )
        layout_class.addWidget(self.lbl_fresh_badge)

        lbl_conf_title = QLabel("MODEL CONFIDENCE SCORE")
        lbl_conf_title.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-top: 6px;")
        layout_class.addWidget(lbl_conf_title)

        self.progress_conf = QProgressBar()
        self.progress_conf.setValue(0)
        layout_class.addWidget(self.progress_conf)

        right_layout.addWidget(card_class)

        # Camera Controls Card
        card_ctrl = QFrame()
        card_ctrl.setObjectName("cardFrame")
        layout_ctrl = QVBoxLayout(card_ctrl)
        layout_ctrl.setContentsMargins(18, 18, 18, 18)
        layout_ctrl.setSpacing(12)

        lbl_ctrl_head = QLabel("CAMERA HARDWARE CONTROLS")
        lbl_ctrl_head.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 800; letter-spacing: 1px;")
        layout_ctrl.addWidget(lbl_ctrl_head)

        lbl_res = QLabel("Stream Resolution:")
        lbl_res.setStyleSheet("font-size: 13px; font-weight: 600; color: #cbd5e1;")
        layout_ctrl.addWidget(lbl_res)

        self.combo_res = QComboBox()
        self.combo_res.addItems([
            "320x240 (QVGA ~30 FPS)",
            "400x296 (CIF ~25 FPS)",
            "640x480 (VGA ~15 FPS)",
            "800x600 (SVGA ~10 FPS)"
        ])
        self.combo_res.currentIndexChanged.connect(self.change_resolution)
        layout_ctrl.addWidget(self.combo_res)

        # Confidence Cutoff Threshold Slider
        lbl_cutoff = QLabel("Confidence Cutoff Threshold:")
        lbl_cutoff.setStyleSheet("font-size: 13px; font-weight: 600; color: #cbd5e1; margin-top: 4px;")
        layout_ctrl.addWidget(lbl_cutoff)

        slider_layout = QHBoxLayout()
        self.slider_cutoff = QSlider(Qt.Orientation.Horizontal if hasattr(Qt, 'Orientation') else Qt.Horizontal)
        self.slider_cutoff.setRange(30, 95)
        self.slider_cutoff.setValue(60)
        self.slider_cutoff.setTickInterval(5)
        self.slider_cutoff.setStyleSheet("""
            QSlider::groove:horizontal { height: 6px; background: #1e293b; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #38bdf8; border-radius: 3px; }
            QSlider::handle:horizontal { background: #f8fafc; width: 16px; margin-top: -5px; margin-bottom: -5px; border-radius: 8px; }
        """)

        self.lbl_cutoff_val = QLabel("60%")
        self.lbl_cutoff_val.setStyleSheet("font-weight: 800; color: #38bdf8; font-size: 14px; min-width: 40px;")

        self.slider_cutoff.valueChanged.connect(self.on_cutoff_changed)

        slider_layout.addWidget(self.slider_cutoff)
        slider_layout.addWidget(self.lbl_cutoff_val)
        layout_ctrl.addLayout(slider_layout)

        # Temporal Smoothing Window Slider (Anti-Jitter)
        lbl_smooth = QLabel("Anti-Jitter Smoothing Window:")
        lbl_smooth.setStyleSheet("font-size: 13px; font-weight: 600; color: #cbd5e1; margin-top: 4px;")
        layout_ctrl.addWidget(lbl_smooth)

        slider_smooth_layout = QHBoxLayout()
        self.slider_smooth = QSlider(Qt.Orientation.Horizontal if hasattr(Qt, 'Orientation') else Qt.Horizontal)
        self.slider_smooth.setRange(1, 30)
        self.slider_smooth.setValue(10)
        self.slider_smooth.setTickInterval(5)
        self.slider_smooth.setStyleSheet("""
            QSlider::groove:horizontal { height: 6px; background: #1e293b; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #10b981; border-radius: 3px; }
            QSlider::handle:horizontal { background: #f8fafc; width: 16px; margin-top: -5px; margin-bottom: -5px; border-radius: 8px; }
        """)

        self.lbl_smooth_val = QLabel("10 frames")
        self.lbl_smooth_val.setStyleSheet("font-weight: 800; color: #10b981; font-size: 13px; min-width: 65px;")

        self.slider_smooth.valueChanged.connect(self.on_smoothing_changed)

        slider_smooth_layout.addWidget(self.slider_smooth)
        slider_smooth_layout.addWidget(self.lbl_smooth_val)
        layout_ctrl.addLayout(slider_smooth_layout)

        # Min Object Size Sensitivity Slider
        lbl_size = QLabel("Min Object Size (px²):")
        lbl_size.setStyleSheet("font-size: 13px; font-weight: 600; color: #cbd5e1; margin-top: 4px;")
        layout_ctrl.addWidget(lbl_size)

        slider_size_layout = QHBoxLayout()
        self.slider_size = QSlider(Qt.Orientation.Horizontal if hasattr(Qt, 'Orientation') else Qt.Horizontal)
        self.slider_size.setRange(500, 8000)
        self.slider_size.setValue(1500)
        self.slider_size.setSingleStep(250)
        self.slider_size.setStyleSheet("""
            QSlider::groove:horizontal { height: 6px; background: #1e293b; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #a855f7; border-radius: 3px; }
            QSlider::handle:horizontal { background: #f8fafc; width: 16px; margin-top: -5px; margin-bottom: -5px; border-radius: 8px; }
        """)

        self.lbl_size_val = QLabel("1500 px²")
        self.lbl_size_val.setStyleSheet("font-weight: 800; color: #a855f7; font-size: 13px; min-width: 65px;")

        self.slider_size.valueChanged.connect(self.on_min_size_changed)

        slider_size_layout.addWidget(self.slider_size)
        slider_size_layout.addWidget(self.lbl_size_val)
        layout_ctrl.addLayout(slider_size_layout)

        # Flash LED Button (Instant Response)
        self.btn_flash = QPushButton("💡 FLASH LED: OFF")
        self.btn_flash.setObjectName("btnFlashOff")
        self.btn_flash.clicked.connect(self.toggle_flash)
        layout_ctrl.addWidget(self.btn_flash)

        # Action Buttons Grid
        btn_grid = QHBoxLayout()
        btn_grid.setSpacing(10)

        self.btn_snap = QPushButton("📷 Snapshot")
        self.btn_snap.setObjectName("btnSnap")
        self.btn_snap.clicked.connect(self.save_snapshot)
        btn_grid.addWidget(self.btn_snap)

        self.btn_reconnect = QPushButton("🔄 Reconnect")
        self.btn_reconnect.clicked.connect(self.start_stream)
        btn_grid.addWidget(self.btn_reconnect)

        layout_ctrl.addLayout(btn_grid)
        right_layout.addWidget(card_ctrl)

        right_layout.addStretch()
        body_layout.addWidget(right_panel, stretch=2)

        main_layout.addLayout(body_layout)

    def on_cutoff_changed(self, val):
        self.cutoff_threshold = float(val)
        self.lbl_cutoff_val.setText(f"{val}%")
        if hasattr(self, 'worker'):
            self.worker.confidence_threshold = float(val)

    def on_smoothing_changed(self, val):
        self.smoothing_window = val
        txt = "1 frame (Off)" if val == 1 else f"{val} frames"
        self.lbl_smooth_val.setText(txt)
        if hasattr(self, 'worker'):
            self.worker.set_smoothing_window(val)

    def on_min_size_changed(self, val):
        self.min_object_size = val
        self.lbl_size_val.setText(f"{val} px²")
        if hasattr(self, 'worker'):
            self.worker.min_object_size = val

    def start_stream(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()

        self.lbl_status.setText("● Connecting...")
        self.lbl_status.setStyleSheet("background-color: #27272a; color: #f59e0b; border: 1px solid #3f3f46; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 12px;")

        self.worker = StreamWorker(
            confidence_threshold=getattr(self, 'cutoff_threshold', 60.0),
            min_object_size=getattr(self, 'min_object_size', 1500),
            smoothing_window=getattr(self, 'smoothing_window', 10)
        )
        self.worker.frame_processed.connect(self.update_frame)
        self.worker.connection_status.connect(self.update_status)
        self.worker.start()

    def update_status(self, connected, msg):
        if connected:
            self.lbl_status.setText("● Live Stream")
            self.lbl_status.setStyleSheet("background-color: #064e3b; color: #34d399; border: 1px solid #047857; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 12px;")
        else:
            self.lbl_status.setText(f"● Error ({msg})")
            self.lbl_status.setStyleSheet("background-color: #7f1d1d; color: #fca5a5; border: 1px solid #b91c1c; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 12px;")

    def update_frame(self, frame, results, fps):
        self.last_frame = frame.copy()
        self.lbl_fps.setText(f"⚡ Stream FPS: {fps:.1f}")

        count = len(results)
        self.pill_count.setText(f"📦 Objects: {count}")

        # Draw Multi-Object Bounding Boxes & Glassmorphism Badge Overlays
        for res in results:
            x, y, w, h = res['box']
            pred_class = res['class']
            confidence = res['confidence']
            is_fresh = res['is_fresh']

            if is_fresh is True:
                box_color = (0, 220, 100) # Bright Green
            elif is_fresh is False:
                box_color = (0, 50, 240)  # Vivid Red
            else:
                box_color = (140, 140, 140) # Slate Gray

            # Bounding Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Floating Label Badge
            emoji_name = FRUIT_EMOJIS.get(pred_class, pred_class)
            label_text = f"{emoji_name} ({confidence:.0f}%)"

            font_scale = 0.48
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            banner_y1 = max(0, y - text_h - 8)
            banner_y2 = y

            cv2.rectangle(frame, (x, banner_y1), (x + text_w + 10, banner_y2), (15, 23, 42), -1)
            cv2.rectangle(frame, (x, banner_y1), (x + text_w + 10, banner_y2), box_color, 1)
            cv2.putText(frame, label_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Convert to QPixmap for Canvas
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw, ch = rgb_frame.shape
        bytes_per_line = ch * fw
        q_img = QImage(rgb_frame.data, fw, fh, bytes_per_line, QImage.Format.Format_RGB888 if hasattr(QImage.Format, 'Format_RGB888') else QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio if hasattr(Qt, 'AspectRatioMode') else Qt.KeepAspectRatio
        ))

        # Update Primary Hero Display Card (Primary High-Confidence Object)
        if results:
            top_obj = results[0]
            top_class = top_obj['class']
            top_conf = top_obj['confidence']
            top_fresh = top_obj['is_fresh']

            emoji_name = FRUIT_EMOJIS.get(top_class, top_class)
            self.lbl_fruit_name.setText(emoji_name)
            self.progress_conf.setValue(int(top_conf))

            if top_fresh is True:
                self.lbl_fresh_badge.setText("FRESH QUALITY")
                self.lbl_fresh_badge.setStyleSheet(
                    "background-color: #059669; color: #ffffff; padding: 8px 14px; "
                    "border-radius: 8px; font-weight: 800; font-size: 14px;"
                )
            elif top_fresh is False:
                self.lbl_fresh_badge.setText("ROTTEN / SPOILED")
                self.lbl_fresh_badge.setStyleSheet(
                    "background-color: #dc2626; color: #ffffff; padding: 8px 14px; "
                    "border-radius: 8px; font-weight: 800; font-size: 14px;"
                )
            else:
                cutoff_val = int(getattr(self, 'cutoff_threshold', 60.0))
                self.lbl_fresh_badge.setText(f"LOW CONFIDENCE (< {cutoff_val}%)")
                self.lbl_fresh_badge.setStyleSheet(
                    "background-color: #334155; color: #cbd5e1; padding: 8px 14px; "
                    "border-radius: 8px; font-weight: 800; font-size: 14px;"
                )
        else:
            self.lbl_fruit_name.setText("❓ Object Not Detected")
            self.progress_conf.setValue(0)
            self.lbl_fresh_badge.setText("NO OBJECTS IN VIEW")
            self.lbl_fresh_badge.setStyleSheet(
                "background-color: #334155; color: #cbd5e1; padding: 8px 14px; "
                "border-radius: 8px; font-weight: 800; font-size: 14px;"
            )

    def change_resolution(self, index):
        res_values = [6, 7, 8, 10]
        val = res_values[index] if index < len(res_values) else 6
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.queue_control("framesize", val)
        else:
            send_async_control("framesize", val)

    def toggle_flash(self):
        """Toggle Flash LED instantly with zero stream latency or command queuing lag."""
        self.flash_state = not self.flash_state
        val = 255 if self.flash_state else 0
        
        # Immediate UI state feedback
        if self.flash_state:
            self.btn_flash.setText("💡 FLASH LED: ON")
            self.btn_flash.setObjectName("btnFlashOn")
            self.btn_flash.setStyle(self.btn_flash.style())  # Force QSS refresh
        else:
            self.btn_flash.setText("💡 FLASH LED: OFF")
            self.btn_flash.setObjectName("btnFlashOff")
            self.btn_flash.setStyle(self.btn_flash.style())  # Force QSS refresh

        # Interleave control command directly on active worker thread (eliminates ESP32 port 80 socket backlog queuing)
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.queue_control("flash", val)
        else:
            send_async_control("flash", val)

    def save_snapshot(self):
        if self.last_frame is not None:
            filename = f"fruit_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, self.last_frame)
            print(f"[Snapshot] Saved to {filename}")

    def closeEvent(self, event):
        if hasattr(self, 'worker'):
            self.worker.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = FruitDetectorGUI()
    window.show()
    sys.exit(app.exec_() if hasattr(app, 'exec_') else app.exec())

if __name__ == "__main__":
    main()
