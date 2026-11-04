import os
import sys
import time
import cv2
import socket
import threading
import urllib.request
from collections import deque
import numpy as np
import torch
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.engine.model import (
    FruitCNN,
    detect_and_classify_multi_objects,
    TemporalObjectTracker,
    CLASS_NAMES
)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

class WebStreamEngine:
    def __init__(self):
        self.esp32_ip = "10.107.58.11"
        self.stream_url = f"http://{self.esp32_ip}/stream"
        self.control_url = f"http://{self.esp32_ip}/control"
        self.is_connected = False
        self.min_obj_size = 800
        self.conf_threshold = 0.50
        self.flash_led = False
        self.tracker = TemporalObjectTracker(persistence_frames=4, alpha=0.35)
        
        # Lock and frame buffers
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0
        self.detected_objects = []
        self.last_raw_frame = None
        self.last_jpeg_bytes = None
        self.pending_controls = deque()
        
        # Load Deep Learning Engine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FruitCNN(num_classes=7).to(self.device)
        model_path = os.path.join(ROOT_DIR, "models", "fruit_classifier.pth")
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[WebEngine] PyTorch Model loaded successfully ({self.device})")
            except Exception as e:
                print(f"[WebEngine Error] Failed to load model weights: {e}")
        self.model.eval()

        # Start Background Streaming Thread
        self.thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.thread.start()

    def set_esp32_ip(self, ip):
        with self.lock:
            self.esp32_ip = ip.strip()
            ip_clean = self.esp32_ip.replace("http://", "").replace("https://", "").rstrip("/")
            if "/stream" in ip_clean:
                self.stream_url = f"http://{ip_clean}"
            else:
                self.stream_url = f"http://{ip_clean}/stream"
            
            base_host = ip_clean.split(":")[0].split("/")[0]
            self.esp32_ip = base_host
            self.control_url = f"http://{self.esp32_ip}/control"
            self.tracker.tracked_objects = []

    def toggle_flash_led(self, state=None):
        if state is None:
            self.flash_led = not self.flash_led
        else:
            self.flash_led = bool(state)
        
        val = 255 if self.flash_led else 0
        self.pending_controls.append(("flash", val))
        return self.flash_led

    def _stream_worker(self):
        """Dedicated high-speed stream worker thread with zero-latency buffer purging."""
        print("[WebEngine Worker] Background Camera Thread Started.")
        
        while self.running:
            with self.lock:
                current_stream_url = self.stream_url
                
            try:
                # Open MJPEG HTTP Connection
                req = urllib.request.Request(current_stream_url, headers={'User-Agent': 'Mozilla/5.0'})
                stream = urllib.request.urlopen(req, timeout=3.0)
                
                with self.lock:
                    self.is_connected = True
                print(f"[WebEngine Worker] Connected to ESP32 stream: {current_stream_url}")
                
                bytes_buffer = b''
                frame_count = 0
                fps_start = time.time()
                
                while self.running:
                    # Execute queued LED control commands
                    if self.pending_controls:
                        var_name, val = self.pending_controls.popleft()
                        try:
                            stream.close()
                        except Exception:
                            pass
                        time.sleep(0.02)
                        try:
                            ctrl_url = f"{self.control_url}?var={var_name}&val={val}"
                            ctrl_req = urllib.request.Request(ctrl_url, headers={'Connection': 'close'})
                            urllib.request.urlopen(ctrl_req, timeout=1.0)
                        except Exception as ce:
                            print(f"[WebEngine Control Error] {ce}")
                        break # Re-open stream cleanly
                    
                    # Read larger chunk to rapidly drain socket buffer
                    chunk = stream.read(16384)
                    if not chunk:
                        break
                    bytes_buffer += chunk
                    
                    # ZERO-LATENCY ALGORITHM: Always find the LATEST complete JPEG frame
                    last_b = bytes_buffer.rfind(b'\xff\xd9')
                    if last_b == -1:
                        continue
                        
                    last_a = bytes_buffer.rfind(b'\xff\xd8', 0, last_b)
                    if last_a == -1:
                        continue
                        
                    # Extract latest JPG frame safely
                    jpg_data = bytes_buffer[last_a:last_b+2]
                    bytes_buffer = bytes_buffer[last_b+2:] # Purge stale historical buffer
                    
                    if not jpg_data or len(jpg_data) < 200:
                        continue

                    try:
                        frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    except Exception:
                        continue

                    if frame is None or frame.size == 0:
                        continue
                        
                    # Calculate FPS
                    frame_count += 1
                    elapsed = time.time() - fps_start
                    if elapsed >= 1.0:
                        calc_fps = frame_count / elapsed
                        with self.lock:
                            self.fps = calc_fps
                        frame_count = 0
                        fps_start = time.time()
                        
                    # Get current vision thresholds
                    with self.lock:
                        min_area = self.min_obj_size
                        conf_thresh = self.conf_threshold * 100.0
                        
                    # Run OpenCV Multi-Object Detection + PyTorch Inference
                    try:
                        det_results = detect_and_classify_multi_objects(
                            frame,
                            self.model,
                            device=self.device,
                            confidence_threshold=conf_thresh,
                            min_area=min_area
                        )
                    except Exception as de:
                        print(f"[Vision Engine Warning] {de}")
                        det_results = []
                    
                    # Anti-Jitter Bounding Box Tracker
                    tracked_boxes = self.tracker.update(det_results)
                    
                    # Render Annotations
                    annotated_frame = frame.copy()
                    for item in tracked_boxes:
                        bx, by, bw, bh = item['box']
                        cls_name = item['class']
                        conf = item['confidence'] / 100.0 if item['confidence'] > 1.0 else item['confidence']
                        
                        color = (34, 197, 94) if item.get('is_fresh', False) else (239, 68, 68)
                        
                        cv2.rectangle(annotated_frame, (bx, by), (bx + bw, by + bh), color, 2)
                        label_text = f"{cls_name} ({conf*100:.0f}%)"
                        cv2.rectangle(annotated_frame, (bx, max(0, by - 25)), (bx + bw, by), color, -1)
                        cv2.putText(annotated_frame, label_text, (bx + 5, max(15, by - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Encode output frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ret and buffer is not None:
                        with self.lock:
                            self.last_raw_frame = annotated_frame.copy()
                            self.last_jpeg_bytes = buffer.tobytes()
                            self.detected_objects = tracked_boxes
                            
            except Exception as e:
                print(f"[WebEngine Stream Warning] Re-establishing stream: {e}")
                with self.lock:
                    self.is_connected = False
                time.sleep(0.5)

engine = WebStreamEngine()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def generate_mjpeg_response():
    """High-speed MJPEG stream generator serving pre-rendered background frames."""
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(blank, "Connecting to ESP32-CAM...", (25, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)
    _, blank_buf = cv2.imencode('.jpg', blank)
    blank_bytes = blank_buf.tobytes()
    
    while True:
        with engine.lock:
            jpeg_bytes = engine.last_jpeg_bytes
            
        if jpeg_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
            
        time.sleep(0.033) # Smooth ~30 FPS frame rate

@app.route('/')
def index():
    return render_template('index.html', local_ip=get_local_ip(), esp32_ip=engine.esp32_ip)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_response(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with engine.lock:
        conn = engine.is_connected
        fps_val = round(engine.fps, 1)
        ip_val = engine.esp32_ip
        flash_val = engine.flash_led
        min_sz = engine.min_obj_size
        conf_th = engine.conf_threshold
        objs = list(engine.detected_objects)
        
    counts = {}
    for obj in objs:
        lbl = obj.get('class', 'Unknown')
        counts[lbl] = counts.get(lbl, 0) + 1
        
    return jsonify({
        'connected': conn,
        'fps': fps_val,
        'esp32_ip': ip_val,
        'flash_led': flash_val,
        'min_obj_size': min_sz,
        'conf_threshold': conf_th,
        'object_count': len(objs),
        'class_counts': counts,
        'detected_objects': [
            {
                'label': o.get('class', 'Unknown'),
                'conf': round(o.get('confidence', 0.0) / 100.0 if o.get('confidence', 0.0) > 1.0 else o.get('confidence', 0.0), 2),
                'box': o.get('box', (0,0,0,0))
            } for o in objs
        ]
    })

@app.route('/api/config', methods=['POST'])
def api_config():
    data = request.json or {}
    if 'esp32_ip' in data:
        engine.set_esp32_ip(data['esp32_ip'])
    if 'min_obj_size' in data:
        with engine.lock:
            engine.min_obj_size = int(data['min_obj_size'])
    if 'conf_threshold' in data:
        with engine.lock:
            engine.conf_threshold = float(data['conf_threshold'])
    if 'flash_led' in data:
        engine.toggle_flash_led(data['flash_led'])
    return jsonify({'status': 'ok', 'esp32_ip': engine.esp32_ip, 'flash_led': engine.flash_led})

@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    with engine.lock:
        frame = engine.last_raw_frame.copy() if engine.last_raw_frame is not None else None
        
    if frame is None:
        return jsonify({'status': 'error', 'message': 'No video frame available'}), 400
        
    snapshots_dir = os.path.join(ROOT_DIR, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    filename = f"snapshot_{int(time.time())}.jpg"
    filepath = os.path.join(snapshots_dir, filename)
    
    cv2.imwrite(filepath, frame)
    return jsonify({'status': 'ok', 'filename': filename, 'url': f'/snapshots/{filename}'})

@app.route('/snapshots/<filename>')
def serve_snapshot(filename):
    snapshots_dir = os.path.join(ROOT_DIR, "snapshots")
    return send_from_directory(snapshots_dir, filename)

def run_web_app(host="0.0.0.0", port=5000):
    local_ip = get_local_ip()
    print("=" * 60)
    print("  ESP32-CAM AI Fruit Detector Web Server")
    print(f"  Local Access:   http://localhost:{port}")
    print(f"  Network Access: http://{local_ip}:{port}")
    print("=" * 60)
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    run_web_app()
