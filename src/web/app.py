import os
import sys
import time
import cv2
import socket
import urllib.request
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from src.engine.model import (
    FruitCNN,
    detect_and_classify_multi_objects,
    TemporalObjectTracker,
    CLASS_NAMES
)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

# Global State
class WebStreamEngine:
    def __init__(self):
        self.esp32_ip = "192.168.1.100"
        self.stream_url = f"http://{self.esp32_ip}:81/stream"
        self.is_connected = False
        self.min_obj_size = 800
        self.conf_threshold = 0.50
        self.flash_led = False
        self.tracker = TemporalObjectTracker(persistence_frames=4, alpha=0.35)
        
        # Performance metrics
        self.fps = 0.0
        self.detected_objects = []
        self.last_frame = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = FruitCNN(num_classes=6).to(self.device)
        model_path = os.path.join(ROOT_DIR, "models", "fruit_classifier.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[WebEngine] Model loaded successfully from {model_path}")
        self.model.eval()

    def set_esp32_ip(self, ip):
        self.esp32_ip = ip.strip()
        self.stream_url = f"http://{self.esp32_ip}:81/stream" if ":81" not in self.esp32_ip else f"http://{self.esp32_ip}"
        if not self.stream_url.startswith("http://"):
            self.stream_url = "http://" + self.stream_url
        if not self.stream_url.endswith("/stream"):
            self.stream_url = self.stream_url.rstrip("/") + "/stream"
        self.tracker.tracked_objects = []

    def toggle_flash_led(self, state=None):
        if state is None:
            self.flash_led = not self.flash_led
        else:
            self.flash_led = bool(state)
        
        # Send asynchronous control command to ESP32
        val = 255 if self.flash_led else 0
        control_url = f"http://{self.esp32_ip}/control?var=flash&val={val}"
        try:
            req = urllib.request.Request(control_url, headers={'Connection': 'close'})
            urllib.request.urlopen(req, timeout=1.5)
        except Exception as e:
            print(f"[WebEngine] LED control error: {e}")
        return self.flash_led

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

def generate_mjpeg_stream():
    """Generator function yielding MJPEG video frames."""
    frame_count = 0
    start_time = time.time()
    
    while True:
        try:
            # Connect to stream
            stream = urllib.request.urlopen(engine.stream_url, timeout=3.0)
            engine.is_connected = True
            bytes_buffer = b''
            
            for chunk in iter(lambda: stream.read(1024), b''):
                bytes_buffer += chunk
                a = bytes_buffer.find(b'\xff\xd8') # JPEG start
                b = bytes_buffer.find(b'\xff\xd9') # JPEG end
                
                if a != -1 and b != -1:
                    jpg = bytes_buffer[a:b+2]
                    bytes_buffer = bytes_buffer[b+2:]
                    
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                        
                    # Calculate FPS
                    frame_count += 1
                    elapsed = time.time() - start_time
                    if elapsed >= 1.0:
                        engine.fps = frame_count / elapsed
                        frame_count = 0
                        start_time = time.time()

                    # Run OpenCV Multi-Object Detection + PyTorch Inference
                    det_results = detect_and_classify_multi_objects(
                        frame,
                        engine.model,
                        device=engine.device,
                        confidence_threshold=engine.conf_threshold * 100.0,
                        min_area=engine.min_obj_size
                    )
                    
                    # Update Anti-Jitter Box Tracker
                    tracked_boxes = engine.tracker.update(det_results)
                    engine.detected_objects = tracked_boxes
                    
                    # Render Bounding Boxes & Badges
                    annotated_frame = frame.copy()
                    for item in tracked_boxes:
                        bx, by, bw, bh = item['box']
                        cls_name = item['class']
                        conf = item['confidence'] / 100.0 if item['confidence'] > 1.0 else item['confidence']
                        
                        # Color coding
                        color = (34, 197, 94) if item.get('is_fresh', False) else (239, 68, 68)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (bx, by), (bx + bw, by + bh), color, 2)
                        
                        # Floating label badge
                        label_text = f"{cls_name} ({conf*100:.0f}%)"
                        cv2.rectangle(annotated_frame, (bx, max(0, by - 25)), (bx + bw, by), color, -1)
                        cv2.putText(annotated_frame, label_text, (bx + 5, max(15, by - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    engine.last_frame = annotated_frame.copy()
                    
                    # Encode output frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            engine.is_connected = False
            time.sleep(1.0)
            # Yield placeholder offline frame
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, "ESP32 Offline / Reconnecting...", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 116, 139), 1, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

import numpy as np

# Routes
@app.route('/')
def index():
    return render_template('index.html', local_ip=get_local_ip(), esp32_ip=engine.esp32_ip)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    counts = {}
    for obj in engine.detected_objects:
        lbl = obj.get('class', 'Unknown')
        counts[lbl] = counts.get(lbl, 0) + 1
        
    return jsonify({
        'connected': engine.is_connected,
        'fps': round(engine.fps, 1),
        'esp32_ip': engine.esp32_ip,
        'flash_led': engine.flash_led,
        'min_obj_size': engine.min_obj_size,
        'conf_threshold': engine.conf_threshold,
        'object_count': len(engine.detected_objects),
        'class_counts': counts,
        'detected_objects': [
            {
                'label': o.get('class', 'Unknown'),
                'conf': round(o.get('confidence', 0.0) / 100.0 if o.get('confidence', 0.0) > 1.0 else o.get('confidence', 0.0), 2),
                'box': o.get('box', (0,0,0,0))
            } for o in engine.detected_objects
        ]
    })

@app.route('/api/config', methods=['POST'])
def api_config():
    data = request.json or {}
    if 'esp32_ip' in data:
        engine.set_esp32_ip(data['esp32_ip'])
    if 'min_obj_size' in data:
        engine.min_obj_size = int(data['min_obj_size'])
    if 'conf_threshold' in data:
        engine.conf_threshold = float(data['conf_threshold'])
    if 'flash_led' in data:
        engine.toggle_flash_led(data['flash_led'])
    return jsonify({'status': 'ok', 'esp32_ip': engine.esp32_ip, 'flash_led': engine.flash_led})

@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    if engine.last_frame is None:
        return jsonify({'status': 'error', 'message': 'No video frame available'}), 400
        
    snapshots_dir = os.path.join(ROOT_DIR, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    filename = f"snapshot_{int(time.time())}.jpg"
    filepath = os.path.join(snapshots_dir, filename)
    
    cv2.imwrite(filepath, engine.last_frame)
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
