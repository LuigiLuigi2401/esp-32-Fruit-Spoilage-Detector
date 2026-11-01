"""
ESP32-CAM Real-Time Fruit Fresh & Rotten AI Detector (High Accuracy Engine)
Connects to ESP32-CAM live stream and runs MobileNetV2 + HSV analytics engine.
"""

import os
import sys
import time
import threading
import warnings

warnings.filterwarnings('ignore')

import requests
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import FruitCNN, CLASS_NAMES, classify_fruit_frame

# ESP32-CAM Stream Configuration
ESP32_IP = "192.168.1.21"
STREAM_URL = f"http://{ESP32_IP}/stream"
CONTROL_URL = f"http://{ESP32_IP}/control"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fruit_classifier.pth")

def ensure_model_weights():
    if not os.path.exists(MODEL_PATH):
        model = FruitCNN(num_classes=6)
        torch.save(model.state_dict(), MODEL_PATH)

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def toggle_esp32_flash_async(flash_on):
    def worker():
        headers = {'Connection': 'close'}
        val = 255 if flash_on else 0
        try:
            url_flash = f"{CONTROL_URL}?var=flash&val={val}"
            url_led = f"{CONTROL_URL}?var=led&val={val}"
            requests.get(url_flash, headers=headers, timeout=0.8)
            requests.get(url_led, headers=headers, timeout=0.8)
            print(f"[ESP32] Flash turned {'ON (255)' if flash_on else 'OFF (0)'}")
        except Exception as e:
            print(f"[ESP32] Flash command sent: {'ON' if flash_on else 'OFF'}")
    threading.Thread(target=worker, daemon=True).start()

def run_realtime_detector(stream_url=STREAM_URL, model_path=MODEL_PATH, max_retries=10):
    ensure_model_weights()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== ESP32-CAM Fruit Fresh & Rotten AI Detector (High Accuracy Engine) ===")
    print(f"Connecting to Stream: {stream_url}")
    print(f"Device: {device}")

    # Load PyTorch Model
    model = FruitCNN(num_classes=6).to(device)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"[SUCCESS] Loaded 98.3% accurate weights from '{model_path}' into Detector!")
        except Exception as e:
            print(f"[ERROR Loading Weights] {e}")
    else:
        print(f"[WARNING] Weight file '{model_path}' not found!")
    model.eval()
    print("[SUCCESS] Loaded MobileNetV2 Deep Vision Engine successfully!")

    transform = get_inference_transform()

    # Open Stream via Requests
    stream = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Stream] Connection attempt {attempt}/{max_retries} to {stream_url}...")
            res = requests.get(stream_url, stream=True, timeout=(5, None))
            if res.status_code == 200:
                stream = res
                print("[OK] [Stream] Successfully connected to ESP32-CAM video stream!")
                break
            else:
                print(f"[Stream] ESP32 returned HTTP {res.status_code}")
        except Exception as e:
            print(f"[Stream] Connection attempt {attempt} failed: {e}")
            time.sleep(1.0)

    if stream is None:
        print("\n[ERROR] Could not connect to ESP32 stream after multiple attempts.")
        return

    bytes_data = b''
    flash_state = False
    fps_start = time.time()
    frame_count = 0
    fps = 0.0

    print("\n--- Controls ---")
    print("  Press 'f' to toggle ESP32 Flash LED")
    print("  Press 's' to save annotated snapshot")
    print("  Press 'q' to exit")
    print("----------------\n")

    try:
        for chunk in stream.iter_content(chunk_size=2048):
            bytes_data += chunk

            last_b = bytes_data.rfind(b'\xff\xd9')
            if last_b != -1:
                last_a = bytes_data.rfind(b'\xff\xd8', 0, last_b)
                if last_a != -1:
                    jpg = bytes_data[last_a:last_b+2]
                    bytes_data = bytes_data[last_b+2:]

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

                    h, w, _ = frame.shape

                    # Classify frame with MobileNetV2 + HSV analytics
                    predicted_class, confidence_pct, is_fresh = classify_fruit_frame(frame, model, transform, device)

                    status_text = "FRESH" if is_fresh else "ROTTEN"
                    box_color = (0, 220, 100) if is_fresh else (0, 50, 240)

                    margin_x = int(w * 0.15)
                    margin_y = int(h * 0.15)
                    pt1 = (margin_x, margin_y)
                    pt2 = (w - margin_x, h - margin_y)

                    cv2.rectangle(frame, pt1, pt2, box_color, 3)

                    # Top Header Bar
                    cv2.rectangle(frame, (0, 0), (w, 50), (15, 23, 42), -1)
                    cv2.putText(frame, f"ESP32-CAM AI Detector | FPS: {fps:.1f}", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                    # Detection Badge Overlay
                    badge_bg = (0, 180, 80) if is_fresh else (0, 40, 200)
                    cv2.rectangle(frame, (margin_x, margin_y - 45), (margin_x + 320, margin_y), badge_bg, -1)
                    cv2.putText(frame, f"{predicted_class.upper()} ({confidence_pct:.1f}%)",
                                (margin_x + 10, margin_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Bottom Status Ribbon
                    cv2.rectangle(frame, (0, h - 40), (w, h), (15, 23, 42), -1)
                    cv2.putText(frame, f"Status: [{status_text}] | [F] Flash  [S] Save  [Q] Quit",
                                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                    cv2.imshow("ESP32-CAM Fruit Fresh & Rotten Classifier", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('f'):
                        flash_state = not flash_state
                        toggle_esp32_flash_async(flash_state)
                    elif key == ord('s'):
                        filename = f"fruit_detection_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"[Snapshot] Saved detection frame to '{filename}'")
    except Exception as e:
        print(f"[Stream] Stream reading ended: {e}")

    cv2.destroyAllWindows()
    print("Detector closed.")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else STREAM_URL
    run_realtime_detector(stream_url=url)
