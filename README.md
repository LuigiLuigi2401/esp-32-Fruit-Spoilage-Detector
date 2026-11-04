# 🍎 ESP32-CAM AI Fruit Spoilage Detector

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-Modern%20UI-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://www.qt.io/)
[![ESP32-CAM](https://img.shields.io/badge/Hardware-ESP32--CAM-000000?style=for-the-badge&logo=espressif&logoColor=white)](https://www.espressif.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

An end-to-end, real-time edge AI computer vision system for fruit freshness classification and spoilage detection. Powered by **ESP32-CAM** MJPEG video streaming, **OpenCV** multi-candidate region proposal, **PyTorch MobileNetV2** deep learning inference, and a modern **PyQt6** glassmorphic desktop & **Flask** web application.

---

## 📸 System Highlights

* **Multi-Object Localization & Batching**: Localizes multiple fruits simultaneously on live stream feeds (including PC screen displays, cards, or conveyor belts) and processes regions of interest in a single batched PyTorch tensor forward pass (~22ms CPU latency).
* **Temporal Multi-Object Anti-Jitter Tracking**: Utilizes IoU spatial matching, 4-frame noise persistence grace periods, and Exponential Moving Average (EMA) coordinate smoothing ($\alpha = 0.35$) to eliminate bounding box flickering.
* **Out-Of-Distribution Non-Fruit Rejection**: Rejects non-fruit background objects (wires, keyboards, monitors) using geometric aspect-ratio constraints ($0.25 \le w/h \le 3.8$) and Softmax logit margin thresholding ($\Delta P \ge 0.12$).
* **Hardware Interleaving Control**: Controls ESP32-CAM Flash LED brightness and stream resolution dynamically over asynchronous non-blocking HTTP socket requests.

---

## 📐 System Architecture Pipeline

```mermaid
flowchart TD
    A["ESP32-CAM Sensor - WiFi MJPEG Stream QVGA 320x240 @ 25 FPS"] --> B["OpenCV Multi-Candidate Region Proposal"]
    
    subgraph Candidate_Proposal ["OpenCV Region Proposal"]
        B --> B1["Saturation Channel Masking S >= 25"]
        B1 --> B2["Foreground Otsu Adaptive Binarization"]
        B2 --> B3["Canny Edge Gradient Closure"]
    end

    Candidate_Proposal --> C["FILTER: Hierarchical Contour Filtering"]
    C --> D["NMS: Non-Maximum Suppression IoU > 0.40"]
    
    D --> E["Batched PyTorch MobileNetV2 Forward Pass"]
    
    E --> F["Temporal Anti-Jitter Tracker & GUI App"]
```

---

## 📁 Repository Directory Structure

```
esp-32-Fruit-Spoilage-Detector/
├── main.py                      # Primary PyQt6 glassmorphism GUI entry point
├── web_main.py                  # Flask web application entry point (network access)
├── requirements.txt             # Python dependencies
├── README.md                    # System documentation and wiki
│
├── src/                         # Core source code package
│   ├── app/                     # PyQt6 GUI Application
│   │   └── gui_app.py           # Main window, video canvas, controls & snapshot manager
│   ├── engine/                  # Computer Vision & Deep Learning Engine
│   │   └── model.py             # FruitCNN, OpenCV region proposal, Tracker, NMS
│   └── utils/                   # Shared utility modules
│
├── docs/                        # Project Documentation & Reports
│   ├── figures/                 # SVG & PNG Report Figures (Confusion matrix, 2D plane, etc.)
│   │   ├── classification_2d_plane.svg
│   │   ├── confusion_matrix.svg
│   │   ├── learning_curves.svg
│   │   └── resolution_benchmark.svg
│   └── reports/                 # Project Progress Reports & Literature Reviews
│       ├── IEEE_Literature_Review_Fruit_Spoilage.pdf
│       └── Report_6_Week6_AI_Hardware_Integration.pdf
│
├── models/                      # Trained Model Artifacts
│   └── fruit_classifier.pth     # PyTorch MobileNetV2 fine-tuned weights
│
├── firmware/                    # Hardware Microcontroller Code
│   └── esp32_cam_streamer.ino   # ESP32-CAM Arduino C++ WiFi MJPEG web server
│
├── training/                    # Neural Network Training Pipelines
│   ├── train_on_official_kaggle.py # Full 20-epoch training on 6-class dataset
│   └── ultra_fast_robust_train.py # 12-epoch fast head refitting & noise augmentation
│
├── evaluation/                  # Evaluation & Benchmarking Scripts
│   ├── generate_plots.py        # Generates 2D PCA plane and Confusion Matrix
│   └── generate_report_figures.py # Generates Learning Curves & Hardware Benchmarks
│
└── snapshots/                   # Saved camera snapshots and captured ROIs
```

---

## ⚡ Quick Start & Installation

### 1. Hardware Setup & Dynamic Wi-Fi Provisioning (ESP32-CAM)
1. Open [`firmware/ESP32Cam_WebServer/ESP32Cam_WebServer.ino`](file:///d:/Projects/Arduino/FruitClassification/firmware/ESP32Cam_WebServer/ESP32Cam_WebServer.ino) in Arduino IDE.
2. Select board **AI Thinker ESP32-CAM** and flash the firmware.

#### 📶 Dynamic Wi-Fi Provisioning (No Code Changes Needed!)
Instead of hardcoding Wi-Fi credentials in the C++ file, you can dynamically configure the ESP32-CAM Wi-Fi network across different environments (e.g., Home, University, Office, Mobile Hotspot):

* **Option A: Python Serial CLI Tool (Recommended over USB)**:
  Connect the ESP32-CAM to your PC via USB and run:
  ```bash
  # Interactive Mode: Scans surrounding Wi-Fi networks and prompts for password
  python -m src.cli.wifi_config

  # Direct CLI Command:
  python -m src.cli.wifi_config --ssid "TargetWiFiName" --password "TargetPassword"
  ```
  *The credentials are stored persistently in the ESP32's internal NVS flash memory across power reboots!*

* **Option B: Fallback Smartphone/Browser Access Point**:
  If Wi-Fi fails to connect after 15 seconds (e.g. when powered by a power bank in a new location), the ESP32-CAM automatically creates a setup hotspot named **`ESP32-CAM-Setup`**.
  1. Connect your phone or laptop to Wi-Fi `ESP32-CAM-Setup`.
  2. Open `http://192.168.4.1` in your browser.
  3. Enter the Wi-Fi name & password and click **Save & Connect**!

> [!NOTE]
> Make sure your ESP32-CAM and host computer/device are connected to the same Wi-Fi subnet for streaming.

### 2. Desktop & Web Application Launch

#### Option A: Modern Desktop Application (PyQt6)
Launch the desktop application:
```bash
python main.py
```

#### Option B: Network Web Application (Flask - Access Anywhere on Local Network)
Launch the web server to access the live camera feed and controls from **any device (phone, tablet, iPad, PC)** on your local Wi-Fi network:
```bash
python web_main.py
```
Open your browser and navigate to:
* **Local Computer**: `http://localhost:5000`
* **Mobile / Other Devices**: `http://<YOUR_LOCAL_IP>:5000` (Use the local IP displayed in terminal output)

> [!TIP]
> The Flask web application is optimized for mobile touchscreens, making it easy to monitor fruit freshness on smartphones or iPads near the camera rig!

---

## 📊 Performance Benchmarks & Figures

### **1. MobileNetV2 Training Convergence (20 Epochs)**
![Learning Curves](docs/figures/learning_curves.png)
*Figure 1: Cross-entropy loss and top-1 accuracy convergence over 20 training epochs.*

### **2. Confusion Matrix under Compression Stream Noise**
![Confusion Matrix](docs/figures/confusion_matrix.png)
*Figure 2: Model achieving 98.3% classification accuracy across 6 fruit categories under simulated ESP32 JPEG compression noise.*

### **3. 2D PCA Feature Subspace**
![2D Classification Plane](docs/figures/classification_2d_plane.png)
*Figure 3: 2D Principal Component Analysis (PCA) feature subspace demonstrating linear cluster separation between fresh and rotten categories.*

### **4. Hardware Resolution & Latency Benchmarks**
![Hardware Benchmarks](docs/figures/resolution_benchmark.png)
*Figure 4: Frame rate (FPS) vs PyTorch CPU inference latency across stream resolutions.*

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
