# 🍎 ESP32-CAM AI Fruit Spoilage Detector

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-Modern%20UI-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://www.qt.io/)
[![ESP32-CAM](https://img.shields.io/badge/Hardware-ESP32--CAM-000000?style=for-the-badge&logo=espressif&logoColor=white)](https://www.espressif.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.style=for-the-badge)](LICENSE)

An end-to-end, real-time edge AI computer vision system for fruit freshness classification and spoilage detection. Powered by **ESP32-CAM** MJPEG video streaming, **OpenCV** multi-candidate region proposal, **PyTorch MobileNetV2** deep learning inference, and a modern **PyQt6** glassmorphic desktop application.

---

## 📸 System Highlights

* **Multi-Object Localization & Batching**: Localizes multiple fruits simultaneously on live stream feeds (including PC screen displays, cards, or conveyor belts) and processes regions of interest in a single batched PyTorch tensor forward pass (~22ms CPU latency).
* **Temporal Multi-Object Anti-Jitter Tracking**: Utilizes IoU spatial matching, 4-frame noise persistence grace periods, and Exponential Moving Average (EMA) coordinate smoothing ($\alpha = 0.35$) to eliminate bounding box flickering.
* **Out-Of-Distribution Non-Fruit Rejection**: Rejects non-fruit background objects (wires, keyboards, monitors) using geometric aspect-ratio constraints ($0.25 \le w/h \le 3.8$) and Softmax logit margin thresholding ($\Delta P \ge 0.12$).
* **Hardware Interleaving Control**: Controls ESP32-CAM Flash LED brightness and stream resolution dynamically over asynchronous non-blocking HTTP socket requests.

---

## 📐 System Architecture Pipeline

```
 [ ESP32-CAM Sensor ]
         │ (WiFi MJPEG Stream: QVGA 320x240 @ 25 FPS)
         ▼
 [ OpenCV Multi-Candidate Region Proposal ]
 ├── Saturation Channel Masking (S ≥ 25)
 ├── Foreground Otsu Adaptive Binarization
 └── Canny Edge Gradient Closure
         │
         ▼
 [ Hierarchical Contour & Parent Container Filtering ]
 └── Suppresses outer monitor frames & desk borders
         │
         ▼
 [ Non-Maximum Suppression (NMS IoU > 0.40) ]
         │
         ▼
 [ Batched PyTorch MobileNetV2 Forward Pass ]
 └── ROI Crops (150x150) -> Stacked Tensor -> Logit Margin Check
         │
         ▼
 [ Temporal Anti-Jitter Tracker & Desktop GUI ]
 └── EMA Bounding Box Smoothness + Floating Glassmorphism Badges
```

---

## 📁 Repository Directory Structure

```
esp-32-Fruit-Spoilage-Detector/
├── main.py                      # Primary desktop application entry point
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

### 1. Hardware Setup (ESP32-CAM)
1. Open [`firmware/esp32_cam_streamer.ino`](file:///d:/Projects/Arduino/FruitClassification/firmware/esp32_cam_streamer.ino) in Arduino IDE.
2. Set your WiFi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. Select board **AI Thinker ESP32-CAM** and flash the firmware.
4. Note the IP address printed in the Serial Monitor (e.g. `http://192.168.1.100`).

### 2. Desktop Software Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/LuigiLuigi2401/esp-32-Fruit-Spoilage-Detector.git
   cd esp-32-Fruit-Spoilage-Detector
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the application:
   ```bash
   python main.py
   ```
4. Enter the ESP32-CAM IP address in the GUI top bar and click **Connect**.

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

Distributed under the MIT License. See `LICENSE` for more information.
