# 🍎 ESP32-CAM Fruit Spoilage Detector AI Studio

A real-time edge AI vision system powered by **ESP32-CAM** and a **Custom PyTorch Deep Convolutional Neural Network (CNN)** for instant fruit classification, freshness scoring, and spoilage detection.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyQt](https://img.shields.io/badge/PyQt-5%2F6-41CD52?style=for-the-badge&logo=qt&logoColor=white)

---

## 🌟 Key Features

- **⚡ Zero-Latency Streaming (<50ms Launch)**: Deferred background neural network initialization for instant UI startup.
- **💡 Real-time Hardware Interleaving**: Instant ESP32 Flash LED toggle and resolution switching without stream disconnects or buffering lag.
- **❓ Confidence Cutoff Threshold**: Interactive slider (30%–95%) that labels low-confidence items as `Object Not Detected`.
- **📈 Anti-Jitter Temporal Probability Smoothing**: Rolling Softmax probability window (1–30 frames) for silky-smooth, flicker-free confidence percentage readouts.
- **📊 6-Class AI Classifier**:
  - 🍎 Fresh Apple / 🍏 Rotten Apple
  - 🍌 Fresh Banana / 🍌 Rotten Banana
  - 🍊 Fresh Orange / 🍊 Rotten Orange

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Desktop Application
```bash
python main.py
```

---

## 📁 Project Architecture

```
esp-32-Fruit-Spoilage-Detector/
├── main.py                             # Root Application Launcher
├── requirements.txt                    # Project Dependencies
├── README.md                           # Documentation
├── firmware/                           # ESP32-CAM Hardware Firmware
│   └── ESP32Cam_WebServer/
│       └── ESP32Cam_WebServer.ino      # C++ Arduino WebServer & Stream Sketch
├── models/                             # Deep Learning Weights & Checkpoints
│   └── fruit_classifier.pth            # Trained PyTorch Model (~14.9 MB)
├── src/                                # Core Source Code Package
│   ├── app/                            # PyQt Application Components
│   │   ├── gui_app.py                  # Glassmorphism PyQt Studio App
│   │   └── cli_app.py                  # Lightweight Terminal Vision App
│   ├── engine/                         # Deep Learning Engine
│   │   └── model.py                    # PyTorch CNN Architecture & Inference
│   └── utils/                          # Stream & Network Helpers
├── training/                           # Dataset Preparation & Model Training
│   ├── train_classifier.py             # Primary Deep Learning Training Pipeline
│   ├── benchmark_models.py             # Accuracy & Latency Benchmarks
│   ├── download_real_photos_and_train.py
│   └── ultra_fast_robust_train.py
└── evaluation/                         # Model Analytics, Plots & Validation
    ├── generate_plots.py               # Visual Analytics Generator
    ├── classification_2d_plane.png     # Decision Boundary Map
    ├── confusion_matrix.png            # Multiclass Evaluation Matrix
    └── learning_curves.png             # Training & Validation Curves
```
