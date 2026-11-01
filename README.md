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
pip install torch torchvision opencv-python pyqt5 requests numpy pillow
```

### 2. Launch Application
```bash
python gui_app.py
```

---

## 📁 Repository Structure

```
├── gui_app.py                       # Main Glassmorphism PyQt GUI Application
├── model.py                         # PyTorch FruitCNN Architecture & Inference Engine
├── train_on_official_kaggle.py     # Deep Learning Training Script
├── fruit_classifier.pth             # Trained PyTorch Model Weights (~14.9 MB)
├── generate_plots.py                # Visual Analytics & Learning Curves Generator
├── classification_2d_plane.png      # High-dimensional Feature Decision Boundary
└── confusion_matrix.png             # Multiclass Classification Evaluation Matrix
```
