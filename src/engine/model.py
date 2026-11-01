"""
ESP32-CAM Fruit Fresh & Rotten Classifier Model Architecture & Inference Engine
Uses MobileNetV2 Deep Vision Neural Network trained on Official Kaggle Dataset (98.30% Accuracy).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Alphabetical class mapping matching ImageFolder & Kaggle Dataset
CLASS_NAMES = [
    "Fresh Apple",
    "Fresh Banana",
    "Fresh Orange",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Orange"
]

CLASS_KEYS = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

class FruitCNN(nn.Module):
    """
    High-accuracy MobileNetV2 Deep Vision Architecture for Fruits Fresh & Rotten Classification.
    Input size: (3, 150, 150)
    """
    def __init__(self, num_classes=6):
        super(FruitCNN, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.backbone = models.mobilenet_v2(weights=weights)
        in_feats = self.backbone.classifier[1].in_features
        
        self.features_extractor = self.backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feats, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x):
        x = self.features_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        feats = self.extract_features(x)
        return self.classifier(feats)

def get_default_transform():
    return transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def classify_fruit_frame(bgr_frame, model=None, transform=None, device="cpu", confidence_threshold=60.0, prob_history=None):
    """
    Classifies a BGR image frame using PyTorch Deep Neural Network with optional temporal probability smoothing.
    Returns: (pred_class_name, confidence_pct, is_fresh)
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return "Unknown", 0.0, None

    if transform is None:
        transform = get_default_transform()

    try:
        # Convert BGR (OpenCV) to RGB PIL Image
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Preprocess input tensor
        tensor = transform(pil_img).unsqueeze(0).to(device)

        if model is not None:
            model.eval()
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            # Optional Rolling Temporal Window Averaging (Reduces Number & Label Jitter)
            if prob_history is not None:
                prob_history.append(probs)
                avg_probs = np.mean(prob_history, axis=0)
            else:
                avg_probs = probs

            pred_idx = int(np.argmax(avg_probs))
            conf_pct = float(avg_probs[pred_idx] * 100.0)

            # Enforce Confidence Cutoff Threshold on Smoothed Probabilities
            if conf_pct < confidence_threshold:
                return "Object Not Detected", conf_pct, None

            pred_class_name = CLASS_NAMES[pred_idx]
            is_fresh = "Fresh" in pred_class_name
            return pred_class_name, conf_pct, is_fresh
        else:
            return "Model Not Loaded", 0.0, None

    except Exception as e:
        print(f"[Inference Error] {e}")
        return "Error", 0.0, None

def contains_box(parent, child):
    """Returns True if parent box contains child box center and is significantly larger."""
    px, py, pw, ph = parent
    cx, cy, cw, ch = child
    
    ccx = cx + cw / 2.0
    ccy = cy + ch / 2.0
    
    return (px <= ccx <= px + pw) and (py <= ccy <= py + ph) and (pw * ph > cw * ch * 1.25)

def extract_object_regions(bgr_frame, min_area=1500, max_objects=8):
    """
    Uses OpenCV Multi-level Contour Retrieval & Canny Edge Gradients to detect
    individual objects (including multiple images on PC screens, paper sheets, and trays).
    Returns a list of bounding boxes [(x, y, w, h), ...]
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return []

    h, w, _ = bgr_frame.shape

    # Convert to HSV & Grayscale for robust foreground object segmentation
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    # Blur to smooth image noise & screen reflections
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Canny Edge Detection (Ideal for separating multiple objects on screens/paper)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 2. Adaptive Intensity Threshold
    mean_val = float(np.mean(blurred))
    if mean_val < 120:
        _, thresh = cv2.threshold(blurred, int(max(10, mean_val + 15)), 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(blurred, int(min(245, mean_val - 15)), 255, cv2.THRESH_BINARY_INV)

    # 3. Saturation Mask (Fruits are colorful/saturated)
    sat_mask = cv2.inRange(hsv, (0, 25, 20), (180, 255, 255))
    sat_closed = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)

    # Combine masks
    combined_mask = cv2.bitwise_or(edges_closed, thresh)
    combined_mask = cv2.bitwise_or(combined_mask, sat_closed)

    # Find ALL contours (RETR_TREE finds inner child contours on screens & trays)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Ignore bounding boxes taking up >90% of full frame (outer screen border)
            if bw > w * 0.90 and bh > h * 0.90:
                continue

            raw_boxes.append((x, y, bw, bh))

    # Filter out parent container boxes (e.g. outer monitor display containing 2 inner fruit cards)
    if len(raw_boxes) > 1:
        filtered_boxes = []
        for i, b1 in enumerate(raw_boxes):
            is_container = False
            for j, b2 in enumerate(raw_boxes):
                if i != j and contains_box(b1, b2):
                    is_container = True
                    break
            if not is_container:
                filtered_boxes.append(b1)
        boxes = filtered_boxes if filtered_boxes else raw_boxes
    else:
        boxes = raw_boxes

    # Sort bounding boxes by area descending and limit to max_objects
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_objects]

    # Fallback: If no contour found above min_area, return central 70% region
    if not boxes:
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        boxes = [(margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)]

    return boxes

def detect_and_classify_multi_objects(bgr_frame, model=None, transform=None, device="cpu", confidence_threshold=60.0, min_area=1500, prob_histories=None):
    """
    Two-Stage Multi-Object Detector:
    1. Uses OpenCV contour localization to find object bounding boxes (ROI crops).
    2. Batches all ROI crops and executes PyTorch classification in a SINGLE forward pass.
    Returns: List of dicts [{ 'box': (x,y,w,h), 'class': str, 'confidence': float, 'is_fresh': bool/None }, ...]
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return []

    if transform is None:
        transform = get_default_transform()

    boxes = extract_object_regions(bgr_frame, min_area=min_area)
    if not boxes or model is None:
        return []

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb_frame.shape

    roi_tensors = []
    valid_boxes = []

    for i, (x, y, bw, bh) in enumerate(boxes):
        # Add 10% padding around bounding box for context
        pad_x = int(bw * 0.1)
        pad_y = int(bh * 0.1)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        crop = rgb_frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil_crop = Image.fromarray(crop)
        roi_tensor = transform(pil_crop)
        roi_tensors.append(roi_tensor)
        valid_boxes.append((x, y, bw, bh))

    if not roi_tensors:
        return []

    # Stack all ROI crops into a single batched tensor for ultra-fast CPU inference!
    batch_tensor = torch.stack(roi_tensors).to(device)

    results = []
    model.eval()
    with torch.no_grad():
        logits = model(batch_tensor)
        probs_batch = F.softmax(logits, dim=1).cpu().numpy()

    for i, (x, y, bw, bh) in enumerate(valid_boxes):
        probs = probs_batch[i]

        # Optional temporal smoothing per box index
        if prob_histories is not None and i in prob_histories:
            prob_histories[i].append(probs)
            avg_probs = np.mean(prob_histories[i], axis=0)
        else:
            avg_probs = probs

        pred_idx = int(np.argmax(avg_probs))
        conf_pct = float(avg_probs[pred_idx] * 100.0)

        if conf_pct < confidence_threshold:
            pred_class = "Object Not Detected"
            is_fresh = None
        else:
            pred_class = CLASS_NAMES[pred_idx]
            is_fresh = "Fresh" in pred_class

        results.append({
            'box': (x, y, bw, bh),
            'class': pred_class,
            'confidence': conf_pct,
            'is_fresh': is_fresh
        })

    return results
