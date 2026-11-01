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

def compute_box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)

def contains_box(parent, child):
    """Returns True if parent box contains child box center and is significantly larger."""
    px, py, pw, ph = parent
    cx, cy, cw, ch = child
    
    ccx = cx + cw / 2.0
    ccy = cy + ch / 2.0
    
    return (px <= ccx <= px + pw) and (py <= ccy <= py + ph) and (pw * ph > cw * ch * 1.25)

def is_valid_fruit_aspect_ratio(bw, bh):
    """Filter out non-fruit line artifacts or extreme rectangular bars (wires, taskbars, edges)."""
    ratio = float(bw) / float(bh)
    return 0.25 <= ratio <= 3.8

def extract_object_regions(bgr_frame, min_area=1000, max_objects=8):
    """
    Multi-Candidate Computer Vision Region Proposal Detector:
    1. Generates object candidates across Saturation, Otsu Brightness, and Canny Edge Channels.
    2. Filters out parent container boxes (e.g. monitor screen enclosing fruit cards).
    3. Merges overlapping bounding boxes via Non-Maximum Suppression (NMS IoU > 0.4).
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return []

    h, w, _ = bgr_frame.shape
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    candidate_boxes = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Mask Channel 1: Saturation Mask (Catches colorful fruits: apples, bananas, oranges)
    sat = hsv[:, :, 1]
    _, sat_thresh = cv2.threshold(sat, 25, 255, cv2.THRESH_BINARY)
    
    # Mask Channel 2: Foreground Otsu Binarization (Catches white paper cards / screen cards)
    _, bright_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Mask Channel 3: Canny Edge Gradients
    edges = cv2.Canny(blurred, 30, 130)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    masks = [sat_thresh, bright_otsu, edges_closed]

    for mask in masks:
        m_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(m_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Ignore boxes taking up >85% of full frame (screen display frame)
                if bw > w * 0.85 or bh > h * 0.85:
                    continue
                # Ignore non-fruit geometric shapes (extreme thin lines / long bars)
                if not is_valid_fruit_aspect_ratio(bw, bh):
                    continue
                candidate_boxes.append((x, y, bw, bh))

    if not candidate_boxes:
        return []

    # Filter 1: Discard parent container boxes containing smaller valid child boxes
    filtered_containers = []
    for i, b1 in enumerate(candidate_boxes):
        is_container = False
        for j, b2 in enumerate(candidate_boxes):
            if i != j and contains_box(b1, b2):
                is_container = True
                break
        if not is_container:
            filtered_containers.append(b1)

    if not filtered_containers:
        filtered_containers = candidate_boxes

    # Filter 2: Non-Maximum Suppression (NMS) to merge overlapping boxes (IoU > 0.4)
    filtered_containers = sorted(filtered_containers, key=lambda b: b[2] * b[3], reverse=True)
    nms_boxes = []

    for box in filtered_containers:
        keep = True
        for kept in nms_boxes:
            if compute_box_iou(box, kept) > 0.4:
                keep = False
                break
        if keep:
            nms_boxes.append(box)

    return nms_boxes[:max_objects]

def detect_and_classify_multi_objects(bgr_frame, model=None, transform=None, device="cpu", confidence_threshold=60.0, min_area=1500, prob_histories=None):
    """
    Two-Stage Multi-Object Detector with Out-Of-Distribution Non-Fruit Rejection:
    1. Uses OpenCV contour localization & geometry filters to find object candidates.
    2. Batches all ROI crops and executes PyTorch classification in a SINGLE forward pass.
    3. Filters out low-confidence / low-margin non-fruit predictions.
    Returns: List of dicts [{ 'box': (x,y,w,h), 'class': str, 'confidence': float, 'is_fresh': bool }, ...]
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return []

    if transform is None:
        transform = get_default_transform()

    boxes = extract_object_regions(bgr_frame, min_area=min_area, max_objects=4)
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

        sorted_probs = np.sort(avg_probs)
        pred_idx = int(np.argmax(avg_probs))
        conf_pct = float(avg_probs[pred_idx] * 100.0)
        top_margin = float(sorted_probs[-1] - sorted_probs[-2])

        # Reject Non-Fruit Objects: Require confidence >= threshold AND prediction margin >= 0.12
        if conf_pct >= confidence_threshold and top_margin >= 0.12:
            pred_class = CLASS_NAMES[pred_idx]
            is_fresh = "Fresh" in pred_class

            results.append({
                'box': (x, y, bw, bh),
                'class': pred_class,
                'confidence': conf_pct,
                'is_fresh': is_fresh
            })

    return results

class TemporalObjectTracker:
    """
    Temporal Multi-Object Anti-Jitter Tracker:
    - Smooths bounding box coordinates across consecutive video frames using EMA (alpha=0.35).
    - Preserves object detections across temporary 4-frame camera stream dropouts to stop box flickering!
    """
    def __init__(self, persistence_frames=4, alpha=0.35):
        self.persistence_frames = persistence_frames
        self.alpha = alpha
        self.tracked_objects = []
        self.next_id = 1

    def update(self, new_detections):
        updated_tracked = []
        matched_new_indices = set()

        # Match new detections with existing tracked objects via IoU
        for tracked in self.tracked_objects:
            best_iou = 0.0
            best_match_idx = -1

            for idx, det in enumerate(new_detections):
                if idx in matched_new_indices:
                    continue
                iou = compute_box_iou(tracked['box'], det['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx

            if best_iou >= 0.25 and best_match_idx != -1:
                # Match found! Smooth box coordinates and update prediction attributes
                matched_det = new_detections[best_match_idx]
                matched_new_indices.add(best_match_idx)

                tx, ty, tw, th = tracked['box']
                nx, ny, nw, nh = matched_det['box']

                sx = int(self.alpha * nx + (1 - self.alpha) * tx)
                sy = int(self.alpha * ny + (1 - self.alpha) * ty)
                sw = int(self.alpha * nw + (1 - self.alpha) * tw)
                sh = int(self.alpha * nh + (1 - self.alpha) * th)

                tracked['box'] = (sx, sy, sw, sh)
                tracked['class'] = matched_det['class']
                tracked['confidence'] = matched_det['confidence']
                tracked['is_fresh'] = matched_det['is_fresh']
                tracked['missed_frames'] = 0
                updated_tracked.append(tracked)
            else:
                # Missed in this frame! Keep for persistence grace period to prevent box flicker
                tracked['missed_frames'] += 1
                if tracked['missed_frames'] <= self.persistence_frames:
                    updated_tracked.append(tracked)

        # Add unmatched new detections as new tracked objects
        for idx, det in enumerate(new_detections):
            if idx not in matched_new_indices:
                new_obj = det.copy()
                new_obj['id'] = self.next_id
                new_obj['missed_frames'] = 0
                self.next_id += 1
                updated_tracked.append(new_obj)

        self.tracked_objects = updated_tracked
        return [obj for obj in self.tracked_objects if obj['missed_frames'] <= self.persistence_frames]
