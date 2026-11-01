import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)
DATASET_DIR = os.path.join(ROOT_DIR, "dataset_official", "fruits-fresh-and-rotten-for-classification", "dataset")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models", "fruit_classifier.pth")
ARTIFACT_DIR = r"C:\Users\Dell\.gemini\antigravity\brain\23529b3d-8d1b-475d-9814-3acd373b3b56"

DISPLAY_NAMES = [
    "Fresh Apple",
    "Fresh Banana",
    "Fresh Orange",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Orange"
]

from src.engine.model import FruitCNN

def run_plots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitCNN(num_classes=6).to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_data = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    all_preds, all_targets, all_features = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = model.extract_features(imgs)
            outputs = model.classifier(feats)
            _, preds = torch.max(outputs, 1)

            all_features.append(feats.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_features = np.vstack(all_features)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_accuracy = np.mean(all_preds == all_targets) * 100.0

    # 1. 2D Classification Plane Plot
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(all_features)

    plt.figure(figsize=(10, 7), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')

    colors = ['#10b981', '#f59e0b', '#38bdf8', '#ef4444', '#b45309', '#0369a1']
    markers = ['o', 's', '^', 'X', 'D', 'v']

    for cls_i in range(6):
        idx = (all_targets == cls_i)
        ax.scatter(
            feats_2d[idx, 0], feats_2d[idx, 1],
            c=colors[cls_i], label=DISPLAY_NAMES[cls_i],
            marker=markers[cls_i], s=80, alpha=0.9, edgecolors='white', linewidths=0.5
        )

    ax.set_title("ESP32-CAM Fruit AI Classification - 2D Feature Representation Plane", color='#38bdf8', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Principal Feature Axis 1 (PCA Projection)", color='#94a3b8', fontsize=11)
    ax.set_ylabel("Principal Feature Axis 2 (PCA Projection)", color='#94a3b8', fontsize=11)
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, linestyle='--', alpha=0.2, color='#475569')

    plt.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', loc='upper right')
    plt.tight_layout()

    FIG_DIR = os.path.join(ROOT_DIR, "docs", "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    out1_paths = [
        os.path.join(FIG_DIR, "classification_2d_plane.png"),
        os.path.join(ARTIFACT_DIR, "classification_2d_plane.png"),
        os.path.join(FIG_DIR, "classification_2d_plane.svg"),
        os.path.join(ARTIFACT_DIR, "classification_2d_plane.svg")
    ]
    for p in out1_paths:
        try:
            plt.savefig(p, dpi=150 if p.endswith('.png') else None, bbox_inches='tight')
        except Exception as e:
            print(f"[Warning] Could not save {p}: {e}")
    plt.close()

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(8, 6), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')
    ax.matshow(cm, cmap='Blues', alpha=0.85)

    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else '#0f172a', fontweight='bold', fontsize=13)

    ax.set_xticklabels([''] + DISPLAY_NAMES, rotation=35, ha='left', color='#94a3b8')
    ax.set_yticklabels([''] + DISPLAY_NAMES, color='#94a3b8')
    ax.set_title(f"Confusion Matrix (Test Accuracy: {test_accuracy:.1f}%)", color='#38bdf8', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    out2_paths = [
        os.path.join(FIG_DIR, "confusion_matrix.png"),
        os.path.join(ARTIFACT_DIR, "confusion_matrix.png"),
        os.path.join(FIG_DIR, "confusion_matrix.svg"),
        os.path.join(ARTIFACT_DIR, "confusion_matrix.svg")
    ]
    for p in out2_paths:
        try:
            plt.savefig(p, dpi=150 if p.endswith('.png') else None, bbox_inches='tight')
        except Exception as e:
            print(f"[Warning] Could not save {p}: {e}")
    plt.close()

    print("Successfully generated and saved PNG and SVG vector plots to both local and artifact directories!")

if __name__ == "__main__":
    run_plots()
