import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset_official", "fruits-fresh-and-rotten-for-classification")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fruit_classifier.pth")
DOCS_FIG_DIR = os.path.join(BASE_DIR, "docs", "figures")
ARTIFACT_DIR = r"C:\Users\Dell\.gemini\antigravity\brain\23529b3d-8d1b-475d-9814-3acd373b3b56"

os.makedirs(DOCS_FIG_DIR, exist_ok=True)

DISPLAY_NAMES = [
    "Fresh Apple",
    "Fresh Banana",
    "Fresh Orange",
    "Not A Fruit",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Orange"
]

class FruitModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
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

def generate_figures():
    print("[*] Locating Test Dataset for 7-Class Figure Generation...")
    test_dir = None
    for root, dirs, files in os.walk(DATASET_ROOT):
        if "test" in dirs:
            test_dir = os.path.join(root, "test")
            break

    if not test_dir or not os.path.exists(test_dir):
        print(f"[ERROR] Test directory not found under {DATASET_ROOT}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Loading Trained 7-Class Weights from {MODEL_PATH} ({device})...")

    model = FruitModel(num_classes=7).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_data = datasets.ImageFolder(test_dir, transform=eval_transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    print(f"[*] Running Test Set Evaluation ({len(test_data)} images)...")

    all_preds, all_targets, all_features = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feats = model.extract_features(imgs)
            outputs = model.classifier(feats)
            _, preds = torch.max(outputs, 1)

            all_features.append(feats.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    final_acc = np.mean(all_preds == all_targets) * 100.0
    print(f"[+] Test Accuracy: {final_acc:.2f}%")

    # --- Plot 1: 2D Feature PCA Plane ---
    print("[*] Generating 2D Feature Representation Plane (PCA)...")
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(all_features[:1000])
    targets_subset = all_targets[:1000]

    plt.figure(figsize=(10, 7), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')

    colors = ['#10b981', '#f59e0b', '#38bdf8', '#a855f7', '#ef4444', '#b45309', '#0369a1']
    markers = ['o', 's', '^', '*', 'X', 'D', 'v']

    for cls_i in range(7):
        idx = (targets_subset == cls_i)
        ax.scatter(
            feats_2d[idx, 0], feats_2d[idx, 1],
            c=colors[cls_i], label=DISPLAY_NAMES[cls_i],
            marker=markers[cls_i], s=70, alpha=0.85, edgecolors='white', linewidths=0.5
        )

    ax.set_title(f"7-Class Dataset - 2D Feature Representation Plane (Acc: {final_acc:.1f}%)", color='#38bdf8', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Principal Feature Axis 1 (PCA Projection)", color='#94a3b8')
    ax.set_ylabel("Principal Feature Axis 2 (PCA Projection)", color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, linestyle='--', alpha=0.2, color='#475569')

    plt.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', loc='upper right')
    plt.tight_layout()

    plt.savefig(os.path.join(DOCS_FIG_DIR, "classification_2d_plane.png"), dpi=150)
    plt.savefig(os.path.join(DOCS_FIG_DIR, "classification_2d_plane.svg"))
    plt.savefig(os.path.join(BASE_DIR, "classification_2d_plane.png"), dpi=150)
    if os.path.exists(ARTIFACT_DIR):
        plt.savefig(os.path.join(ARTIFACT_DIR, "classification_2d_plane.png"), dpi=150)
        plt.savefig(os.path.join(ARTIFACT_DIR, "classification_2d_plane.svg"))
    plt.close()

    # --- Plot 2: Confusion Matrix ---
    print("[*] Generating 7-Class Confusion Matrix SVG & PNG...")
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(9, 7), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')
    ax.matshow(cm, cmap='Blues', alpha=0.85)

    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else '#0f172a', fontweight='bold', fontsize=11)

    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=35, ha='left', color='#94a3b8')
    ax.set_yticklabels(DISPLAY_NAMES, color='#94a3b8')
    ax.set_title(f"Confusion Matrix - 7-Class Test Evaluation (Acc: {final_acc:.1f}%)", color='#38bdf8', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()

    plt.savefig(os.path.join(DOCS_FIG_DIR, "confusion_matrix.png"), dpi=150)
    plt.savefig(os.path.join(DOCS_FIG_DIR, "confusion_matrix.svg"))
    plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
    if os.path.exists(ARTIFACT_DIR):
        plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=150)
        plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.svg"))
    plt.close()

    print("[SUCCESS] Successfully generated and saved 7-class SVG and PNG figures!")

if __name__ == "__main__":
    generate_figures()
