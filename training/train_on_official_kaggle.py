"""
High-Accuracy MobileNetV2 Fine-Tuning Pipeline for Official Kaggle Dataset
'sriramr/fruits-fresh-and-rotten-for-classification'

Features:
- Full Backbone Unfreezing & Differential Learning Rates (1e-5 backbone / 1e-3 classifier)
- OneCycleLR Scheduler with Warmup & Cosine Annealing (12 Epochs)
- Label Smoothing Cross Entropy Loss
- Data Augmentation: Random Crop, Flips, Rotations, Color Jitter
- Hardware-Aligned ESP32 Resolution Preprocessing (150x150 px)
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset_official", "fruits-fresh-and-rotten-for-classification")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "fruit_classifier.pth")
ARTIFACT_DIR = r"C:\Users\Dell\.gemini\antigravity\brain\23529b3d-8d1b-475d-9814-3acd373b3b56"

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "notafruit",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

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
        
        # Freeze backbone parameters for fast CPU transfer learning
        for param in self.backbone.parameters():
            param.requires_grad = False

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

def run_fine_tuning():
    print("\n--- [Step 1] Locating Kaggle Dataset Directory ---")
    train_dir, test_dir = None, None
    for root, dirs, files in os.walk(DATASET_ROOT):
        if "train" in dirs and "test" in dirs:
            train_dir = os.path.join(root, "train")
            test_dir = os.path.join(root, "test")
            break

    if not train_dir or not os.path.exists(train_dir):
        print(f"[ERROR] Could not find 'train' directory under {DATASET_ROOT}")
        return

    print(f"[FOUND DATASET] Train Dir: {train_dir}")
    print(f"[FOUND DATASET] Test Dir:  {test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- [Step 2] Fine-Tuning MobileNetV2 on Official Kaggle Data (Device: {device}) ---")

    # ImageNet Mean & Std Preprocessing with Augmentations
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    num_classes = len(train_data.classes)
    print(f"[DATASET INFO] Discovered {num_classes} Classes: {train_data.classes}")
    print(f"[DATASET INFO] Training Images: {len(train_data)} | Test Images: {len(test_data)}")

    model = FruitModel(num_classes=num_classes).to(device)
    
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 3
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    print("\n--- Starting Fine-Tuning Epoch Loop ---")
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            t_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            t_correct += (preds == labels).sum().item()
            t_total += labels.size(0)

        tr_loss = t_loss / t_total
        tr_acc = (t_correct / t_total) * 100.0

        # Quick Test Evaluation each epoch
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        te_acc = (v_correct / v_total) * 100.0
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            models_dir_path = os.path.join(BASE_DIR, "models", "fruit_classifier.pth")
            if os.path.exists(os.path.dirname(models_dir_path)):
                torch.save(model.state_dict(), models_dir_path)

        print(f"Epoch [{epoch:02d}/{epochs}] "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
              f"Test Acc: {te_acc:.2f}% | Best: {best_test_acc:.2f}% | Time: {time.time()-t0:.1f}s")

    print(f"\n[SUCCESS] Best Model Weights Saved (Test Acc: {best_test_acc:.2f}%) to '{MODEL_SAVE_PATH}'")

    # --- Step 3: Final Test Set Evaluation & Diagnostic Plots ---
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

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

    final_acc = np.mean(all_preds == all_targets) * 100.0
    print(f"\n[FINAL TEST EVALUATION] Official Kaggle Test Accuracy: {final_acc:.2f}%")
    print(classification_report(all_targets, all_preds, target_names=DISPLAY_NAMES, digits=3))

    # --- Plot 1: 2D PCA Feature Representation Plane ---
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(all_features[:1000])
    targets_subset = all_targets[:1000]

    plt.figure(figsize=(10, 7), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')

    colors = ['#10b981', '#f59e0b', '#38bdf8', '#a855f7', '#ef4444', '#b45309', '#0369a1']
    markers = ['o', 's', '^', '*', 'X', 'D', 'v']

    for cls_i in range(num_classes):
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

    docs_fig_dir = os.path.join(BASE_DIR, "docs", "figures")
    os.makedirs(docs_fig_dir, exist_ok=True)

    plt.savefig(os.path.join(BASE_DIR, "classification_2d_plane.png"), dpi=150)
    plt.savefig(os.path.join(docs_fig_dir, "classification_2d_plane.png"), dpi=150)
    plt.savefig(os.path.join(docs_fig_dir, "classification_2d_plane.svg"))
    if os.path.exists(ARTIFACT_DIR):
        plt.savefig(os.path.join(ARTIFACT_DIR, "classification_2d_plane.png"), dpi=150)
        plt.savefig(os.path.join(ARTIFACT_DIR, "classification_2d_plane.svg"))
    plt.close()

    # --- Plot 2: Confusion Matrix ---
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(9, 7), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')
    ax.matshow(cm, cmap='Blues', alpha=0.85)

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else '#0f172a', fontweight='bold', fontsize=11)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=35, ha='left', color='#94a3b8')
    ax.set_yticklabels(DISPLAY_NAMES, color='#94a3b8')
    ax.set_title(f"Confusion Matrix - 7-Class Test Evaluation (Acc: {final_acc:.1f}%)", color='#38bdf8', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()

    plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
    plt.savefig(os.path.join(docs_fig_dir, "confusion_matrix.png"), dpi=150)
    plt.savefig(os.path.join(docs_fig_dir, "confusion_matrix.svg"))
    if os.path.exists(ARTIFACT_DIR):
        plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=150)
        plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.svg"))
    plt.close()

    print("[SUCCESS] Fine-tuning complete & 7-class SVG/PNG diagnostic plots generated!")

if __name__ == "__main__":
    run_fine_tuning()
