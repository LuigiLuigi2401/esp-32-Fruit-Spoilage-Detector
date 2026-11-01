"""
Fast Refitting MobileNetV2 Classifier Head on Official Kaggle Dataset
Saves PyTorch weights to fruit_classifier.pth & generates 2D PCA & Confusion Matrix plots.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset_official", "fruits-fresh-and-rotten-for-classification")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "fruit_classifier.pth")
ARTIFACT_DIR = r"C:\Users\Dell\.gemini\antigravity\brain\23529b3d-8d1b-475d-9814-3acd373b3b56"

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

DISPLAY_NAMES = [
    "Fresh Apple",
    "Fresh Banana",
    "Fresh Orange",
    "Rotten Apple",
    "Rotten Banana",
    "Rotten Orange"
]

class FruitModel(nn.Module):
    """MobileNetV2 Deep Vision Architecture for ESP32 Fruit Classification."""
    def __init__(self, num_classes=6):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.backbone = models.mobilenet_v2(weights=weights)
        
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

def run_fast_refit():
    print("--- [Step 1] Locating Dataset ---")
    train_dir = os.path.join(DATASET_ROOT, "dataset", "train")
    test_dir = os.path.join(DATASET_ROOT, "dataset", "test")

    tf = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=tf)
    test_ds = datasets.ImageFolder(test_dir, transform=tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    print(f"Loaded {len(train_ds)} train images, {len(test_ds)} test images.")

    print("\n--- [Step 2] Feature Extraction via MobileNetV2 Backbone ---")
    weights = models.MobileNet_V2_Weights.DEFAULT
    mobilenet = models.mobilenet_v2(weights=weights).eval()

    def extract_all(loader):
        feats, lbls = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                f = mobilenet.features(imgs)
                f = nn.functional.adaptive_avg_pool2d(f, (1, 1))
                f = torch.flatten(f, 1)
                feats.append(f.numpy())
                lbls.extend(labels.numpy())
        return np.vstack(feats), np.array(lbls)

    t0 = time.time()
    X_train, y_train = extract_all(train_loader)
    X_test, y_test = extract_all(test_loader)
    print(f"[SUCCESS] Feature extraction completed in {time.time()-t0:.1f} seconds!")

    print("\n--- [Step 3] Fast Refitting Classifier Head ---")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    t1 = time.time()
    clf.fit(X_train, y_train)
    print(f"[SUCCESS] Classifier refit completed in {time.time()-t1:.2f} seconds!")

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred) * 100.0
    print(f"\n=======================================================")
    print(f"   OFFICIAL KAGGLE TEST ACCURACY: {test_acc:.2f}%")
    print(f"=======================================================\n")
    print(classification_report(y_test, y_pred, target_names=DISPLAY_NAMES, digits=3))

    # --- Step 4: Transfer Refitted Classifier Weights to PyTorch Model ---
    model = FruitModel(num_classes=6)
    
    # Train a quick 3-epoch PyTorch Linear Classifier Head on pre-extracted features
    # to perfectly match PyTorch state_dict format
    class PyTorchHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 6)
            )
        def forward(self, x):
            return self.fc(x)

    head = PyTorchHead()
    opt = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    head_ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    head_loader = torch.utils.data.DataLoader(head_ds, batch_size=128, shuffle=True)

    print("Fitting PyTorch Classifier Head...")
    for ep in range(12):
        head.train()
        for bx, by in head_loader:
            opt.zero_grad()
            out = head(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()

    # Copy PyTorch head into full model architecture
    model.classifier = head.fc
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[SUCCESS] Saved PyTorch weights to '{MODEL_SAVE_PATH}'")

    # Evaluate final PyTorch model
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test, dtype=torch.float32)
        pt_preds = model.classifier(X_te_t).argmax(1).numpy()
    pt_acc = accuracy_score(y_test, pt_preds) * 100.0
    print(f"[PYTORCH MODEL TEST ACCURACY] {pt_acc:.2f}%")

    # --- Step 5: Plot 2D PCA Feature Representation Plane ---
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(X_test[:1000])
    targets_subset = y_test[:1000]

    plt.figure(figsize=(10, 7), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')

    colors = ['#10b981', '#f59e0b', '#38bdf8', '#ef4444', '#b45309', '#0369a1']
    markers = ['o', 's', '^', 'X', 'D', 'v']

    for cls_i in range(6):
        idx = (targets_subset == cls_i)
        ax.scatter(
            feats_2d[idx, 0], feats_2d[idx, 1],
            c=colors[cls_i], label=DISPLAY_NAMES[cls_i],
            marker=markers[cls_i], s=70, alpha=0.85, edgecolors='white', linewidths=0.5
        )

    ax.set_title(f"Official Kaggle Dataset - 2D Feature Representation Plane (Acc: {pt_acc:.1f}%)", color='#38bdf8', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Principal Feature Axis 1 (PCA Projection)", color='#94a3b8')
    ax.set_ylabel("Principal Feature Axis 2 (PCA Projection)", color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, linestyle='--', alpha=0.2, color='#475569')

    plt.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', loc='upper right')
    plt.tight_layout()

    plt.savefig(os.path.join(BASE_DIR, "classification_2d_plane.png"), dpi=150)
    plt.savefig(os.path.join(ARTIFACT_DIR, "classification_2d_plane.png"), dpi=150)
    plt.close()

    # --- Step 6: Plot Confusion Matrix ---
    cm = confusion_matrix(y_test, pt_preds)

    plt.figure(figsize=(8, 6), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')
    ax.matshow(cm, cmap='Blues', alpha=0.85)

    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else '#0f172a', fontweight='bold', fontsize=11)

    ax.set_xticklabels([''] + DISPLAY_NAMES, rotation=35, ha='left', color='#94a3b8')
    ax.set_yticklabels([''] + DISPLAY_NAMES, color='#94a3b8')
    ax.set_title(f"Confusion Matrix - Official Kaggle Test Set (Acc: {pt_acc:.1f}%)", color='#38bdf8', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()

    plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
    plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    print("[SUCCESS] All Kaggle dataset evaluation & plots complete!")

if __name__ == "__main__":
    run_fast_refit()
