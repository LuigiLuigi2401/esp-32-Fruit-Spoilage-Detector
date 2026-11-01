"""
Multi-Method Model Benchmark Suite for ESP32 Fruit Spoilage Detection
Trains and benchmarks MobileNetV3-Large, EfficientNet-B0, ResNet-50, and Ensemble.
Evaluates on both Clean Kaggle Test Set and Simulated Noisy ESP32 Camera Stream Benchmark.
"""

import os
import sys
import time
import io
from PIL import Image, ImageFilter
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset_official", "fruits-fresh-and-rotten-for-classification")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "fruit_classifier.pth")
ARTIFACT_DIR = r"C:\Users\Dell\.gemini\antigravity\brain\23529b3d-8d1b-475d-9814-3acd373b3b56"

CLASS_NAMES = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]
DISPLAY_NAMES = ["Fresh Apple", "Fresh Banana", "Fresh Orange", "Rotten Apple", "Rotten Banana", "Rotten Orange"]

# --- Custom Real-World ESP32 Camera Noise Simulator ---
class ESP32CameraNoiseSim:
    """Simulates low-quality ESP32-CAM JPEG compression artifacts & blur."""
    def __init__(self, quality_range=(30, 70)):
        self.quality_range = quality_range

    def __call__(self, img):
        # 1. Random JPEG compression quality
        q = np.random.randint(self.quality_range[0], self.quality_range[1])
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        img = Image.open(buf)

        # 2. Random slight lens blur
        if np.random.rand() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5)))
        return img

# --- Preprocessing & Augmentations ---
train_augmentation = transforms.Compose([
    transforms.Resize((150, 150)),
    ESP32CameraNoiseSim(quality_range=(35, 75)),
    transforms.RandomResizedCrop(150, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

clean_test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

noisy_test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    ESP32CameraNoiseSim(quality_range=(30, 50)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model_architecture(model_type, num_classes=6):
    if model_type == "mobilenet_v3":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        m = models.mobilenet_v3_large(weights=weights)
        in_f = m.classifier[0].in_features
        m.classifier = nn.Sequential(
            nn.Linear(in_f, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        return m, "MobileNetV3-Large"

    elif model_type == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        m = models.efficientnet_b0(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_f, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        return m, "EfficientNet-B0"

    elif model_type == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        m = models.resnet50(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        return m, "ResNet-50"

def train_and_eval_model(model_type, train_loader, clean_test_loader, noisy_test_loader, epochs=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = get_model_architecture(model_type)
    model = model.to(device)

    # Freeze initial layers, fine-tune top representation blocks
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    print(f"\n=======================================================")
    print(f" Training {model_name} with ESP32 Real-World Augmentations")
    print(f"=======================================================")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_total += labels.size(0)

        tr_acc = (tr_correct / tr_total) * 100.0
        print(f"[{model_name}] Epoch [{ep:02d}/{epochs}] Loss: {tr_loss/tr_total:.4f} | Train Acc: {tr_acc:.2f}% | Time: {time.time()-t0:.1f}s")

    # Evaluate Clean & Noisy Test Sets
    model.eval()
    def evaluate(loader):
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                out = model(imgs)
                pr = torch.softmax(out, dim=1)
                probs.append(pr.cpu().numpy())
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(labels.numpy())
        return np.array(preds), np.array(targets), np.vstack(probs)

    clean_preds, targets, clean_probs = evaluate(clean_test_loader)
    noisy_preds, _, noisy_probs = evaluate(noisy_test_loader)

    clean_acc = accuracy_score(targets, clean_preds) * 100.0
    noisy_acc = accuracy_score(targets, noisy_preds) * 100.0

    print(f"[{model_name}] Clean Test Acc: {clean_acc:.2f}% | ESP32 Noisy Stream Acc: {noisy_acc:.2f}%")
    return model, clean_acc, noisy_acc, clean_probs, noisy_probs, targets

def main():
    train_dir = os.path.join(DATASET_ROOT, "dataset", "train")
    test_dir = os.path.join(DATASET_ROOT, "dataset", "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_augmentation)
    clean_test_ds = datasets.ImageFolder(test_dir, transform=clean_test_transform)
    noisy_test_ds = datasets.ImageFolder(test_dir, transform=noisy_test_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    clean_test_loader = DataLoader(clean_test_ds, batch_size=32, shuffle=False, num_workers=0)
    noisy_test_loader = DataLoader(noisy_test_ds, batch_size=32, shuffle=False, num_workers=0)

    results = {}
    all_models = {}
    probs_dict = {}

    model_keys = ["mobilenet_v3", "efficientnet_b0", "resnet50"]

    for key in model_keys:
        model, clean_acc, noisy_acc, c_probs, n_probs, targets = train_and_eval_model(
            key, train_loader, clean_test_loader, noisy_test_loader, epochs=5
        )
        results[key] = {"clean_acc": clean_acc, "noisy_acc": noisy_acc}
        all_models[key] = model
        probs_dict[key] = {"clean": c_probs, "noisy": n_probs}

    # --- Method 4: Multi-Model Ensemble ---
    print(f"\n=======================================================")
    print(f" Evaluating Multi-Model Soft-Voting Ensemble")
    print(f"=======================================================")

    ens_clean_probs = sum([probs_dict[k]["clean"] for k in model_keys]) / len(model_keys)
    ens_noisy_probs = sum([probs_dict[k]["noisy"] for k in model_keys]) / len(model_keys)

    ens_clean_preds = ens_clean_probs.argmax(axis=1)
    ens_noisy_preds = ens_noisy_probs.argmax(axis=1)

    ens_clean_acc = accuracy_score(targets, ens_clean_preds) * 100.0
    ens_noisy_acc = accuracy_score(targets, ens_noisy_preds) * 100.0

    results["ensemble"] = {"clean_acc": ens_clean_acc, "noisy_acc": ens_noisy_acc}

    print(f"\n=== SUMMARY BENCHMARK RESULTS ===")
    print(f"{'Method':<20} | {'Clean Kaggle Acc':<18} | {'ESP32 Noisy Stream Acc':<22}")
    print("-" * 68)
    for k, name in [("mobilenet_v3", "MobileNetV3-Large"), ("efficientnet_b0", "EfficientNet-B0"), ("resnet50", "ResNet-50"), ("ensemble", "Multi-Model Ensemble")]:
        print(f"{name:<20} | {results[k]['clean_acc']:>16.2f}% | {results[k]['noisy_acc']:>20.2f}%")

    # Select overall best performing model for deployment
    best_key = max(results.keys(), key=lambda x: results[x]["noisy_acc"])
    print(f"\n[BEST MODEL SELECTED FOR DEPLOYMENT] Best Real-World Architecture: {best_key.upper()} (Noisy Stream Acc: {results[best_key]['noisy_acc']:.2f}%)")

    # Save weights of best single model or MobileNet model for fast inference
    if best_key == "ensemble":
        top_single = max([k for k in model_keys], key=lambda x: results[x]["noisy_acc"])
        torch.save(all_models[top_single].state_dict(), MODEL_SAVE_PATH)
    else:
        torch.save(all_models[best_key].state_dict(), MODEL_SAVE_PATH)

    print(f"[SUCCESS] Saved deployed model weights to '{MODEL_SAVE_PATH}'")

    # --- Plot Confusion Matrix for Noisy Benchmark ---
    best_preds = ens_noisy_preds if best_key == "ensemble" else probs_dict[best_key]["noisy"].argmax(axis=1)
    cm = confusion_matrix(targets, best_preds)

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
    ax.set_title(f"Confusion Matrix - ESP32 Noisy Camera Benchmark (Acc: {results[best_key]['noisy_acc']:.1f}%)", color='#38bdf8', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()

    plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
    plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    print("[SUCCESS] Benchmark evaluation complete and diagnostic artifacts generated!")

if __name__ == "__main__":
    main()
