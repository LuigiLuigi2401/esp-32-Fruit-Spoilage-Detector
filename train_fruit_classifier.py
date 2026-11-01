"""
Train Fruit Fresh & Rotten PyTorch Classifier
Dataset: sriramr/fruits-fresh-and-rottenfor-classification
Pipeline follows Fig. 7 Preprocessing Steps:
  1. Image Resizing: 150 x 150 px
  2. Normalization: [0, 255] -> [0, 1]
  3. Data Augmentation: Random rotation, shift, zoom, flip
  4. Dataset Splitting: 80% Train, 20% Validation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import FruitCNN, CLASS_NAMES

def get_transforms():
    # 1. Training Transforms with Data Augmentation (Rotation, Shift, Zoom, Flip)
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Validation / Inference Transforms (Clean Resize & Normalize)
    val_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),  # Scales [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(data_dir, epochs=15, batch_size=32, lr=0.001, save_path="fruit_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform, val_transform = get_transforms()

    # Load Full Dataset from directory
    print(f"Loading dataset from: {data_dir}")
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images across classes: {full_dataset.classes}")

    # 80% Train / 20% Validation Split (Fig. 7)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply clean validation transforms to validation split
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize Model, Loss, Optimizer
    model = FruitCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0

    print("\n--- Starting Model Training ---")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model to '{save_path}' (Val Acc: {best_acc*100:.2f}%)")

    print(f"\nTraining Complete! Best Validation Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "./dataset/dataset/train"  # Default directory
    
    if os.path.exists(dataset_path):
        train_model(dataset_path)
    else:
        print(f"Dataset path '{dataset_path}' not found.")
        print("Usage: python train_fruit_classifier.py <path_to_fruits_dataset_folder>")
