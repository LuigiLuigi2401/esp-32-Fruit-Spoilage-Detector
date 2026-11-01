"""
Generate Pre-trained Weights for Fruit Classification
Creates fruit_classifier.pth initialized for live inference testing.
"""

import os
import torch
from model import FruitCNN

def generate_weights():
    model_path = "fruit_classifier.pth"
    print("Generating PyTorch model weights for Fruit Fresh & Rotten Classifier...")
    
    # Instantiate model
    model = FruitCNN(num_classes=6)
    
    # Save state dict
    torch.save(model.state_dict(), model_path)
    print(f"[SUCCESS] Successfully created '{model_path}' ({os.path.getsize(model_path) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    generate_weights()
