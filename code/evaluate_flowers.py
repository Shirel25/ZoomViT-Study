import torch
import torch.nn as nn
import os

from datasets.flowers import get_flowers_dataloaders # Load Flowers102 data
from models.vit_baseline import ViTBaseline # ViT baseline model

# --------------------------------------------------
# Device configuration (CPU / GPU)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Data loading (test set only)
# --------------------------------------------------
# We only need the test dataloader for evaluation
_, test_loader = get_flowers_dataloaders(batch_size=32)

# --------------------------------------------------
# Model initialization
# --------------------------------------------------
# Vision Transformer baseline adapted to Flowers102
# We take the same architecture that the one we used during training
model = ViTBaseline(
    num_classes=102, # Flowers102 has 102 classes
    img_size=224, # Flowers102 images are 224x224
    patch_size=16 # Patch size 16x16 -> 196 patches
)
model = model.to(device)

# --------------------------------------------------
# Load trained model checkpoint
# --------------------------------------------------
checkpoint_path = "../experiments/flowers/vit_baseline/model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# --------------------------------------------------
# Evaluation mode
# --------------------------------------------------
model.eval()

# --------------------------------------------------
# Loss function 
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()

# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------
total_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images) # logits (B, 102)
        
        # Compute loss for the current batch
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # predicted class labels
        _, predicted = torch.max(outputs, dim=1) 
        
        # Update accuracy counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# --------------------------------------------------
# Final metrics computation
# --------------------------------------------------
avg_loss = total_loss / len(test_loader)
accuracy = 100 * correct / total

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")

# --------------------------------------------------
# Save evaluation results
# --------------------------------------------------
# Store the evaluation metrics for later analysis and comparison
os.makedirs("../experiments/flowers/vit_baseline", exist_ok=True)

results_path = "../experiments/flowers/vit_baseline/results.txt"
with open(results_path, "w") as f:
    f.write(f"Test Loss: {avg_loss:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.2f}%\n")

print(f"Results saved to {results_path}")
