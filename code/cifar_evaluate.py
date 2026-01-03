import torch
import torch.nn as nn
import os

from datasets.cifar import get_cifar_dataloaders # Load CIFAR-10 data
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
_, test_loader = get_cifar_dataloaders(batch_size=64)

# --------------------------------------------------
# Model initialization
# --------------------------------------------------
# Vision Transformer baseline adapted to CIFAR-10
# We take the same architecture that the one we used during training
model = ViTBaseline(
    num_classes=10, # CIFAR-10 has 10 classes
    img_size=32, # CIFAR images are 32x32
    patch_size=8 # Patch size 8x8 -> 16 patches
)
model = model.to(device)

# --------------------------------------------------
# Load trained model checkpoint
# --------------------------------------------------
checkpoint_path = "../experiments/cifar/vit_baseline/model.pth"
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
        outputs = model(images) # logits (B, 10)
        
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
os.makedirs("../experiments/cifar/vit_baseline", exist_ok=True)

results_path = "../experiments/cifar/vit_baseline/results.txt"
with open(results_path, "w") as f:
    f.write(f"Test Loss: {avg_loss:.4f}\n")
    f.write(f"Test Accuracy: {accuracy:.2f}%\n")

print(f"Results saved to {results_path}")
