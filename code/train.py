import os
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cifar import get_cifar_dataloaders # Load CIFAR-10 data
from models.vit_baseline import ViTBaseline # ViT baseline model

# --------------------------------------------------
# Device configuration (CPU / GPU)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Data loading
# --------------------------------------------------
# Train and test dataloaders return batches of (images, labels)
train_loader, test_loader = get_cifar_dataloaders(batch_size=64)

# --------------------------------------------------
# Model initialization
# --------------------------------------------------
# Vision Transformer baseline adapted to CIFAR-10
model = ViTBaseline(
    num_classes=10, # CIFAR-10 has 10 classes
    img_size=32, # CIFAR images are 32x32
    patch_size=8 # Patch size 8x8 -> 16 patches
)
model = model.to(device)

# --------------------------------------------------
# Loss function and optimizer
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --------------------------------------------------
# Training loop
# --------------------------------------------------
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # Reset gradients from the previous step

        # Forward pass: compute logits
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass + optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print progress every 100 batches
        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Average loss over all batches
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# --------------------------------------------------
# Save trained model
# --------------------------------------------------
# Create directory if it does not exist
os.makedirs("experiments/cifar/vit_baseline", exist_ok=True)

# Save model weights
torch.save(
    model.state_dict(),
    "experiments/cifar/vit_baseline/model.pth"
)

print("Model saved to experiments/cifar/vit_baseline/model.pth")