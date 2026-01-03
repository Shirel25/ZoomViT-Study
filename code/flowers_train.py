import os
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.flowers import get_flowers_dataloaders # Load Flowers102 data
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
train_loader, test_loader = get_flowers_dataloaders(batch_size=32)

# --------------------------------------------------
# Model initialization
# --------------------------------------------------
# Vision Transformer baseline adapted to Flowers102
model = ViTBaseline(
    num_classes=102, # Flowers102 has 102 classes
    img_size=224, # Flowers102 images are 224x224
    patch_size=16 # Patch size 16x16 -> 196 patches
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
num_epochs = 5 # 3 epochs may be too few for Flowers102

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
os.makedirs("../experiments/flowers/vit_baseline", exist_ok=True)

# Save model weights
torch.save(
    model.state_dict(),
    "../experiments/flowers/vit_baseline/model.pth"
)

print("Model saved to experiments/flowers/vit_baseline/model.pth")