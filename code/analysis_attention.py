import torch

from datasets.cifar import get_cifar_dataloaders
from models.vit_baseline import ViTBaseline
from utils import compute_importance_map, plot_importance_map


# --------------------------------------------------
# Device configuration
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# Load CIFAR-10 test data
# --------------------------------------------------
# We only need a few test images for visualization
_, test_loader = get_cifar_dataloaders(batch_size=1)


# --------------------------------------------------
# Load trained ViT baseline model
# --------------------------------------------------
model = ViTBaseline(
    num_classes=10,
    img_size=32,
    patch_size=8
)
model = model.to(device)

checkpoint_path = "../experiments/cifar/vit_baseline/model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()


# --------------------------------------------------
# Get one test image
# --------------------------------------------------
images, labels = next(iter(test_loader))
images = images.to(device)


# --------------------------------------------------
# Forward pass with attention extraction
# --------------------------------------------------
with torch.no_grad():
    logits, tokens = model(images, return_attention=True)


# --------------------------------------------------
# Compute importance map
# --------------------------------------------------
# For CIFAR-10 with 8x8 patches -> 4x4 grid
importance_maps = compute_importance_map(tokens, grid_size=4)


# --------------------------------------------------
# Plot result
# --------------------------------------------------
num_images = 5
classes = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)

    with torch.no_grad():
        logits, tokens = model(images, return_attention=True)

    importance_maps = compute_importance_map(tokens, grid_size=4)

    print(f"Image {i+1}")
    print("True label:", classes[labels[0].item()])

    plot_importance_map(images[0], importance_maps[0])

    if i + 1 >= num_images:
        break
