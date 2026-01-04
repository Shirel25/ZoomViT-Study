import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from datasets.flowers import get_flowers_dataloaders
from models.vit_baseline import ViTBaseline
from utils import *

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Load Flowers102 dataset
# --------------------------------------------------
_, test_loader = get_flowers_dataloaders(batch_size=1)
dataset = test_loader.dataset

# --------------------------------------------------
# Load trained ViT baseline
# --------------------------------------------------
model = ViTBaseline(
    num_classes=102, # Flowers102 has 102 classes
    img_size=224, # image size 224x224
    patch_size=16 # patch size 16x16, nb patches (224/16)^2=14x14=196
).to(device)

checkpoint_path = "../experiments/flowers/vit_baseline/model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --------------------------------------------------
# Forward pass + importance map
# --------------------------------------------------
NUM_IMAGES = 30
indices = random.sample(range(len(dataset)), NUM_IMAGES)

for i, idx in enumerate(indices):
    image, _ = dataset[idx]
    image = image.unsqueeze(0).to(device)
    print(f"Processing image {i+1}/{NUM_IMAGES} (index {idx})")
    
    # ------------------------------
    # Forward pass + importance map
    # ------------------------------
    with torch.no_grad():
        _, tokens = model(image, return_attention=True)

    importance_map = compute_importance_map(tokens, grid_size=14)[0]

    # --------------------------------
    # Build zoom mask and apply zoom
    # --------------------------------
    mask = build_zoom_mask(importance_map, keep_ratio=0.3)
    # zoomed_image = apply_image_zoom(image[0], mask)
    zoomed_image = apply_image_crop_zoom(image[0], importance_map, keep_ratio=0.3, patch_size=16)

    # ----------------
    # Visualization
    # ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    img = image[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    axes[0].imshow(img)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    # Importance map overlay
    axes[1].imshow(img)
    axes[1].imshow(mask.cpu().numpy(), alpha=0.4, cmap="jet")
    axes[1].set_title("Zoom mask (overlay)")
    axes[1].axis("off")

    # Zoomed image
    zoom_img = zoomed_image.permute(1, 2, 0).cpu().numpy()
    zoom_img = (zoom_img - zoom_img.min()) / (zoom_img.max() - zoom_img.min())
    axes[2].imshow(zoom_img)
    axes[2].set_title("Zoomed image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
