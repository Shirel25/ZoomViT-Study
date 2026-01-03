import torch
import random 
import numpy as np

from datasets.flowers import get_flowers_dataloaders
from models.vit_baseline import ViTBaseline
from utils import compute_importance_map, plot_importance_map

# --------------------------------------------------
# Set random seeds for reproducibility
# --------------------------------------------------
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------
# Device configuration
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------
# # Load Flowers102 test data
# --------------------------------------------------
# We only need a few test images for visualization
_, test_loader = get_flowers_dataloaders(batch_size=1)


# --------------------------------------------------
# Load trained ViT baseline model
# --------------------------------------------------
model = ViTBaseline(
     num_classes=102, # Flowers102 has 102 classes
    img_size=224, # Flowers102 images are 224x224
    patch_size=16 # Patch size 16x16 -> 196 patches
)
model = model.to(device)

checkpoint_path = "../experiments/flowers/vit_baseline/model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()



# --------------------------------------------------
# Compute importance map for a few test images
# --------------------------------------------------
num_images = 30

dataset = test_loader.dataset

indices = random.sample(range(len(dataset)), num_images)

for i, idx in enumerate(indices):
    image, _ = dataset[idx]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        _, tokens = model(image, return_attention=True)

    importance_maps = compute_importance_map(tokens, grid_size=14)
    print(f"Image {i+1}")

    plot_importance_map(image[0], importance_maps[0])
