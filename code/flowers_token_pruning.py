import torch
import random
import numpy as np
import torch.nn.functional as F

from datasets.flowers import get_flowers_dataloaders
from models.vit_baseline import ViTBaseline

# ====================================================
# Results file
# ====================================================
results_file = "../experiments/flowers/vit_baseline/pruning_results.txt"

with open(results_file, "w") as f:
    f.write("Token-level pruning results (keep_ratio = 0.3)\n")
    f.write("------------------------------------------------\n")

# ====================================================
# Token Pruning
# ====================================================
def prune_tokens(tokens, keep_ratio=0.3):
    cls_token = tokens[:, :1, :]      # (1, 1, D)
    patch_tokens = tokens[:, 1:, :]   # (1, N, D)

    importance = torch.norm(patch_tokens, dim=-1)  # (1, N)

    N = patch_tokens.shape[1]
    k = max(1, int(N * keep_ratio))

    idx = torch.topk(importance, k=k, dim=1).indices
    idx = idx.unsqueeze(-1).expand(-1, -1, patch_tokens.size(-1))

    pruned_patches = torch.gather(patch_tokens, dim=1, index=idx)
    pruned_tokens = torch.cat([cls_token, pruned_patches], dim=1)

    return pruned_tokens


# ====================================================
# Reproducibility
# ====================================================
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ====================================================
# Dataset
# ====================================================
_, test_loader = get_flowers_dataloaders(batch_size=1)
dataset = test_loader.dataset

# ====================================================
# Model
# ====================================================
model = ViTBaseline(
    num_classes=102,
    img_size=224,
    patch_size=16
).to(device)

model.load_state_dict(
    torch.load("../experiments/flowers/vit_baseline/model.pth", map_location=device)
)
model.eval()

# ====================================================
# Fixed test images
# ====================================================
TEST_INDICES = {
    "good": 1143,
    "inverted": 5238,
    "diffuse": 204
}

# ====================================================
# Evaluation loop
# ====================================================
for name, idx in TEST_INDICES.items():
    image, label = dataset[idx]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Baseline prediction
        logits = model(image)
        pred_before = logits.argmax(dim=1).item()
        probs_before = F.softmax(logits, dim=1)
        conf_before = probs_before[0, pred_before].item()

        # Token extraction
        tokens = model.model.forward_features(image)

        # Token pruning
        pruned_tokens = prune_tokens(tokens, keep_ratio=0.3)

        # Classification from pruned tokens
        cls_token = pruned_tokens[:, 0]
        logits_pruned = model.forward_with_pruning(image,keep_ratio=0.3,prune_layer=6)

        pred_after = logits_pruned.argmax(dim=1).item()
        probs_after = F.softmax(logits_pruned, dim=1)
        conf_after = probs_after[0, pred_after].item()

        # Log results
        with open(results_file, "a") as f:
            f.write(f"\nImage type: {name}\n")
            f.write(f"Prediction BEFORE pruning (class): {pred_before}\n")
            f.write(f"Confidence BEFORE pruning: {conf_before:.4f}\n\n")
            f.write(f"Prediction AFTER pruning (class):  {pred_after}\n")
            f.write(f"Confidence AFTER pruning:  {conf_after:.4f}\n\n")



    print(f"\nImage type: {name}")
    print(f"Prediction BEFORE pruning (class): {pred_before}")
    print(f"Confidence BEFORE pruning: {conf_before:.4f}")
    print(f"Prediction AFTER pruning (class):  {pred_after}")
    print(f"Confidence AFTER pruning:  {conf_after:.4f}")
