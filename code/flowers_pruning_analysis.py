import os
import torch
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets.flowers import get_flowers_dataloaders
from models.vit_baseline import ViTBaseline

# ====================================================
# Settings
# ====================================================
KEEP_RATIOS = [0.1, 0.3, 0.5]
SAVE_DIR = "../experiments/flowers/vit_baseline/pruning_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ====================================================
# Dataset
# ====================================================
_, test_loader = get_flowers_dataloaders(batch_size=32)

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
# Storage
# ====================================================
results = []

# ====================================================
# Evaluation Loop
# ====================================================
for keep_ratio in KEEP_RATIOS:

    correct = 0
    total = 0
    confidences = []
    avg_tokens = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model.forward_with_pruning(
                images,
                keep_ratio=keep_ratio,
                prune_layer=6
            )

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            batch_conf = probs[range(len(preds)), preds]
            confidences.extend(batch_conf.cpu().numpy())

            # Approximate number of tokens kept
            N_original = (224 // 16) ** 2
            kept = int(N_original * keep_ratio)
            avg_tokens.append(kept)

    accuracy = correct / total
    mean_conf = np.mean(confidences)
    mean_tokens = np.mean(avg_tokens)

    results.append((keep_ratio, accuracy, mean_conf, mean_tokens))

    print(f"\nKeep Ratio: {keep_ratio}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Confidence: {mean_conf:.4f}")
    print(f"Avg Tokens: {mean_tokens}")

# ====================================================
# Save results
# ====================================================
with open(os.path.join(SAVE_DIR, "pruning_results.txt"), "w") as f:
    for r in results:
        f.write(
            f"Keep Ratio: {r[0]} | "
            f"Accuracy: {r[1]:.4f} | "
            f"Confidence: {r[2]:.4f} | "
            f"Tokens: {r[3]}\n"
        )

# ====================================================
# Plot Accuracy
# ====================================================
ratios = [r[0] for r in results]
accuracies = [r[1] for r in results]
confidences = [r[2] for r in results]
tokens = [r[3] for r in results]

plt.figure()
plt.plot(ratios, accuracies, marker='o')
plt.xlabel("Keep Ratio")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Keep Ratio")
plt.savefig(os.path.join(SAVE_DIR, "accuracy_vs_ratio.png"))
plt.close()

plt.figure()
plt.plot(ratios, confidences, marker='o')
plt.xlabel("Keep Ratio")
plt.ylabel("Mean Confidence")
plt.title("Confidence vs Keep Ratio")
plt.savefig(os.path.join(SAVE_DIR, "confidence_vs_ratio.png"))
plt.close()

plt.figure()
plt.plot(ratios, tokens, marker='o')
plt.xlabel("Keep Ratio")
plt.ylabel("Avg Tokens")
plt.title("Tokens vs Keep Ratio")
plt.savefig(os.path.join(SAVE_DIR, "tokens_vs_ratio.png"))
plt.close()