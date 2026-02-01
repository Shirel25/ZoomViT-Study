import os
import matplotlib.pyplot as plt
import numpy as np
import re

# --------------------------------------------------
# Paths
# --------------------------------------------------
results_file = "experiments/flowers/vit_baseline/pruning_results.txt"
output_dir = "experiments/flowers/vit_baseline"
output_path = os.path.join(output_dir, "pruning_confidence_plot.png")

# --------------------------------------------------
# Create output directory if needed
# --------------------------------------------------
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Containers for parsed results
# --------------------------------------------------
attention_types = []
confidence_before = []
confidence_after = []

current_type = None

# --------------------------------------------------
# Parse results file
# --------------------------------------------------
with open(results_file, "r") as f:
    for line in f:
        line = line.strip()

        # Detect attention type
        if line.startswith("Image type:"):
            current_type = line.split(":")[1].strip()
            attention_types.append(current_type.capitalize())

        # Confidence before pruning
        elif line.startswith("Confidence BEFORE pruning"):
            value = float(re.findall(r"\d+\.\d+", line)[0])
            confidence_before.append(value)

        # Confidence after pruning
        elif line.startswith("Confidence AFTER pruning"):
            value = float(re.findall(r"\d+\.\d+", line)[0])
            confidence_after.append(value)

# --------------------------------------------------
# Sanity check
# --------------------------------------------------
assert len(attention_types) == len(confidence_before) == len(confidence_after), \
    "Mismatch in parsed pruning results"

# --------------------------------------------------
# Prepare bar positions
# --------------------------------------------------
x = np.arange(len(attention_types))
width = 0.35

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(8, 5))

# Custom colors (soft pastel)
before_color = "#8ECAE6"  # light blue
after_color  = "#F4A6C1"  # light pink

plt.bar(
    x - width / 2,
    confidence_before,
    width,
    label="Before pruning",
    color=before_color
)

plt.bar(
    x + width / 2,
    confidence_after,
    width,
    label="After pruning",
    color=after_color
)

# --------------------------------------------------
# Formatting
# --------------------------------------------------
plt.ylabel("Prediction confidence", fontsize=12)
plt.xlabel("Attention regime", fontsize=12)
plt.title("Effect of token pruning on prediction confidence",
            fontsize=14,
            pad=10)
plt.xticks(x, attention_types)
plt.legend(frameon=False)

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# --------------------------------------------------
# Save figure
# --------------------------------------------------
plt.savefig(output_path, dpi=300)
print(f"Figure saved to {output_path}")

# --------------------------------------------------
# Show figure
# --------------------------------------------------
plt.show()
