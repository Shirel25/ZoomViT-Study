import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# --------------------------------------------------
# Compute patch importance map from ViT tokens
# --------------------------------------------------
def compute_importance_map(tokens, grid_size):
    """Compute a spatial importance map from ViT token embeddings"""
    # --------------------------------------------------
    # Remove CLS token
    # --------------------------------------------------
    # tokens_patches shape: (B, N, D)
    tokens_patches = tokens[:, 1:, :]

    # --------------------------------------------------
    # Compute importance score per patch : patch -> scalar
    # --------------------------------------------------
    # Use L2 norm over embedding dimension
    # importance shape: (B, N)
    importance = torch.norm(tokens_patches, dim=-1)

    # --------------------------------------------------
    # Reshape to spatial grid
    # --------------------------------------------------
    # (B, N) -> (B, grid_size, grid_size)
    importance_maps = importance.view(
        importance.shape[0],
        grid_size,
        grid_size
    )

    return importance_maps


# --------------------------------------------------
# Plot image with importance map overlay
# --------------------------------------------------
# def plot_importance_map(image, importance_map):
#     """Plot original image, importance map, and overlay """
#     # --------------------------------------------------
#     # Prepare image
#     # --------------------------------------------------
#     if torch.is_tensor(image):
#         image = image.unsqueeze(0)  # (1, 3, 32, 32)
#         image = F.interpolate(
#             image,
#             size=(128, 128),        # upscale image
#             mode="nearest"
#         )
#         image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()

#     # Normalize to [0,1]
#     image = (image - image.min()) / (image.max() - image.min() + 1e-8)
#     # --------------------------------------------------
#     # Prepare importance map
#     # --------------------------------------------------
#     if torch.is_tensor(importance_map):
#         importance_map = importance_map.unsqueeze(0).unsqueeze(0)  # (1,1,4,4)
#         importance_map = F.interpolate(
#             importance_map,
#             size=(32, 32),
#             mode="nearest"
#         )
#         importance_map = importance_map.squeeze().cpu().numpy()

#     # ---------------------------
#     # Plot
#     # ---------------------------
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     axs[0].imshow(image, interpolation="nearest")
#     axs[0].set_title("Original image")
#     axs[0].axis("off")

#     axs[1].imshow(importance_map, cmap="jet")
#     axs[1].set_title("Patch importance (upsampled)")
#     axs[1].axis("off")

#     axs[2].imshow(image, interpolation="nearest")
#     axs[2].imshow(importance_map, cmap="jet", alpha=0.4)
#     axs[2].set_title("Overlay")
#     axs[2].axis("off")

#     # Draw patch grid
#     H, W, _ = image.shape
#     patch_size = H // 4  # 4x4 grid

#     for i in range(1, 4):
#         plt.axhline(i * patch_size, color="white", linewidth=1)
#         plt.axvline(i * patch_size, color="white", linewidth=1)

#     plt.tight_layout()
#     plt.show()

def plot_importance_map(image, importance_map, alpha=0.4):
    """
    Plot original image, importance map, and transparent overlay.

    Args:
        image (Tensor): shape (3, H, W)
        importance_map (Tensor): shape (G, G) e.g. 14x14
        alpha (float): transparency of overlay
    """

    # -----------------------------
    # Prepare image
    # -----------------------------
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    H, W, _ = img.shape

    # -----------------------------
    # Prepare importance map
    # -----------------------------
    imp = importance_map.unsqueeze(0).unsqueeze(0)  # (1,1,G,G)

    # Upsample to image resolution
    imp_up = F.interpolate(
        imp,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    imp_up = imp_up.squeeze().cpu().numpy()

    # Normalize for visualization
    imp_up = (imp_up - imp_up.min()) / (imp_up.max() - imp_up.min() + 1e-8)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    # Importance map only
    axes[1].imshow(imp_up, cmap="jet")
    axes[1].set_title("Patch importance (upsampled)")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(imp_up, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
