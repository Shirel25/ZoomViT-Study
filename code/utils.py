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

# --- For Cifar --- 
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

# --------------------------------------------------
# Plot image with importance map overlay
# --------------------------------------------------
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


# --------------------------------------------------
# Build zoom mask from importance map
# --------------------------------------------------
def build_zoom_mask(importance_map, keep_ratio=0.3, image_size=224):
    """
    Build a binary zoom mask from a patch-level importance map.
    Args:
        importance_map (Tensor): shape (G, G), e.g. 14x14
        keep_ratio (float): fraction of patches to keep (e.g. 0.3)
        image_size (int): target image size (e.g. 224)
    Returns:
        mask_up (Tensor): shape (image_size, image_size), values in {0,1}
        1 -> keep, 0 -> discard
    """
    G = importance_map.shape[0]

    # Flatten importance values
    flat = importance_map.flatten()

    # Compute threshold (top-k)
    k = int(keep_ratio * flat.numel())
    threshold = torch.topk(flat, k).values.min()

    # Binary mask at patch level
    mask = (importance_map >= threshold).float()  # (G, G)

    # Upsample to image resolution
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,G,G)
    mask_up = F.interpolate(
        mask,
        size=(image_size, image_size),
        mode="nearest"
    )

    return mask_up.squeeze()  # (H, W)


# --------------------------------------------------
# Apply zoom effect using binary mask
# --------------------------------------------------
def apply_image_zoom(image, mask, background_factor=0.3):
    """
    Apply a zoom effect on an image using a binary mask.
    Args:
        image (Tensor): shape (3, H, W)
        mask (Tensor): shape (H, W), values in {0,1}
        background_factor (float): attenuation factor for non-important regions
    """
    # Ensure mask has same number of channels
    mask = mask.unsqueeze(0)  # (1, H, W)

    # Zoomed image: keep important regions, attenuate background
    zoomed_image = image * mask + image * (1 - mask) * background_factor

    return zoomed_image # shape (3, H, W)


def apply_image_crop_zoom(
    image,
    importance_map,
    keep_ratio=0.15,
    patch_size=16,
    min_crop_size=64
):
    """
    True spatial zoom based on importance map.
    image: Tensor (3, H, W)
    importance_map: Tensor (G, G)
    """

    C, H, W = image.shape
    G = importance_map.shape[0]

    # 1. Flatten importance
    scores = importance_map.flatten()
    k = max(1, int(len(scores) * keep_ratio))

    threshold = torch.topk(scores, k).values.min()
    mask = importance_map >= threshold  # (G, G)

    # 2. Bounding box in patch space
    ys, xs = torch.where(mask)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # 3. Convert to pixel coordinates
    top = y_min * patch_size
    left = x_min * patch_size
    bottom = (y_max + 1) * patch_size
    right = (x_max + 1) * patch_size

    # 4. Enforce minimum crop size (important!)
    crop_h = bottom - top
    crop_w = right - left

    if crop_h < min_crop_size:
        pad = (min_crop_size - crop_h) // 2
        top = max(0, top - pad)
        bottom = min(H, bottom + pad)

    if crop_w < min_crop_size:
        pad = (min_crop_size - crop_w) // 2
        left = max(0, left - pad)
        right = min(W, right + pad)

    # 5. Crop
    cropped = image[:, top:bottom, left:right]

    # 6. Resize back to original size
    zoomed = F.interpolate(
        cropped.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    return zoomed
