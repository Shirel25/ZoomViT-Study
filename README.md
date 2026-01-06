# ZoomViT – Intent-Guided Adaptive Processing for Vision Transformers

This repository contains a **study and implementation** of the concepts introduced in the paper:

> **Vision Transformers Need Zoomer: Efficient ViT with Visual Intent-Guided Zoom Adapter**  
> *(Anonymous, 2026)*

The goal of this project is to **experimentally validate the core hypothesis of ZoomViT**:  
**Vision Transformers perform better and more efficiently when their visual intent is guided toward class-decisive regions.**

---

## 1. Project Motivation

Standard Vision Transformers (ViTs) process images using **uniform, fixed-size patches**, which often leads to:

- **Redundancy**: Background pixels are processed with the same computational cost as the main subject.
- **Misalignment**: The model’s attention can be *hijacked* by visually salient but class-irrelevant objects  
  (e.g., leaves instead of a flower).

ZoomViT proposes a **bio-inspired mechanism** that simulates *foveal vision*, allocating higher resolution and computational focus to semantically important regions.

---

## 2. Dataset Strategy

To properly study visual intent and spatial importance, two datasets were used:

- **CIFAR-10**  
  Used for initial pipeline validation and fast training of the baseline ViT.

- **Oxford Flowers-102**  
  Used as the **primary dataset** for Zoom and Pruning analysis.  
  Its higher resolution (224×224) enables interpretable spatial importance maps, which are otherwise too coarse on low-resolution datasets such as CIFAR-10.

This dataset choice was critical to meaningfully analyze visual intent alignment.

---

## 3. Implementation Pipeline

### Step 1: Baseline Vision Transformer

A modular Vision Transformer implemented from scratch in **PyTorch**, including:

- Patch Embedding and Positional Encodings  
- 12 Transformer blocks with Multi-Head Self-Attention  
- Supervised training to establish a reference **visual intent baseline**

This baseline serves as the anchor point for all subsequent comparisons.

---

### Step 2: Visual Intent Extraction (Importance Maps)

Instead of reproducing the full Zoomer distillation framework, this project uses **attention-based hooks** as a proxy for visual intent.

By aggregating:
- attention weights,
- gradients,
- and relevance across layers,

we generate **Importance Maps** that highlight **class-decisive regions** responsible for the model’s predictions.

This approximation preserves the *intent-guided philosophy* of ZoomViT while remaining computationally tractable.

---

### Step 3: Adaptive Actions (Zoom & Pruning)

This project primarily validates **Stage 2** of the ZoomViT paper through two adaptive mechanisms:

1. **Image-Level Zoom (Stage 1 Simulation)**  
   Images are dynamically cropped and resized based on the bounding box extracted from importance maps.

2. **Token-Level Pruning (Stage 2 – Architectural Modification)**  
   - The token sequence is pruned **after the 6th Transformer block**.
   - Only the top *X% most important tokens* are retained.
   - The remaining Transformer blocks **recompute global attention exclusively on relevant tokens**.

This is not simple token removal: it forces a **dynamic reorganization of attention**, directly modifying the model’s internal reasoning process.

---

## 4. Key Results & Analysis

The experiments reveal three behaviors described in the original paper:

- **Good Alignment**  
  When the model correctly identifies the subject, pruning background tokens often *increases confidence* by removing **negative tokens**.

- **Inverted Intent**  
  When the model’s visual intent is misaligned (focused on background), pruning reinforces the error, illustrating the **Visual Intent Misalignment** phenomenon.

- **Diffuse Intent**  
  When the model is uncertain, importance maps are scattered. In this case, pruning slightly reduces confidence due to loss of contextual cues.

These results empirically confirm that **pruning is beneficial only when visual intent is correctly aligned**.

---

## 5. Repository Structure

```text
.
├── code/
│   ├── datasets/
│   │   ├── cifar.py
│   │   └── flowers.py
│   │
│   ├── models/
│   │   └── vit_baseline.py
│   │
│   ├── cifar_train.py
│   ├── cifar_evaluate.py
│   ├── cifar_analysis_attention.py
│   │
│   ├── flowers_train.py
│   ├── flowers_evaluate.py
│   ├── flowers_analysis_attention.py
│   ├── flowers_zoom_image.py
│   ├── flowers_token_pruning.py
│   │
│   └── utils.py
│
├── experiments/
│   ├── cifar/
│   │   └── vit_baseline/
│   │       ├── visualizations/
│   │       └── results.txt
│   │
│   └── flowers/
│       └── vit_baseline/
│           ├── 1_visualizations/
│           │   ├── 1_good_attention/
│           │   ├── 2_inverted_attention/
│           │   └── 3_diffuse_attention/
│           │
│           ├── 2_zoom_image/
│           │   ├── 1_good_attention/
│           │   ├── 2_inverted_attention/
│           │   └── 3_diffuse_attention/
│           │
│           ├── 3_image_level_zoom/
│           │   ├── 1_good_attention/
│           │   ├── 2_inverted_attention/
│           │   └── 3_diffuse_attention/
│           │
│           ├── pruning_results.txt
│           └── results.txt
│
├── docs/
│   └── 01_paper_summary.md
│
├── Paper_6623_Vision_Transformers_Need.pdf
├── .gitignore
└── README.md

```

---

## 6. Conclusion
This project bridges the gap between research theory and practical implementation.

By simplifying the ZoomViT adapter while preserving its core principles, we demonstrate that selective attention is not merely an efficiency tool, but a robustness mechanism that helps Vision Transformers filter out misleading information in complex visual scenes.
