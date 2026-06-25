# SnakeBiteAI: Deep Geospatial-Visual Fusion Framework for Fine-Grained Snake Species Classification

SnakeBiteAI is an automated clinical tool for identifying Indonesian snake species to aid in snakebite emergencies. The project introduces a deep geospatial-visual fusion framework that merges high-resolution visual embeddings from ConvNeXt-Large with a mathematically grounded Continuous Kernel Density Estimation (KDE) soft geospatial prior.

This project secured **First Place at the Harvard Health Systems Innovation Lab (HSIL) Hackathon 2026** (Bandung Regional Hub). The experimental code and methodology in this repository accompany our scientific paper: **"Deep Geospatial-Visual Fusion Framework for Fine-Grained Snake Species Classification in Indonesia."**

---

## Features

- **Object Detection (YOLO26l):** Efficient isolation of snake regions from complex, highly cluttered rainforest backgrounds.
- **Visual Backbone (ConvNeXt-Large):** Visual feature extraction at a high 384×384 resolution for capturing fine-grained subordinate attributes.
- **Soft KDE Geospatial Prior:** Location-aware classification using a Continuous KDE fusion layer, which penalizes geographically implausible taxa dynamically.
- **Strong Performance:** Achieves a **Macro F1-score of 0.8175** on a dataset of 179 Indonesian snake species, outperforming standalone vision models and external baselines such as DINOv2 and BioCLIP 2.

---

## Repository Structure

```text
├── data/
│   └── README.md                    # Instructions for dataset download
├── notebooks/
│   ├── 01_training_pipeline.ipynb   # Complete training workflow (backbone, KDE, XGBoost)
│   └── 02_ablation_study.ipynb      # Comparative evaluation against DINOv2, BioCLIP 2
├── paper.docx                       # Project documentation and research findings
└── README.md                        # This file
```

## Environment Setup & Installation

All experiments are executed in a Python 3 environment. We recommend using a GPU-accelerated environment (e.g., Kaggle, Google Colab, or local with an NVIDIA GPU).

Dependencies:

```bash
pip install torch torchvision
pip install timm xgboost albumentations pandas numpy scikit-learn opencv-python tqdm
pip install open_clip_torch  # Required for the BioCLIP 2 baseline evaluation
```

## Dataset Requirements

The dataset contains 12,271 iNaturalist snake observations spanning 179 recognized Indonesian species, divided into training (9,823 images) and testing (2,448 images) subsets.

Because of size constraints, the images are not hosted directly in this repository. To run the notebooks, please download the datasets below and adjust the configuration paths (CFG) within the notebooks:

- **YOLO-Cropped Visual Dataset:** [Kaggle: snake-yolo-cropped](https://www.kaggle.com/datasets/muhffikkri/snake-yolo-cropped)
- **Pre-trained Weights:** Provided within the snake-yolo-cropped Kaggle dataset.
- **Snake Coordinate Dataset (Spatial Data):** [Kaggle: snake-coordinate-dataset](https://www.kaggle.com/datasets/muhffikkri/snake-coordinate-dataset)

Ensure your directory structure matches the CFG paths in `01_training_pipeline.ipynb`.

## Tutorial: Running the Pipeline

### Step 1: Training and Fusion Pipeline

The primary training workflow is documented in `notebooks/01_training_pipeline.ipynb`. This notebook walks you through:

1. **Visual Feature Extraction:** Passes the YOLO-cropped images through a pre-trained ConvNeXt-Large backbone.
2. **KDE Geospatial Prior Modeling:** Fits Kernel Density Estimation on species-specific coordinates.
3. **Classification:** Uses XGBoost with 5-Fold Stratified Cross-Validation on visual features.
4. **Multimodal Fusion:** Fuses visual predictions with KDE geospatial likelihoods to generate the final classification.

**Usage:** Simply open the notebook in Jupyter/Kaggle and execute the cells sequentially. Ensure that `CFG.TRAIN_CROP_DIR`, `CFG.TEST_CROP_DIR`, and `CFG.GEO_CSV` correctly point to your downloaded dataset.

### Step 2: Ablation Study & Baselines

The `notebooks/02_ablation_study.ipynb` reproduces the ablation experiments detailed in our paper. It evaluates our proposed model against:

- A frozen ImageNet baseline (ConvNeXt-Large without fine-tuning).
- The framework without advanced augmentations (RandAugment & CutMix).
- The framework without XGBoost class balancing.
- The framework without Geospatial Fusion (Visual-Only Baseline).
- External Vision-Language Models: DINOv2 and BioCLIP 2.

**Usage:** After establishing the data paths, run all cells to reproduce the ablation results table. 

**Note:** Training the external foundation models (DINOv2 and BioCLIP 2) requires significant GPU memory (16GB+ VRAM).

## Key Results

| Model / Configuration | Macro F1-Score | Note |
| :--- | :--- | :--- |
| **Full Proposed Model (Visual + Geo KDE)** | **0.8175** | **Proposed Fusion** |
| Baseline: DINOv2 ViT-L/14 (Fine-tuned) | 0.8089 | External Baseline |
| Geo-Aware Alt: SINR-style Soft Geo Prior | 0.8065 | Alternative Geo-Fusion |
| Baseline: BioCLIP 2 (Fine-tuned) | 0.7074 | External Baseline |
| w/o CutMix & RandAugment | 0.6381 | Ablation |
| w/o Geo Fusion (Visual-Only Baseline) | 0.6305 | Ablation |
| w/o XGBoost Class Balancing | 0.6162 | Ablation |
| Baseline: Frozen ConvNeXt-L (ImageNet) | 0.5240 | Frozen Baseline |

Incorporating the continuous soft KDE geospatial prior improves performance by **+0.1871 (+29.6% relative improvement)** over the visual-only baseline.
