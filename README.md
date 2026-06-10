# Image Forgery Detection and Localization

## Overview

This project aims to detect manipulated images and localize forged regions using a two-stage framework.

The system first identifies suspicious regions through a U-Net-based segmentation model and then performs final forgery classification using a LightGBM classifier.

---

## Project Structure

```text
forensic_project/

├── step00_setup.py
├── step01_data_prep.py
├── step02_train_stage1.py
├── step03_heatmap_features.py
├── step04_train_stage2.py
├── step05_evaluate.py
├── utils.py

├── data/
│   └── dataset.csv

├── checkpoints/
│   ├── run0_best.pth
│   └── run1_best.pth

└── results/
```

## Dataset

### DEFACTO Dataset
Synthetic image dataset used for forgery localization.

### COCO Dataset
Authentic image dataset used as negative samples.

### Data Distribution

- Total Images: 30,000
- Train: 70%
- Validation: 20%
- Test: 10%
- Real : Fake = 1 : 1

---

## Methodology

### Stage 1: Forgery Localization (U-Net)

The first stage predicts manipulated regions at the pixel level.

Outputs:

- Forgery Mask
- Localization Heatmap

Objective:

Learn where image manipulation occurs rather than only determining whether manipulation exists.

### Stage 2: Forgery Classification (LightGBM)

The second stage utilizes localization information generated from Stage 1.

Input Features:

- Heatmap Statistics
- Localization Features

Output:

- Real Image
- Forged Image

---

## Pipeline

Data Preparation
↓
U-Net Training
↓
Heatmap Generation
↓
Feature Extraction
↓
LightGBM Training
↓
Final Evaluation

---

## Key Features

- U-Net based forgery localization
- Pixel-level manipulation detection
- Heatmap generation
- Feature extraction
- LightGBM classification
- Group leakage prevention
- Stratified Group K-Fold validation
- Confusion Matrix visualization

---

## Technologies

- Python
- PyTorch
- U-Net
- LightGBM
- OpenCV
- Albumentations
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib

---

## My Contribution

### Stage 1 Development

- Data preprocessing pipeline
- U-Net training pipeline implementation
- Forgery mask generation
- Heatmap generation
- Localization feature extraction
- Model evaluation and visualization

---

## Future Work

- U-Net++ architecture exploration
- Attention-based localization models
- Explainable AI (XAI) integration
- Real-world image forgery benchmark evaluation
