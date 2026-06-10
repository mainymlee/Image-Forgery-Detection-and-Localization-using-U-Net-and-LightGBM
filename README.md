# Image Forgery Localization

## Project Overview

This project aims to localize manipulated regions in composite images using deep learning-based semantic segmentation.

Unlike conventional image classification models that only determine whether an image is manipulated, this project focuses on identifying the exact forged region at the pixel level.

---

## Model Architecture

### Stage 1: Forgery Localization

- Model: U-Net++
- Encoder: EfficientNet-B4
- Pretrained Weights: ImageNet
- Input Size: 512 × 512

### Loss Function

- Binary Cross Entropy (BCE)
- Dice Loss

### Optimizer

- AdamW

---

## Training Strategy

### Validation

- Group-based Hold-out Validation

### Warm-up Training

- Encoder Frozen
- Decoder Training Only

### Fine-tuning

- Encoder Unfrozen
- Differential Learning Rate

### Early Stopping

- Patience = 10

---

## Key Improvements (v2)

- U-Net → U-Net++
- Input Resolution: 256 → 512
- Positive Pixel Weight: 2.0 → 4.0
- Improved localization of small manipulated regions

---

## Evaluation Metrics

- IoU (Intersection over Union)
- Dice Score
- Precision
- Recall
- False Positive Area Ratio

---

## Tech Stack

- Python
- PyTorch
- segmentation-models-pytorch
- OpenCV
- NumPy
- Pandas
