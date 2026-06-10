# Two-Stage Image Forgery Detection using U-Net++ and LightGBM

## Overview

This project aims to detect manipulated images and localize forged regions using a two-stage deep learning framework.

The system first identifies suspicious regions through a U-Net++ segmentation model and then performs final forgery classification using a LightGBM classifier.

The objective is not only to determine whether an image is forged, but also to explain where the manipulation has occurred.

---

## Pipeline

```text
Input Image
      │
      ▼
U-Net++
(Forgery Localization)
      │
      ▼
Heatmap Generation
      │
      ▼
Feature Extraction
      │
      ▼
LightGBM
(Forgery Classification)
      │
      ▼
Real / Forged
```

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

---

## Dataset

### DEFACTO Dataset

Synthetic image dataset used for forgery localization.

### COCO Dataset

Authentic image dataset used as negative samples.

### Data Distribution

* Total Images: 30,000
* Real Images: 15,000
* Forged Images: 15,000
* Real : Fake = 1 : 1

---

## Methodology

### Stage 1: Forgery Localization (U-Net++)

The first stage predicts manipulated regions at the pixel level.

#### Outputs

* Forgery Mask
* Localization Heatmap

#### Objective

Learn where image manipulation occurs rather than only determining whether manipulation exists.

---

### Stage 2: Forgery Classification (LightGBM)

The second stage utilizes localization information generated from Stage 1.

#### Input Features

* Heatmap Statistics
* Localization Features

#### Outputs

* Real Image
* Forged Image

#### Objective

Classify images based on forgery localization characteristics extracted from Stage 1.

---

## Key Features

* U-Net++ based forgery localization
* Pixel-level manipulation detection
* Heatmap generation
* Feature extraction
* LightGBM classification
* Group leakage prevention
* Group-aware validation strategy
* Confusion Matrix visualization
* Two-stage detection framework

---

## Technologies

* Python
* PyTorch
* U-Net++
* LightGBM
* OpenCV
* Albumentations
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib

---

## Results

| Metric | Score |
|----------|----------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |

### Example Result

#### Input Image

![Input Image](images/compostie_images_sample.jpg)

#### Predicted Heatmap

![Heatmap](images/heatmap_sample.jpg)

#### Interpretation

The U-Net++ model highlights manipulated regions through a localization heatmap, which is subsequently used by the LightGBM classifier for final forgery detection.

---

## My Contribution

As the Stage 1 developer, I was responsible for:

* Dataset preprocessing
* U-Net++ model training
* Forgery mask generation
* Heatmap generation
* Localization feature extraction
* Model evaluation
* Visualization and analysis
* Training pipeline implementation

---

## Future Work

* Advanced U-Net++ optimization
* Attention-based localization networks
* Explainable AI (XAI) integration
* Real-world image forgery benchmark evaluation
* Cross-dataset generalization experiments

---

## License

This project is developed for academic and research purposes.
