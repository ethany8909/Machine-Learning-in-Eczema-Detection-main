# eczema-cnn-classifier

## Overview
CNN-based binary classifier for eczema detection using skin images. Built with TensorFlow/Keras, with data augmentation, evaluation metrics, and feature map visualization for interpretability.

## Features
- CNN model (Conv → Pool → Dense)
- Data augmentation (flip, rotation, zoom)
- Evaluation: accuracy, confusion matrix, classification report
- Misclassification analysis
- Feature map visualization

## Installation
```bash
git clone https://github.com/yourusername/eczema-cnn.git
cd eczema-cnn
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
## Installation
```bash
git clone https://github.com/yourusername/eczema-cnn.git
cd eczema-cnn
pip install tensorflow keras numpy matplotlib seaborn scikit-learn

---

## Data Sources

This project was trained and evaluated on a curated dataset of ~2000 images drawn from 7 publicly available sources, selected to ensure representation across diverse skin tones and eczema presentations.

| # | Dataset | Link |
|---|---------|------|
| 1 | Eczema Infected + Normal | [Kaggle](https://www.kaggle.com/datasets/adityush/eczema2) |
| 2 | Skin Diseases Image Dataset | [Kaggle](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset) |
| 3 | Skin Disease Detection — CNN Input | [Kaggle](https://www.kaggle.com/code/srishtisinha169/skin-disease-detection-using-cnn/input) |
| 4 | DermNet Skin Disease Image Dataset | [Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) |
| 5 | DermNet NZ — Atopic Dermatitis Images | [DermNet NZ](https://dermnetnz.org/images/atopic-dermatitis-images) |
| 6 | Dupixent Atopic Dermatitis Skin Gallery | [Dupixent](https://www.dupixent.com/atopicdermatitis/about/skin-gallery) |
| 7 | Arsenic Skin Disease Dataset | [Kaggle](https://www.kaggle.com/datasets/armaanoajay/arsenic-skin) |

See [DATASETS.md](./DATASETS.md) for full attribution, license notes, and details on how each dataset was used.
