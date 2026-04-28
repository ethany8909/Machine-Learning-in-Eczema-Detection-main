# Datasets

This project uses the following publicly available datasets for training, validation, and testing. All images are used strictly for non-commercial academic research purposes.

---

## Dataset Summary

| # | Dataset | Source | Author / Publisher | Usage |
|---|---------|--------|--------------------|-------|
| 1 | Eczema Infected + Normal | [Kaggle](https://www.kaggle.com/datasets/adityush/eczema2) | Aditya Ush | Training / validation — binary eczema vs. normal |
| 2 | Skin Diseases Image Dataset | [Kaggle](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset) | Ismail Promus | Training — multi-class, negative samples |
| 3 | Skin Disease Detection (CNN Input) | [Kaggle](https://www.kaggle.com/code/srishtisinha169/skin-disease-detection-using-cnn/input) | Srishti Sinha | Supplementary training images |
| 4 | DermNet Skin Disease Image Dataset | [Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) | Shubham Goel / DermNet NZ | Training and testing — broad skin condition coverage |
| 5 | DermNet NZ — Atopic Dermatitis Images | [DermNet NZ](https://dermnetnz.org/images/atopic-dermatitis-images) | DermNet NZ | Test set — clinical eczema images |
| 6 | Dupixent Atopic Dermatitis Skin Gallery | [Dupixent](https://www.dupixent.com/atopicdermatitis/about/skin-gallery) | Sanofi / Regeneron | Real-world presentation reference, diverse skin tones |
| 7 | Arsenic Skin Disease Dataset | [Kaggle](https://www.kaggle.com/datasets/armaanoajay/arsenic-skin) | Armaan O. Ajay | Generalization — non-eczema skin presentations |

---

## Notes on Data Use

- Images from **DermNet NZ** are published under a peer-reviewed clinical dermatology license. Academic use is permitted with attribution.
- Images from **Dupixent** are sourced from a publicly accessible patient skin gallery maintained by Sanofi and Regeneron Pharmaceuticals.
- All Kaggle datasets are publicly available under their respective dataset licenses. Please check each dataset's Kaggle page for specific license terms before redistribution.
- No patient-identifiable information was used or stored in this project.

---

## Skin Tone Diversity

A key objective of this project is evaluating model fairness across diverse skin tones. Datasets were selected in part to ensure representation across the Fitzpatrick skin type scale. Bias analysis results are reported in the paper.

---

## Citation

If you use this project or its dataset curation in your own work, please cite this repository using the `CITATION.cff` file provided, and credit the original dataset authors listed above.
