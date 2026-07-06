# Daily Log

## 2026-07-05

**Data**
- Filtered the 1145 eczema/psoriasis images from the metadata CSV (two `Sub_class` groups).
- Built a subject-level, class-stratified 70/15/15 split (`dataset_split/`).
- Generated Table S1 (class × Fitzpatrick composition); confirmed FST 6 is underpowered.
- Ran a data-cleaning pass → 1145 → 1125 (removed 7 off-label, 13 low-confidence; 0 corrupt, 0 duplicates). New `dataset_split_clean/`.

**Models**
- Trained all 5 image backbones — Hybrid CNN-Transformer won (AUROC 0.807).
- Trained metadata-only models — MLP strongest single modality (AUROC 0.897).
- Built + trained late fusion and gate network.

**The big fix**
- Diagnosed the fusion "modality collapse" (gate stuck at 99.8% image; fusion worse than metadata alone).
- Fixed it: warm-start metadata branch, LayerNorm gate inputs, gate-entropy penalty.
- Confirmed fix: late fusion AUROC 0.910, gate 0.888 — both now beat every single modality; gate balanced at 0.56/0.44.

**Infra / evaluation**
- Added novel metrics (AUROC, AUPRC, MCC, Brier, ECE, equalized-odds / worst-group fairness).
- Added focal-loss option for class imbalance.

**Figures & docs**
- Generated ROC curves, confusion matrices, AUROC bar, fairness heatmap, gate-weights plot, misclassified montage.
- Made the pipeline flowchart (`pipeline_flowchart.svg`).
- Regenerated all figures against the fixed fusion results.
- Wrote `PROJECT_NOTEBOOK.md` (full journey) and this log.

**Decisions made**
- Hold off on committing to a single modality (image vs metadata) for now.
- Defer temperature scaling / calibration to Week 5.
- Keep image backbone frozen on CPU (no NVIDIA GPU available).

**Next (running overnight)**
- Full re-run of image models + fixed fusion on the cleaned split (`dataset_split_clean/`).
