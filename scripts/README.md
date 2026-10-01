# scripts/

The internal development pipeline, by stage. Run from the repository root.

| Folder | Script | Purpose |
|---|---|---|
| `data/` | `split_dataset.py` | Patient-level 70/15/15 split, jointly stratified by class and Fitzpatrick band |
| | `clean_dataset.py` | Perceptual-hash de-duplication; clean split and the five-fold CV manifest |
| | `make_table_s1.py` | Cohort composition by split, class and skin tone |
| `train/` | `train_image_models.py` | Five image backbones on the single split |
| | `train_fusion.py` | Metadata model, late fusion and gate network, in three metadata regimes |
| | `cross_validate.py`, `fusion_cv.py` | Five-fold CV of the backbones and the triage fusion models |
| | `baselines.py` | Majority-class, random and colour-histogram floors |
| `evaluate/` | `fairness_report.py`, `relative_fairness.py` | Per-tone metrics, gaps and paired-bootstrap comparisons |
| | `calibration.py`, `compare_results.py`, `auroc_comparison_cv.py` | Calibration, single-split vs CV, model comparison |
| | `gate_premise.py` | Gate weighting by skin tone |
| | `gradcam_atlas.py`, `misclassified.py` | Visual checks. **Their outputs contain patient images; keep them local.** |
| `figures/` | `make_figures.py`, `spectrum_figures.py` | Manuscript figures |
| | `readme_figures.py` | Summary figures in `docs/figures` |
