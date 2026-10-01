# experiments/

Analyses that build on the trained internal models. Run from the repository root, for example
`python experiments/external/external_validate_images.py`. Scripts within a folder import from one
another, so keep them together.

## external/: external validation on SCIN and follow-up analyses

| Script | Purpose |
|---|---|
| `scin_inventory.py` | Download SCIN case and label tables; count eligible eczema/psoriasis cases |
| `scin_adapter.py` | Label SCIN cases with the keyword rule; map both cohorts into one metadata space |
| `scin_download.py` | Download the SCIN images for the eligible cases |
| `external_validate_images.py` | Apply the internal image models to SCIN unchanged, stratified by skin tone |
| `external_validate_metadata.py` | Metadata model trained in the shared space, tested on SCIN |
| `external_validate_fusion.py` | Late fusion and gate network on SCIN |
| `external_fusion_sweep.py` | Internal vs external AUROC across fixed fusion weights |
| `mitigation_tone_reweight.py` | Retrain with tone-aware loss weighting; external re-test |
| `leave_one_source_out.py` | Within-cohort source split as a proxy for external shift |
| `reverse_validation.py` | Train on SCIN, test on the internal cohort |
| `gate_robust_experiment.py` | Gate variants: modality dropout and reliability signals |
| `crop_redundancy_experiment.py`, `crop_selectivity_experiment.py` | How much body-site information the image itself carries |

## improvements/: the ten-point improvement study, generated images and Fitzpatrick17k

| Script | Purpose |
|---|---|
| `common.py`, `lab.py` | Shared data loading, statistics, classifier heads, generators, fusion networks |
| `extract_features.py` | Encode every image once with DINOv2 (ViT-S/B/L) and ResNet-50 backbones |
| `run_experiments.py` | Grouped experiments: labels, imbalance, augmentation, synthetic data, missing values, redundancy, fusion, backbones, final model and adoption tests |
| `train_endtoend.py`, `endtoend_compare.py` | End-to-end ResNet-50 check of the frozen-feature conclusions |
| `synth_images.py`, `synth_eval.py` | Darker-skin and text-to-image sets from SD-Turbo, and their pre-specified evaluation |
| `fitz17k_download.py`, `fitz17k_eval.py` | The second external test set, scored once |
| `resummarize.py`, `make_tables.py` | Recompute summaries at full precision; render `TABLES.md` |
