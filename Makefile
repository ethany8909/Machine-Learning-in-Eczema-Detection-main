# Reproduce the study end to end, in order. Targets read DERMAFAIR_DATA_DIR and write to
# DERMAFAIR_RESULTS_DIR (defaults: ./data and ./results; see docs/DATA.md).
# Times are approximate for an 8-core CPU without a GPU.
PY ?= python
D  := $(or $(DERMAFAIR_DATA_DIR),data)
R  := $(or $(DERMAFAIR_RESULTS_DIR),results)
IMP := experiments/improvements

.PHONY: help install lint test data internal spectrum external improvements synthetic-images fitzpatrick17k figures all

help:  ## List the targets
	@grep -E '^[a-z0-9_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-17s %s\n", $$1, $$2}'

install:  ## Install the package and development tools
	$(PY) -m pip install -e ".[dev]"

lint:  ## Lint and check formatting
	ruff check . && ruff format --check .

test:  ## Run the unit tests
	$(PY) -m pytest

data:  ## Split, de-duplicate and describe the internal cohort (minutes)
	$(PY) scripts/data/split_dataset.py
	$(PY) scripts/data/clean_dataset.py
	$(PY) scripts/data/make_table_s1.py

internal:  ## Five image backbones, single split and 5-fold CV (about 12 h)
	$(PY) scripts/train/train_image_models.py --split-root $(D)/dataset_split_clean --metadata-csv $(D)/Skin_Metadata-1.csv \
		--epochs 15 --patience 5 --device cpu --out-dir $(R)/image_models_clean
	$(PY) scripts/train/train_fusion.py --stage all --split-root $(D)/dataset_split_clean --metadata-csv $(D)/Skin_Metadata-1.csv \
		--out-dir $(R)/fusion_clean --image-ckpt-dir $(R)/image_models_clean/checkpoints \
		--image-metrics $(R)/image_models_clean/metrics.json --metadata-ckpt $(R)/fusion_clean/checkpoints/metadata_mlp.pt \
		--gate-entropy 0.1 --device cpu
	$(PY) scripts/train/cross_validate.py --manifest $(D)/manifest_clean.csv \
		--architectures cnn resnet50 custom_resnet50 vit_b16 hybrid --k 5 --epochs 15 --patience 5 --device cpu --out-dir $(R)/cv_clean
	$(PY) scripts/train/baselines.py

spectrum:  ## Metadata regimes (autonomous / triage / expert) and fusion CV (about 6 h)
	for r in autonomous triage expert; do \
		$(PY) scripts/train/train_fusion.py --stage all --regime $$r --split-root $(D)/dataset_split_clean \
			--metadata-csv $(D)/Skin_Metadata-1.csv --out-dir $(R)/spectrum/$$r \
			--image-ckpt-dir $(R)/image_models_clean/checkpoints --image-metrics $(R)/image_models_clean/metrics.json \
			--metadata-ckpt $(R)/spectrum/$$r/checkpoints/metadata_mlp.pt \
			--gate-entropy 0.1 --epochs 30 --patience 6 --batch-size 32 --device cpu || exit 1; \
	done
	$(PY) scripts/train/fusion_cv.py --regime triage --backbone-arch resnet50 --k 5 --epochs 25 --patience 6 \
		--gate-entropy 0.1 --device cpu --out-dir $(R)/cv_triage
	$(PY) scripts/evaluate/fairness_report.py
	$(PY) scripts/evaluate/relative_fairness.py
	$(PY) scripts/evaluate/calibration.py
	$(PY) scripts/evaluate/auroc_comparison_cv.py
	$(PY) scripts/evaluate/gate_premise.py
	$(PY) scripts/figures/spectrum_figures.py

external:  ## SCIN download, label harmonization and external validation (hours)
	$(PY) experiments/external/scin_inventory.py
	$(PY) experiments/external/scin_adapter.py
	$(PY) experiments/external/scin_download.py
	$(PY) experiments/external/external_validate_images.py
	$(PY) experiments/external/external_validate_metadata.py
	$(PY) experiments/external/external_validate_fusion.py
	$(PY) experiments/external/external_fusion_sweep.py
	$(PY) experiments/external/mitigation_tone_reweight.py
	$(PY) experiments/external/leave_one_source_out.py
	$(PY) experiments/external/gate_robust_experiment.py
	$(PY) experiments/external/crop_redundancy_experiment.py
	$(PY) experiments/external/crop_selectivity_experiment.py
	$(PY) experiments/external/reverse_validation.py

improvements:  ## Foundation-model features and the ten-point improvement study (about 10 h)
	$(PY) $(IMP)/extract_features.py
	$(PY) $(IMP)/extract_features.py --sizes s l
	$(PY) $(IMP)/run_experiments.py --groups baseline labels backbones
	$(PY) $(IMP)/run_experiments.py --groups imbalance fusion --labels original harmonized
	$(PY) $(IMP)/run_experiments.py --groups augment synthetic missing redundancy
	$(PY) $(IMP)/train_endtoend.py --name A_image_harmonized_mild --fusion none --policy mild
	$(PY) $(IMP)/train_endtoend.py --name B_midfusion_harmonized_mild --fusion mid --policy mild
	$(PY) $(IMP)/endtoend_compare.py
	$(PY) $(IMP)/run_experiments.py --groups final
	$(PY) $(IMP)/make_tables.py

synthetic-images:  ## Generated darker-skin images with SD-Turbo, then their evaluation (about 4 h)
	$(PY) $(IMP)/synth_images.py color tone neutral text --strength 0.1
	$(PY) $(IMP)/synth_eval.py
	$(PY) $(IMP)/make_tables.py

fitzpatrick17k:  ## Second external test set, scored once under the pre-specified plan
	$(PY) $(IMP)/fitz17k_download.py
	$(PY) $(IMP)/fitz17k_eval.py
	$(PY) $(IMP)/make_tables.py

figures:  ## README summary figures
	$(PY) scripts/figures/readme_figures.py

all: data internal spectrum external improvements synthetic-images fitzpatrick17k figures  ## Everything, in order
