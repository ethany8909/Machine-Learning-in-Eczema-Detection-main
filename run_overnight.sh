#!/usr/bin/env bash
# Unattended overnight run on the CLEANED, joint-stratified data.
# Runs three stages sequentially, each logged separately. Clean-data results go to
# *_clean folders so the existing single-split results are NOT overwritten.
cd "C:/Users/ethan/OneDrive/Documents/Stage2-EczemaPsoriasis/dermafair" || exit 1
export PYTHONPATH=.
mkdir -p results
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] START overnight run" | tee results/overnight_status.log

# 1) All 5 image backbones on the clean split -> results/image_models_clean/
echo "[$(ts)] stage 1: image models (clean split)" | tee -a results/overnight_status.log
python -u scripts/train_image_models.py \
  --split-root ../dataset_split_clean --metadata-csv ../Skin_Metadata-1.csv \
  --epochs 15 --patience 5 --device cpu \
  --out-dir results/image_models_clean > results/overnight_1_images.log 2>&1
echo "[$(ts)] stage 1 done (exit $?)" | tee -a results/overnight_status.log

# 2) Metadata + fixed fusion on the clean split -> results/fusion_clean/
echo "[$(ts)] stage 2: fusion (clean split)" | tee -a results/overnight_status.log
python -u scripts/train_fusion.py \
  --stage all --split-root ../dataset_split_clean --metadata-csv ../Skin_Metadata-1.csv \
  --out-dir results/fusion_clean \
  --image-ckpt-dir results/image_models_clean/checkpoints \
  --image-metrics results/image_models_clean/metrics.json \
  --metadata-ckpt results/fusion_clean/checkpoints/metadata_mlp.pt \
  --gate-entropy 0.1 --device cpu >> results/overnight_2_fusion.log 2>&1
echo "[$(ts)] stage 2 done (exit $?)" | tee -a results/overnight_status.log

# 3) 5-fold cross-validation, ALL FIVE backbones -> results/cv_clean/
echo "[$(ts)] stage 3: 5-fold CV (all 5 backbones)" | tee -a results/overnight_status.log
python -u scripts/cross_validate.py \
  --manifest ../manifest_clean.csv \
  --architectures cnn resnet50 custom_resnet50 vit_b16 hybrid \
  --k 5 --epochs 15 --patience 5 --device cpu \
  --out-dir results/cv_clean >> results/overnight_3_cv.log 2>&1
echo "[$(ts)] stage 3 done (exit $?)" | tee -a results/overnight_status.log

# 4) regenerate figures against CLEAN results + build CV comparison table
echo "[$(ts)] stage 4: figures (clean) + comparison table" | tee -a results/overnight_status.log
DERMAFAIR_IMG_DIR=results/image_models_clean/predictions \
DERMAFAIR_FUS_DIR=results/fusion_clean/predictions \
DERMAFAIR_FIG_OUT=results/figures_clean \
  python -u scripts/make_figures.py > results/overnight_4_figures.log 2>&1
python -u scripts/misclassified.py --split-root ../dataset_split_clean \
  --pred results/image_models_clean/predictions/hybrid.npz \
  --out results/figures_clean/misclassified.png >> results/overnight_4_figures.log 2>&1
python -u scripts/compare_results.py >> results/overnight_4_figures.log 2>&1
echo "[$(ts)] stage 4 done (exit $?)" | tee -a results/overnight_status.log

echo "[$(ts)] ALL DONE" | tee -a results/overnight_status.log
