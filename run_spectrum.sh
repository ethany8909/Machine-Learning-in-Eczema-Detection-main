#!/usr/bin/env bash
# Step (2): descriptor-free spectrum re-run — metadata + fusion for all 3 regimes
# on the SAME clean split (dataset_split_clean). Image backbone is regime-independent
# (reused from results/image_models_clean/). Each regime writes to results/spectrum/<regime>/.
cd "C:/Users/ethan/OneDrive/Documents/Stage2-EczemaPsoriasis/dermafair" || exit 1
export PYTHONPATH=.
mkdir -p results/spectrum
ts() { date '+%Y-%m-%d %H:%M:%S'; }
STATUS=results/spectrum/spectrum_status.log
echo "[$(ts)] START spectrum re-run (autonomous, triage, expert)" > "$STATUS"

for R in autonomous triage expert; do
  echo "[$(ts)] regime: $R" | tee -a "$STATUS"
  python -u scripts/train_fusion.py --stage all --regime "$R" \
    --split-root ../dataset_split_clean --metadata-csv ../Skin_Metadata-1.csv \
    --out-dir "results/spectrum/$R" \
    --image-ckpt-dir results/image_models_clean/checkpoints \
    --image-metrics results/image_models_clean/metrics.json \
    --metadata-ckpt "results/spectrum/$R/checkpoints/metadata_mlp.pt" \
    --gate-entropy 0.1 --epochs 30 --patience 6 --batch-size 32 --device cpu \
    > "results/spectrum/${R}.log" 2>&1
  echo "[$(ts)] regime $R done (exit $?)" | tee -a "$STATUS"
done

echo "[$(ts)] ALL REGIMES DONE" | tee -a "$STATUS"
