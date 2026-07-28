#!/usr/bin/env bash
# Overnight: 5-fold CV of the metadata + fusion models in the TRIAGE regime,
# reusing per-fold backbones from results/cv_clean/ (no leakage).
cd "C:/Users/ethan/OneDrive/Documents/Stage2-EczemaPsoriasis/dermafair" || exit 1
export PYTHONPATH=.
echo "[$(date '+%F %T')] START triage fusion CV" > results/cv_triage_run.log
python -u scripts/fusion_cv.py --regime triage --backbone-arch resnet50 --k 5 \
  --epochs 25 --patience 6 --gate-entropy 0.1 --device cpu \
  --out-dir results/cv_triage >> results/cv_triage_run.log 2>&1
echo "[$(date '+%F %T')] ALL DONE (exit $?)" >> results/cv_triage_run.log
