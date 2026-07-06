# Project Notebook — Fairness-Aware Multimodal Eczema–Psoriasis Diagnosis

A running log of decisions, results, and the "ups and downs" — kept for reference when
writing the manuscript (Methods, Limitations, and the Discussion of what did/didn't work).

---

## 1. Project at a glance
- **Task:** binary differential diagnosis, eczema/dermatitis (label 0) vs psoriasis/lichenoid (label 1).
- **Angle:** compare image architectures + multimodal fusion, and evaluate every model for
  **fairness across Fitzpatrick skin tones**. Novelty core = a learned **gate network** that
  adaptively weights image vs metadata per sample.
- **Toolkit:** the `dermafair/` package (models, trainer, fairness metrics, figures).

## 2. Dataset (important corrections vs the original plan)
- **1145 images, not 514.** The timeline/README were written for DermaCon-IN (514). The actual
  working set is 1145 images filtered from a 5450-row `Skin_Metadata-1.csv` (DATASET_0 + DATASET_1).
  **Reconcile this number before writing Methods.**
- **Class definition:** filter the `Sub_class` column to the two inflammatory subclasses
  ("Eczema and Dermatitis" → 727; "Psoriasis and Lichenoid disorders" → 418). ~1.74:1 imbalance.
- **Split:** subject-level (by `Subject_ID`, 645 subjects), class-stratified 70/15/15, seed 42.
  Subject-level is essential — up to 11 images per patient, so image-level splitting would leak.
- **FST 6 is critically underpowered:** 32 images total; psoriasis×FST6 = 5 (0 val / 1 test).
  Any FST 6 fairness claim is descriptive only. (See `TABLE_S1_composition.md`.)

## 3. Compute reality
- **CPU-only.** Machine has an AMD Radeon 780M iGPU → no CUDA. ROCm (Linux-only) and DirectML
  (unreliable for ViT) are not usable. Installed CPU PyTorch (torch 2.12.1+cpu, Python 3.14).
- Consequences: ViT-B/16 is the slow pole (hours); fusion **freezes the image backbone** (can't
  afford to fine-tune the whole backbone on CPU). Code auto-detects CUDA (`--device auto`), so a
  real NVIDIA/cloud GPU works unchanged.
- Practical note: long CPU runs were repeatedly interrupted; models are checkpointed and results
  written incrementally so runs can resume without losing completed models.

## 4. Results so far

### Week 2 — image architectures (test set, N=171)
| Model | Acc | Bal-Acc | AUROC | F1 | ECE |
|---|---|---|---|---|---|
| CNN (from scratch) | 0.398 | 0.502 | 0.553 | 0.498 | 0.188 |
| ResNet-50 | 0.749 | 0.730 | 0.773 | 0.650 | 0.158 |
| Custom ResNet-50 | 0.713 | 0.695 | 0.776 | 0.608 | 0.143 |
| ViT-B/16 | 0.678 | 0.687 | 0.715 | 0.610 | 0.080 |
| **Hybrid CNN-Transformer** | **0.772** | **0.763** | **0.807** | 0.693 | 0.208 |

- **Up:** Hybrid is the best image model (AUROC 0.807) and is auto-selected as the fusion backbone.
- **Down:** the from-scratch CNN is degenerate — class-weighting over-corrected and it predicts
  psoriasis 85% of the time (eczema recall 0.15). Expected "baseline floor" on 812 images, but
  document it so raw-accuracy numbers aren't misread.
- **Watch:** Hybrid is badly **over-confident** (ECE 0.208) — several 100%-confident wrong
  predictions. Flagged for temperature scaling in Week 5.

### Week 3 — metadata + fusion
| Model | Acc | AUROC | Notes |
|---|---|---|---|
| Metadata-only (MLP) | 0.819 | **0.897** | strongest single modality |
| Late fusion (0.5/0.5) | 0.760 | 0.806 | ~= image alone (broken run) |
| Gate network (broken) | 0.766 | 0.793 | collapsed to 99.8% image |

*(These are the initial BROKEN numbers, kept as history. After the collapse fix,
late fusion reaches AUROC 0.910 and gate 0.888 — see §5.)*

- **Big finding / down:** metadata alone (AUROC 0.897) beats every image model, yet the first
  gate network learned to put **99.8% weight on image** and fusion ended up *worse* than metadata
  alone. See §5 for the diagnosis, fix, and corrected numbers.

## 5. The fusion bug (modality collapse) — diagnosis & fix
**Symptom:** gate weight ≈ 0.998 on image; fusion AUROC (0.806) ≈ image-only (0.807) and *below*
metadata-only (0.897). A working fusion should be ≥ the best single modality.

**Root cause:** in the first fusion run the image branch was loaded **pre-trained + frozen** while
the metadata branch was **randomly initialized** inside the fusion graph. So at epoch 1 the image
path already had low loss and metadata was noise → the gate routed to image → once w_meta ≈ 0 the
metadata branch got ~no gradient → permanent collapse ("greedy modality dominance").

**Fix (implemented):**
1. **Warm-start the metadata branch** from the trained metadata-only MLP (not random init).
2. **Warm-start the per-branch heads** from each branch's trained classifier, so both modalities
   emit meaningful logits from epoch 1.
3. **LayerNorm on gate-input features** so the gate isn't biased by raw feature-scale mismatch
   between a deep image backbone and a shallow metadata MLP.
4. **Gate-entropy penalty** (`--gate-entropy`, default 0.1): rewards using both modalities,
   preventing early collapse.

**Result of the fix (same split, `results/fusion_fixed/`):**
| Model | Acc | AUROC | F1 | ECE | gate img-weight |
|---|---|---|---|---|---|
| metadata-only (ref) | 0.801 | 0.879 | 0.702 | 0.133 | — |
| late fusion (broken → fixed) | 0.760 → **0.877** | 0.806 → **0.910** | 0.667 → 0.821 | 0.117 → 0.065 | — |
| gate network (broken → fixed) | 0.766 → **0.830** | 0.793 → **0.888** | 0.688 → 0.764 | 0.179 → 0.129 | 0.999 → **0.565 (std 0.09)** |

- **Up:** fusion now *works* — both variants exceed the best single modality (image 0.807, metadata 0.879).
  The gate no longer collapses (0.565/0.435, per-sample std 0.09). Late-fusion is also now well-calibrated (ECE 0.065).
- **Down / open question for Week 5:** the *learned* gate (0.888) does **not** beat the *fixed* late fusion
  (0.910). The entropy penalty (λ=0.1) pulls the gate toward ~0.5/0.5, so it essentially reproduces late
  fusion. λ is now a tunable knob with a real tension: too high → mimics late fusion; too low → collapses to
  one modality. Whether an *adaptive tone-dependent* gate can beat fixed fusion is exactly the Week-5 novelty
  test — and per the plan, "gate ≈ fixed fusion" is still an honest, publishable finding.
- **Fairness (small-N, noisy — FST6 has 4 test images):** accuracy-gap late 0.175 / meta 0.206 / gate 0.250;
  worst-group accuracy late 0.825 / meta 0.794 / gate 0.750. No clean fairness winner yet.

## 6. Metrics & figures added
- **Novel metrics** (`dermafair/fairness/advanced_metrics.py`): AUROC, AUPRC, balanced accuracy,
  MCC, Cohen's κ, Brier, ECE/MCE; group-fairness: equalized-odds diff, equal-opportunity diff,
  demographic- & predictive-parity diff, AUROC gap, worst-group acc/AUROC/F1. On top of the
  existing Kruskal–Wallis + bootstrap-CI code.
- **Focal loss** added to the trainer (`--loss focal`) as a fallback for class imbalance.
- **Figures** (`dermafair/results/figures/`, via `scripts/make_figures.py` + `scripts/misclassified.py`):
  ROC overlays, confusion matrices, AUROC bar, per-Fitzpatrick fairness heatmap, gate-weights-per-epoch,
  misclassified montage, and the pipeline flowchart (`pipeline_flowchart.svg`).

## 7. Data cleaning pass (`clean_dataset.py` → `CLEANING_REPORT.md`)
Applied in order, 1145 → **1125** images (−20, 1.7%):
1. Corrupt/missing: **0** (all images valid).
2. Off-label (Tinea/Balanitis inside the two subclasses): **7 removed**.
3. Low-confidence (annotator Confidence < 4/5): **13 removed** (label-noise reduction).
4. Near-duplicate (pHash ≤ 5): **0** — the set has no near-duplicate leakage (nearest pair
   distance ≈ 16). A clean-data finding worth stating for reviewers.
- Gradability/Quality columns were checked: **all 1145 already "sufficient"**, nothing to filter.
- Output: `dataset_split_clean/` (subject-level, stratified, same seed).

## 8. Open decisions / risks
- **Metadata leakage question (unresolved, deferred):** metadata beats images largely because the
  `Descriptors` field (plaque, erythema, scaling) encodes the dermatologist's reasoning. Decide the
  deployment scenario — clinician-in-the-loop (descriptors available) vs patient-selfie (image only,
  descriptors unavailable → training on them is leakage). This changes the whole product story.
- **Temperature scaling** (calibration) deferred to Week 5.
- **GPU fine-tuning** deferred until NVIDIA/cloud access; on CPU the backbone stays frozen.

## 9. Status vs 8-week plan
- Week 1 (data): done (split + Table S1). Outstanding: `LEAKAGE_AUDIT.md` write-up (dedup now done),
  dummy baselines.
- Week 2 (image benchmarking): done.
- Week 3 (metadata + fusion): done; fusion **re-run with the collapse fix**.
- Week 4 (fairness engine): mostly prepped — compile Table 3 (master fairness table) + heatmap/Pareto.
- Week 5: gate behavioral analysis, Grad-CAM, temperature scaling.

## 10. Session changelog

### 2026-07-05
- **Fusion collapse fixed and confirmed** (§5): late fusion AUROC 0.806 → **0.910**, gate 0.793 → **0.888**;
  gate image-weight 0.999 → **0.565** (no collapse). Both fusion variants now beat the best single modality.
  Fixes: warm-start metadata branch + heads, LayerNorm gate inputs, gate-entropy penalty (λ=0.1).
- **Open Week-5 question surfaced:** learned gate (0.888) does not yet beat fixed late fusion (0.910);
  entropy λ pulls the gate toward late-fusion behavior. λ is now a tunable knob to explore.
- **Figures regenerated** against the fixed run — `scripts/make_figures.py` now points at
  `results/fusion_fixed/`. ROC/AUROC/gate-weight plots reflect the corrected results.
- **Data cleaning pass** completed (§7): 1145 → 1125; clean split in `dataset_split_clean/`.
- **Pending (user running overnight):** full re-run of the 5 image models + fixed fusion on the
  cleaned split, so all clean-data results are consistent. Command:
  `python scripts/train_image_models.py --split-root ../dataset_split_clean --metadata-csv ../Skin_Metadata-1.csv --epochs 15 --patience 5 --device cpu`
  then `python scripts/train_fusion.py --stage all --split-root ../dataset_split_clean --metadata-csv ../Skin_Metadata-1.csv --out-dir results/fusion_clean --metadata-ckpt results/fusion_clean/checkpoints/metadata_mlp.pt --gate-entropy 0.1 --device cpu`.
- **New practice:** from now on, important conversation details/decisions are logged in this changelog, dated.
- **Timeline checkpoint:** Weeks 1–3 complete; currently between Week 3 and Week 4. Next up is Week 4
  (fairness master table + heatmap/Pareto), gated on the overnight clean-data re-run.
- **Implemented overfitting/high-FST mitigations** (see §8): joint class×Fitzpatrick stratification +
  `manifest_clean.csv`, and a k-fold CV harness (`scripts/cross_validate.py`). Clean split regenerated
  (1125 imgs; train/val/test 802/177/146). **Overnight plan — all 5 backbones in CV (user has time).**
  One unattended runner `dermafair/run_overnight.sh` runs 3 stages sequentially, each logged, writing to
  `*_clean` folders so existing single-split results are preserved:
  1. image models (5) on clean split → `results/image_models_clean/`
  2. metadata + fixed fusion on clean split → `results/fusion_clean/`
  3. **5-fold CV of ALL 5 backbones** (cnn, resnet50, custom_resnet50, vit_b16, hybrid) → `results/cv_clean/`
  Est. CPU wall-time ≈ 9–11 h (CV of all 5 ≈ 7–8 h). Live progress in `results/overnight_status.log`.
- **Overnight run LAUNCHED 2026-07-05 12:25.** Added: (a) CV now saves per-fold checkpoints + predictions
  (`results/cv_clean/checkpoints|predictions/`); (b) `run_overnight.sh` stage 4 auto-regenerates figures
  against clean results (`results/figures_clean/`, via env-var overrides in `make_figures.py`) and writes a
  single-split-vs-CV comparison table (`scripts/compare_results.py` → `results/cv_clean/COMPARISON.md`).
  Reliability note: if the background job dies on session teardown, relaunch with `bash run_overnight.sh`
  in a normal terminal (survives independently).
- **Run died on session teardown, relaunched detached (2026-07-05 13:02).** First launch (via the
  in-session background mechanism) was killed mid-CNN; also spawned duplicate racing instances. Cleaned
  up all stray python/bash, then relaunched a SINGLE run via PowerShell `Start-Process` (hidden, detached
  from the session) → PID independent of Claude Code. Lesson: for long unattended runs use a detached
  process (`Start-Process`) or a normal terminal, not the in-session background tool.
- **Risk review — overfitting & high-FST scarcity (for Limitations):**
  - *Overfitting:* mitigated by early stopping (patience), train-only augmentation, weight decay 1e-4,
    transfer learning, and dropout (custom-resnet50 head 0.5, metadata MLP 0.2). BUT train→val loss
    divergence is clear (e.g. ResNet-50 train 0.06 / val 0.81; metadata MLP train 0.08 / val 0.96) —
    early stopping caps the harm but residual overfit on 812 images is real. Gaps: single split (no
    k-fold CV), small val set (162) → noisy stopping.
  - *High-FST scarcity:* FST 6 = 32 images total (val 3 / test 4); split is stratified by **class only**,
    not jointly by Fitzpatrick, so tone balance across splits is uncontrolled. Mitigations available:
    joint class×FST-stratified re-split, tone-aware augmentation/oversampling, always report bootstrap
    CIs. FST 6 remains descriptive-only regardless; FST 1–2 absent (Indian cohort).
- **Overfitting + high-FST mitigations IMPLEMENTED (2026-07-05):**
  1. **Joint class×Fitzpatrick stratification** — `clean_dataset.py` now stratifies subjects jointly by
     class *and* dominant Fitzpatrick band. Result: FST 6 spread across splits (val 3→6, test 4→5) and
     across all 5 CV folds (8/7/7/4/6) instead of landing near-zero. Also writes `manifest_clean.csv`
     (per-image class/label/subject/fst/split/fold).
  2. **K-fold cross-validation** — new `scripts/cross_validate.py` runs k-fold CV (subject-level,
     joint-stratified via the manifest) and reports **mean ± std** per architecture → `results/cv/CV_RESULTS.md`.
     New CV data loader: `folder_split.build_cv_dataloaders_from_manifest`. Verified working (CNN k=2 smoke).
  - **CPU caveat:** CV wall-time ≈ k × single-split time, so on CPU run CV on the light backbones
    (cnn, resnet50, custom_resnet50); do vit_b16/hybrid CV on a GPU later.

### 2026-07-06 — overnight run COMPLETE (finished 01:07)
- **5-fold CV, image backbones (AUROC mean ± std):** resnet50 **0.779 ± 0.026** (best + most stable),
  hybrid 0.768 ± 0.063, vit_b16 0.763 ± 0.067, custom_resnet50 0.734 ± 0.039, cnn 0.615 ± 0.055.
  *Key CV insight:* single-split made Hybrid look best, but CV shows Hybrid/ViT are **high-variance**
  (±0.06–0.07) while **ResNet-50 is strong AND stable** — the single split overstated Hybrid/ViT.
  (`results/cv_clean/CV_RESULTS.md`)
- **Fusion (clean, SINGLE-split — not yet CV'd):** metadata_mlp 0.886, **late_fusion 0.956**,
  gate_network **0.801** (gate img-weight 0.653). Late fusion is excellent, BUT the **learned gate again
  UNDERPERFORMS fixed late fusion** — badly this time (0.801 < 0.886 metadata < 0.956 late). The gate
  leans 65% on the *weaker* image modality, dragging it below a plain 0.5/0.5 average. Reinforces the
  Week-5 pivot: adaptive gating is not beating fixed fusion.
- **Caveats:** fusion numbers are single-split only; late_fusion 0.956 is high and MUST be cross-validated
  before it's a headline claim (fusion CV not run tonight — image backbones only). Gate design needs
  revisiting (entropy λ; why it favors the weaker modality).
- **Outputs:** `results/{image_models_clean,fusion_clean,cv_clean}/`, figures in `results/figures_clean/`,
  `results/cv_clean/COMPARISON.md`.
