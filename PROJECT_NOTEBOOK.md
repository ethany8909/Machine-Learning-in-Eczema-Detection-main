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

### 2026-07-06 — gate-network diagnosis (why it underperforms)
Problem is no longer collapse (fixed) but **overfitting of the learned fusion on small data**:
- Gate can represent late fusion exactly (weights=0.5) yet learned something WORSE → the learned
  weighting is a liability. It put 0.65 on the weaker image modality vs 0.35 on stronger metadata.
- Mechanisms: (1) late fusion has zero trainable fusion params so it can't overfit; the gate MLP learns
  per-sample weights from ~800 imgs that key off training-set image reliability and don't transfer.
  (2) Gate uses separate co-trained `image_head`/`meta_head` instead of the branches' own trained
  classifiers, degrading metadata's pathway → gate 0.801 < metadata-only 0.886. (3) Possibly no real
  per-sample (tone-adaptive) signal to exploit at this scale.
- Consistent across both runs (orig split 0.888<0.910; clean 0.801<0.956) → real, not noise. Still
  single-split; CV the fusion for error bars.
- **Fixes to try:** verify gate-weight vs Fitzpatrick correlation (Week 5 premise check); freeze both
  branches' trained classifiers so the gate only weights fixed logits; shrink/regularize the gate;
  anneal entropy λ from 0.5-ish. If premise fails → honest pivot: fixed fusion is the recommendation.

### 2026-07-06 — WEEK 4 COMPLETE (fairness evaluation engine)
`scripts/fairness_report.py` → `results/week4_fairness/`: **Table 3** (master fairness table, 8 models ×
per-tone metrics + gaps + fairness score + Kruskal-Wallis + 1000× bootstrap CIs), **Figure 2** (per-Fitzpatrick
heatmap: accuracy/TPR/FPR/F1), **Figure 3** (accuracy–fairness Pareto).
- **Headline: Late fusion is Pareto-optimal** — best accuracy (0.877) & AUROC (0.956), best equalized-odds
  (0.250), highest worst-group acc (0.851), tightest acc-gap CI [0.103,0.263], KW p=0.73 (no sig disparity).
- **No model shows a statistically significant tone disparity** (all KW p>0.05; smallest CNN p=0.15) — expected
  at this N. Report as descriptive + CIs, do NOT claim "proven fair."
- **Caveats:** TPR/equalized-odds gaps are inflated by the degenerate FST-6 cell (n≈1 psoriasis); wide CIs
  everywhere from small per-tone N; fusion numbers still single-split (CV pending). Gate network is also the
  worst fusion on fairness (eq-odds 0.917, widest CI).

### 2026-07-06 — metadata leakage QUANTIFIED (descriptor ablation)
Sklearn logistic regression on clean test (independent of torch pipeline):
- All metadata (age,sex,body,descriptors) AUROC 0.894; **NO descriptors 0.795**; descriptors-only 0.831;
  body-site-only 0.753.
- **Conclusion:** metadata's dominance is ~half LEAKAGE (Descriptors encode the dermatologist's morphological
  read → −0.10 when removed) and ~half REAL (body-site distribution genuinely diagnostic: flexural=eczema,
  extensor=psoriasis, 0.753 alone). Descriptor-free metadata (0.795) ≈ best image model (~0.79 CV).
- **Implications:** (1) headline comparison should use descriptor-FREE (deployment-realistic) metadata, with
  the descriptor ablation reported as a leakage finding; (2) re-run fusion with descriptor-free metadata —
  image and metadata become comparable (~0.79 each), which is the regime where the gate could add value.

### 2026-07-06 — framing decisions (grill session) + relative-fairness machinery
- **Paper spine decided:** 3-regime SPECTRUM (autonomous {age,sex} → triage {+body_part, PRIMARY} →
  expert {+descriptors}), image constant across regimes; metadata/fusion vary. Triage is the recommended
  operating point; descriptor version reported as a leakage ablation. Contribution = fairness benchmark +
  reusable protocol + leakage finding; gate demoted to "one benchmarked fusion strategy" (option C: bounded
  ~1-day rescue gated on the Day-1 premise check).
- **Fairness reporting:** primary Table 3 = triage; centerpiece = fairness-vs-regime trend figure.
- **Statistical framing (small-N):** estimation + CIs over significance; FST 6 descriptive-only; relative
  consistency-based claims; spectrum trend as dose-response evidence.
- **Relative-fairness IMPLEMENTED:** `dermafair/fairness/relative.py` + `scripts/relative_fairness.py` —
  paired bootstrap (shared resamples across models) → P(model A fairer than B) + cross-metric consistency.
  Illustrative result (contaminated data): no model consistently fairer than late fusion (consistency ≤0.24).
  Rerun on descriptor-free triage results.
- **TODO from this session:** descriptor-free re-run (metadata+fusion, all 3 regimes); rebuild all
  metadata/fusion figures; update flowchart for spectrum; then Day-1 gate premise check.
- **Venue:** JMIR AI primary, IEEE J-BHI backup (gate is negative → flips the original method-forward rule).
- **Protocol = co-headline contribution (full product):** clean API, config, interpretation guide, tests,
  pip-install, Zenodo DOI, runnable example. ~80% already built. Rule: new analysis code goes INSIDE the
  `dermafair` package, not one-off scripts.
- **Execution scope:** single-split all 3 regimes descriptor-free (bootstrap gives CIs, no CV needed for
  relative claims) + CV only the TRIAGE regime. Critical path: (1) parametrize metadata feature set →
  (2) single-split 3 regimes → (3) Day-1 gate premise check → (4) rebuild figures/tables → (5) overnight CV triage.
- **GRILL COMPLETE — all 10 framing decisions locked.**
- **Critical-path step (1) DONE:** regimes are feature-ablation on the SAME images/split (not a data
  partition). `MetadataEncoder(single_fields, multi_fields)` + `REGIMES` + `.for_regime()`;
  `build_multimodal_dataloaders(regime=)`; `train_fusion.py --regime`. Nested dims: autonomous 8 ⊂ triage 50
  ⊂ expert 84. Validated: triage metadata AUROC 0.79 (matches descriptor-free ablation → no leakage;
  logistic 0.793 ≥ mlp 0.772). Next: step (2) single-split all 3 regimes descriptor-free.
- **Step (2) LAUNCHED 2026-07-12 23:34** (detached, PID-independent): `run_spectrum.sh` → metadata+fusion for
  autonomous/triage/expert on the clean split → `results/spectrum/<regime>/`. Image backbone reused (regime-
  independent). Early confirmation: autonomous metadata AUROC 0.64 (weak, demographics-only) — spectrum
  bottom rung as predicted. ETA ~1 h. Monitor `results/spectrum/spectrum_status.log`.

### 2026-07-13 — SPECTRUM COMPLETE (descriptor-free, all 3 regimes) — key results
AUROC (image-only ≈0.78–0.79, regime-independent):
| regime | metadata | late fusion | gate | gate img-wt |
|---|---|---|---|---|
| autonomous (age,sex) | 0.591 | 0.779 | 0.789 | 0.62 |
| **triage (+body)** | 0.780 | **0.820** | 0.809 | 0.68 |
| expert (+desc) | 0.907 | 0.949 | 0.916 | 0.55 |
- **Dose-response confirmed:** metadata 0.59→0.78→0.91 as clinical context added (centerpiece trend).
- **Multimodal value in triage:** fusion 0.820 > image ~0.79 > metadata 0.78. Clean, non-leaky headline.
- **GATE REHABILITATED:** descriptor-free at triage, gate 0.809 ≈ late 0.820 (~0.01 tie), NOT the earlier
  blowout (0.801 vs 0.956) — that gap was descriptor *leakage* inflating late fusion. Honest finding upgrades
  from "gate badly loses" → "gate matches fixed fusion."
- **Regime-level adaptation:** gate image-weight drops to 0.55 in expert (metadata strongest) vs 0.68 triage —
  it down-weights image when metadata is strong. Makes the Day-1 per-tone premise check worth running.
- Data: `results/spectrum/<regime>/`. Next: (3) per-tone gate premise check on triage; (4) rebuild
  spectrum figures + Table 3 from results/spectrum.

### 2026-07-13 — gate premise check + spectrum figures (steps 3 & 4)
- **Gate premise (per-tone weights):** metadata-weight trends UP with darker skin, consistently across all
  3 regimes (Spearman rho +0.11..+0.15) but NOT significant (triage p=0.11). Verdict: **SUGGESTIVE but
  UNDERPOWERED** → hedged secondary finding, not headline. → `results/week5_gate/` (Fig 4 + PREMISE_CHECK.md).
- **Table 3 (descriptor-free triage):** metadata-only fairest (gap 0.118) but least accurate; late fusion most
  accurate (0.82) but larger gap (0.229) → real accuracy–fairness tradeoff. → `results/week5_triage_fairness/`.
- **Spectrum figures** (`results/week5_spectrum/`):
  - `spectrum_accuracy.png` — CLEAN centerpiece: value-of-info dose-response; fusion > both modalities at
    triage; leakage jump at expert. Publishable as-is.
  - `spectrum_fairness.png` — HONEST reality: gaps modest but CIs enormous & fully overlapping → fairness is
    statistically indistinguishable across models/regimes (underpowered). Point estimates hint metadata-at-
    triage fairest / gate worsens with context, but not reliable. This figure = the motivation for larger
    tone-balanced data, not a fairness verdict. Fully consistent with the estimation/power framing.
- **Remaining TODO:** update pipeline flowchart (still cites leaky 0.897); optional bounded gate-rescue
  (frozen classifiers) since premise is suggestive; overnight-CV the triage regime.

### 2026-07-13 03:00 — triage fusion CV LAUNCHED (detached)
`run_cv_triage.sh` → `scripts/fusion_cv.py` 5-fold CV of metadata + late + gate in the TRIAGE regime,
**reusing per-fold backbones** `results/cv_clean/checkpoints/resnet50_fold<i>.pt` (trained without fold i's
test → no leakage; only the cheap fusion layer retrains per fold). New code: `MultimodalListDataset` +
`build_cv_multimodal_from_manifest` in folder_split.py. Smoke (fold0/1ep): late 0.813, gate 0.796 — path OK.
ETA ~2 h → `results/cv_triage/CV_TRIAGE.md`. Log: `results/cv_triage_run.log` (marker "ALL DONE").

### 2026-07-13 05:04 — TRIAGE CV COMPLETE (5-fold, mean ± std)
| model | AUROC | accuracy | acc gap |
|---|---|---|---|
| metadata-only | 0.697 ± 0.047 | 0.645 ± 0.028 | 0.199 ± 0.092 |
| late fusion | 0.771 ± 0.022 | 0.734 ± 0.027 | 0.253 ± 0.084 |
| **gate** | **0.785 ± 0.033** | 0.745 ± 0.030 | **0.193 ± 0.100** |
(image backbone resnet50 CV = 0.779)
- **Finding 1 — gate edges late fusion under CV (REVERSES single-split):** gate 0.785 > late 0.771 AND fairer
  (gap 0.193 < 0.253). Stds overlap (not significant) but direction flipped in gate's favor → gate upgraded
  from "negative result" to "competitive-to-marginally-better fusion strategy."
- **Finding 2 — multimodal gain smaller than single-split implied (honest tempering):** under CV, fusion
  (0.771–0.785) ≈ image backbone (0.779); single-split had shown fusion 0.82 vs image 0.79 (+0.03). Honest
  claim: "fusion ≥ best single modality, clearly > metadata-only, gains modest & within CV noise." This is
  why CV was worth running.
- **Headline framing update:** lead with the value-of-information accuracy spectrum + the gate being
  competitive-and-fairer under CV; hedge the absolute multimodal-gain magnitude.

### 2026-07-13 — on the modest absolute accuracy (~0.78–0.82) — Discussion/Limitations framing
Numbers are lower than headline derm-AI SOTA (~0.90–0.96). Reasons, in impact order: (1) ~800 train images
vs SOTA's 10k–100k+ (dominant factor); (2) eczema-vs-psoriasis is a genuinely hard differential (both
inflammatory/scaly; dermatologists disagree); (3) broad lumped classes (eczema+dermatitis, psoriasis+lichenoid);
(4) HONEST leakage-free eval (subject-level split + CV) — much reported SOTA is inflated by image-level
leakage (this project literally started from a "suspicious high-accuracy anomaly"); (5) CPU limits (frozen
backbones, no TTA/ensembling/tuning). NOT a problem because venue = methods/fairness (JMIR), contribution =
protocol + benchmark + leakage finding, not SOTA accuracy. DOES constrain: must frame as proof-of-concept /
protocol demo, NOT deployment-ready (say so in Limitations). Credibility = numbers are REAL not high.
Optional lifts (off-thesis): GPU fine-tune, TTA, ensembling, higher res, more/external data (also fixes FST-6 power).

### 2026-07-13 — Tier 1 + Tier 2 artifacts DONE
- **Flowchart fixed** (`results/figures/pipeline_flowchart.svg`): now shows the 3-regime metadata spectrum
  (autonomous/triage/expert, expert flagged LEAKY), ResNet-50 as fusion backbone, gate "CV: fairer,
  competitive"; removed the bogus 0.897 claim.
- **Grad-CAM atlas** (`results/week5_gradcam/gradcam_atlas.png`, Fig 8) — 4 conv archs × 4 FST bands.
  Finding: ResNet-50 attends to lesions; CNN scatters to background (weak baseline); attention less focused
  on FST 6 for several models. Manual Grad-CAM via cam_target_layer hooks. ViT omitted (needs reshape).
  LOCAL ONLY — patient photos (results/ is gitignored).
- **Dummy baselines** (`results/baselines/BASELINES.md`): majority-class acc **0.692** (so report bal-acc/
  AUROC, not raw acc — several image models barely clear majority); stratified-random AUROC 0.500;
  color-histogram AUROC **0.538 ≈ chance** (task NOT trivially color-separable — confound check passes).
- **LEAKAGE_AUDIT.md** (root): methods-section writeup — pHash dedup (0), subject-level split, train-only
  aug, color + source-folder confound checks (source class-frac 0.31 vs 0.42, mild; color-probe at chance),
  descriptor-leakage resolution, sanity floor. Resolves the original high-accuracy anomaly.
- **Status:** analysis + all Results artifacts complete. Remaining = writing (Results draft), protocol
  packaging + Zenodo DOI, finish interrupted GitHub push. Nothing running.

### 2026-07-13 — figures consolidated + calibration (Fig S1)
- **All 9 paper figures consolidated** into `results/paper_figures/` with clean names (fig1..fig8 + figS1).
  Old scattered folders (`figures/`, `figures_clean/`, `week4_fairness/`) are STALE/leaky — ignore; the
  paper set is `paper_figures/` only. fig8 Grad-CAM = LOCAL ONLY (patient photos).
- **Calibration (Fig S1, reliability diagrams, triage):** late fusion best-calibrated (ECE 0.067), metadata
  0.079; **gate poorly calibrated (ECE 0.143)** ≈ raw image (0.144). Finding: gate is competitive on
  AUROC/fairness but UNRELIABLE in confidence → caveat + motivates temperature scaling (deferred). Late
  fusion is accurate AND well-calibrated → pragmatic recommendation.
- **Added `fig2b_auroc_comparison_cv.png`** — cross-model AUROC bar chart with 5-fold CV error bars
  (descriptor-free). Ranking: Gate 0.785 > ResNet-50 0.779 > Late 0.771 > Hybrid 0.768 > ViT 0.763 >
  Custom 0.734 > Metadata 0.697 > CNN 0.615. Top 4 overlap (indistinguishable); metadata & CNN clearly
  weaker; late fusion tightest CI (±0.022). paper_figures now has 10 figures.

### 2026-07-13 — full manuscript draft written (MANUSCRIPT.md)
Structured after the JMIR AI reference (Vivek & Ramesh multimodal melanoma paper the user provided) —
Abstract/Intro/Methods/Results/Discussion/Conclusion + back matter — but entirely original prose grounded
in our real results and honest framing (estimation not significance, descriptor leakage, gate competitive-
but-poorly-calibrated, fairness underpowered, proof-of-concept not deployment). Citations left as `[ref]`
placeholders (user will supply). Tables 2/3/5 populated with real numbers; Table 1 (lit) and Table 4
(per-tone fairness) left to fill; author/affiliation/ethics/dataset-citation placeholders.

### 2026-07-13 — manuscript typeset to PDF (MANUSCRIPT.pdf, 21 pp)
- Figures + tables moved INLINE into their sections (no bottom dump): Table 1 in Intro; Fig 1 in Methods;
  Figs 2A/2B+Table 2, Fig 3, Table 3+Fig S1, Figs 4A/4B/5+Table 4, Fig 6, Fig 7, Table 5 in matching Results
  subsections. Verified 5 tables / 10 images, each exactly once.
- `build_manuscript_pdf.py` renders MANUSCRIPT.md -> MANUSCRIPT.pdf via ReportLab (no LaTeX/pandoc on this
  machine). Typography: US Letter, 1in margins, Times serif, 10.5pt justified body, 16pt centered title,
  bold section heads / bold-italic subheads, indented 10pt abstract, 7.4pt tables w/ shaded header + zebra
  rows, running header + page numbers. SVG (Fig 1) embedded as VECTOR via svglib; 9 rasters embedded.
- Journal conventions applied: figure captions BELOW figures (tables above), caption+figure bound with
  KeepTogether so they never split across pages; figure height capped 6.1in to limit page gaps.
- Fixes during build: WinAnsi char sanitation (arrows/Greek/<=/~ mapped) since Times base-14 can't encode
  them; abbreviations list was collapsing into one run-on paragraph -> now one line each.
- Tooling installed: reportlab, svglib, pypdf, pypdfium2 (pypdfium2 also used to render pages for visual QA).
