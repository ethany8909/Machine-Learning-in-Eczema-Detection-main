# Data Integrity & Leakage Audit

Methods-section documentation of the checks ensuring reported performance reflects
genuine diagnostic signal, not leakage or confounds. This project began from a
*suspiciously high accuracy* observed in earlier work, so integrity was audited
explicitly. All checks are on the cleaned set (1,125 images) and the subject-level,
joint class × Fitzpatrick–stratified split.

## 1. Duplicate / near-duplicate images
Perceptual hashing (`imagehash.phash`, 64-bit) over all 1,125 images, pairwise
Hamming distance. **Zero near-duplicate clusters** at threshold ≤ 5 (nearest pair
= 16 bits). No identical or near-identical images span train/val/test. *(Ref:
`clean_dataset.py`, `CLEANING_REPORT.md`.)*

## 2. Patient-level (subject-level) splitting
Splitting is by `Subject_ID` (645 unique subjects; up to 11 images per subject),
not by image. **No patient contributes images to more than one split.** Image-level
splitting — a common source of inflated dermatology-AI results — is therefore
excluded by construction.

## 3. Augmentation leakage
Data augmentation (horizontal flip, ≤15° rotation, mild color jitter) is applied in
the **training transform only**; validation and test use resize + normalize. No
augmented view of a test image is ever seen in training.

## 4. Confound checks — labels are not trivially separable
- **Color:** a logistic regression on per-channel RGB histograms achieves
  **AUROC 0.538 (≈ chance)** on the test set — the eczema/psoriasis distinction is
  *not* trivially recoverable from color/exposure. *(Ref: `scripts/baselines.py`.)*
- **Acquisition source:** classes are only mildly imbalanced across the two source
  folders (psoriasis fraction 0.31 vs 0.42), both sources appear in every split, and
  the color-histogram probe above — which would capture any low-level camera/source
  signature — is at chance. No exploitable source confound.

## 5. Resolution of the high-accuracy anomaly
The inflated accuracy that motivated this audit is explained by **label leakage in
the clinical `Descriptors` metadata field** (morphologic terms such as "silvery
plaque" logged by the diagnosing dermatologist). Quantified by ablation: removing
descriptors drops metadata AUROC from 0.894 → 0.795. The paper therefore uses a
descriptor-free **triage** feature set (age, sex, body site) as its primary
operating point and reports the descriptor version only as a labelled leakage
ablation. *(Ref: `PROJECT_NOTEBOOK.md`, 2026-07-06.)* The **image** models are
modest and un-anomalous (CV AUROC ≈ 0.78), consistent with genuine, leakage-free
learning on a small, hard differential.

## 6. Sanity floor
Trivial baselines on the clean test split: majority-class accuracy = **0.692**,
stratified-random AUROC = 0.500, color-histogram AUROC = 0.538. Because majority
accuracy is high (class imbalance ~1.7:1), all results are reported as **balanced
accuracy / AUROC / per-tone metrics**, not raw accuracy. *(Ref:
`results/baselines/BASELINES.md`.)*

## Summary
No image-level leakage (subject split), no duplicate leakage (pHash), no
augmentation leakage, and no trivial color/source confound. The one genuine leakage
vector — expert descriptors — is identified, quantified, and quarantined to a
labelled ablation. Reported image/fusion performance reflects real signal.
