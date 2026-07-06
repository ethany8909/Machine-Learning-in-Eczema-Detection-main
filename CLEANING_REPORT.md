# Data Cleaning Report

Original eczema/psoriasis images: **1145** -> cleaned: **1125** (removed 20).

## Removals by step

1. **Corrupt/missing:** 0
2. **Off-label** (Tinea/Balanitis): 7
3. **Low-confidence** (Confidence < 4): 13
4. **Near-duplicate** (pHash <= 5): 0 from 0 clusters

### off_label (7)
- IMG_0626.jpg (Tinea Faciei)
- IMG_2124.jpg (Tinea pedis)
- IMG_2125.jpg (Tinea pedis)
- IMG_2992.jpg (Eczemated Tinea)
- IMG_3860.jpg (Zoon's Balanitis)
- IMG_9095.jpg (Balanitis)
- IMG_9811_1.jpg (Tinea Corporis)

### low_confidence (13)
- IMG_2964.jpg (conf=3)
- IMG_3116.jpg (conf=3)
- IMG_3416.jpg (conf=3)
- IMG_3418.jpg (conf=3)
- IMG_4338.jpg (conf=3)
- IMG_5342_1.jpg (conf=3)
- IMG_7707_1.jpg (conf=3)
- IMG20241212153837.jpg (conf=3)
- IMG20241212153848.jpg (conf=3)
- IMG20241212153857.jpg (conf=3)
- IMG20241218153728.jpg (conf=3)
- IMG20241218153838.jpg (conf=3)
- IMG20241220121929.jpg (conf=3)

## Split
- Subject-level, **joint class x Fitzpatrick** stratified (0.7, 0.15, 0.15) split -> `dataset_split_clean/`
- `manifest_clean.csv` written with per-image 5-fold assignment (subject-level, joint-stratified) for cross-validation.

