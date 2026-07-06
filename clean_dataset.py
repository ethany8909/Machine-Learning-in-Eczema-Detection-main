"""Data cleaning pass for the eczema/psoriasis set.

Applies, in order:
  1. Corrupt / unreadable image removal
  2. Off-label removal   (labels that are not eczema/psoriasis/dermatitis/lichenoid,
                          e.g. Tinea*, Balanitis* — they sit inside the two subclasses
                          but are a different disease process)
  3. Low-confidence removal  (annotator Confidence == 3 of 5 — reduces label noise)
  4. Perceptual-hash dedup   (near-exact duplicate images; keeps the highest-confidence
                              copy in each duplicate cluster)

Then rebuilds a subject-level, class-stratified 70/15/15 split in dataset_split_clean/
and writes CLEANING_REPORT.md documenting every change.
"""
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Skin_Metadata-1.csv"
SOURCE_DIRS = [ROOT / "DATASET_0", ROOT / "DATASET_1"]
OUT_SPLIT = ROOT / "dataset_split_clean"
REPORT = ROOT / "CLEANING_REPORT.md"
SEED = 42
RATIOS = (0.70, 0.15, 0.15)
KFOLDS = 5                  # for cross-validation fold assignment in the manifest

CLASS_MAP = {
    "Inflammatory skin diseases (Eczema and Dermatitis)": "eczema_dermatitis",
    "Inflammatory skin diseases (Psoriasis and Lichenoid disorders)": "psoriasis_lichenoid",
}
# labels to drop as off-target (case-insensitive substring match)
OFFLABEL_TOKENS = ["tinea", "balanitis"]
MIN_CONFIDENCE = 4          # drop Confidence < 4
PHASH_MAX_DISTANCE = 5      # <=5 bits on a 64-bit pHash == near-exact duplicate

log = []
def note(s=""):
    print(s)
    log.append(s)


# ---- load filtered rows + locate files -------------------------------------
file_index = {}
for d in SOURCE_DIRS:
    for f in d.iterdir():
        if f.is_file():
            file_index[f.name] = f

rows = []
with open(CSV_PATH, encoding="utf-8-sig", newline="") as fh:
    for r in csv.DictReader(fh):
        if r["Sub_class"] in CLASS_MAP:
            rows.append(r)
note(f"Starting eczema/psoriasis rows: {len(rows)}")
start_n = len(rows)

removed = defaultdict(list)   # reason -> [image_name]

# ---- step 1: corrupt / missing --------------------------------------------
kept = []
for r in rows:
    name = r["Image_name"]
    p = file_index.get(name)
    if p is None:
        removed["missing_file"].append(name); continue
    try:
        with Image.open(p) as im:
            im.verify()
        kept.append(r)
    except Exception:
        removed["corrupt_image"].append(name)
rows = kept
note(f"Step 1 — corrupt/missing removed: {len(removed['corrupt_image'])+len(removed['missing_file'])}")

# ---- step 2: off-label -----------------------------------------------------
kept = []
for r in rows:
    lab = r["Disease_label"].lower()
    if any(tok in lab for tok in OFFLABEL_TOKENS):
        removed["off_label"].append(f"{r['Image_name']} ({r['Disease_label']})")
    else:
        kept.append(r)
rows = kept
note(f"Step 2 — off-label removed: {len(removed['off_label'])}")

# ---- step 3: low confidence ------------------------------------------------
kept = []
for r in rows:
    try:
        conf = int(r["Confidence"])
    except (ValueError, KeyError):
        conf = 5
    if conf < MIN_CONFIDENCE:
        removed["low_confidence"].append(f"{r['Image_name']} (conf={conf})")
    else:
        kept.append(r)
rows = kept
note(f"Step 3 — low-confidence (<{MIN_CONFIDENCE}) removed: {len(removed['low_confidence'])}")

# ---- step 4: perceptual-hash dedup ----------------------------------------
hashes = {}
for r in rows:
    p = file_index[r["Image_name"]]
    hashes[r["Image_name"]] = imagehash.phash(Image.open(p).convert("RGB"))

names = [r["Image_name"] for r in rows]
conf_of = {r["Image_name"]: int(r["Confidence"]) if r["Confidence"].isdigit() else 5 for r in rows}
subj_of = {r["Image_name"]: r["Subject_ID"] for r in rows}
cls_of = {r["Image_name"]: CLASS_MAP[r["Sub_class"]] for r in rows}

assigned = set()
dup_clusters = []
for i, a in enumerate(names):
    if a in assigned:
        continue
    cluster = [a]
    for b in names[i + 1:]:
        if b in assigned:
            continue
        if hashes[a] - hashes[b] <= PHASH_MAX_DISTANCE:
            cluster.append(b); assigned.add(b)
    if len(cluster) > 1:
        assigned.update(cluster)
        dup_clusters.append(cluster)

dup_remove = set()
cross_subject, cross_class = 0, 0
for cluster in dup_clusters:
    keep = max(cluster, key=lambda n: (conf_of[n], n))   # highest confidence wins
    for n in cluster:
        if n != keep:
            dup_remove.add(n)
            removed["duplicate"].append(f"{n} ~= {keep}")
    if len({subj_of[n] for n in cluster}) > 1:
        cross_subject += 1
    if len({cls_of[n] for n in cluster}) > 1:
        cross_class += 1

rows = [r for r in rows if r["Image_name"] not in dup_remove]
note(f"Step 4 — duplicate clusters: {len(dup_clusters)}, images removed: {len(dup_remove)}")
note(f"         (clusters spanning >1 subject: {cross_subject}; spanning >1 class: {cross_class})")
note(f"\nFinal cleaned rows: {len(rows)}  (removed {start_n - len(rows)} total, {100*(start_n-len(rows))/start_n:.1f}%)")

# ---- rebuild split with JOINT class x Fitzpatrick stratification -----------
# Subjects are stratified jointly by their dominant class AND dominant Fitzpatrick
# band, so scarce dark-skin subjects (FST 5-6) are spread proportionally across
# train/val/test instead of landing randomly (previously FST 6 got 0 in val).
import random

def parse_fst(s):
    for tok in (s or "").replace("FST", "").split():
        if tok.isdigit():
            return int(tok)
    return -1

subj_imgs = defaultdict(list)                 # subject -> [(name, cls, fst)]
subj_cls_votes = defaultdict(Counter)
subj_fst_votes = defaultdict(Counter)
for r in rows:
    cls = CLASS_MAP[r["Sub_class"]]
    fst = parse_fst(r["Fitzpatrick"])
    subj_imgs[r["Subject_ID"]].append((r["Image_name"], cls, fst))
    subj_cls_votes[r["Subject_ID"]][cls] += 1
    subj_fst_votes[r["Subject_ID"]][fst] += 1
subj_cls = {s: v.most_common(1)[0][0] for s, v in subj_cls_votes.items()}
subj_fst = {s: v.most_common(1)[0][0] for s, v in subj_fst_votes.items()}

rng = random.Random(SEED)
strata = defaultdict(list)                     # (class, fst) -> [subjects]
for s in subj_imgs:
    strata[(subj_cls[s], subj_fst[s])].append(s)

split_of, fold_of = {}, {}
for key in sorted(strata, key=str):
    sids = sorted(strata[key]); rng.shuffle(sids)
    n = len(sids); ntr = round(n * RATIOS[0]); nva = round(n * RATIOS[1])
    for i, s in enumerate(sids):
        split_of[s] = "train" if i < ntr else ("val" if i < ntr + nva else "test")
        fold_of[s] = i % KFOLDS                # round-robin folds within each stratum

if OUT_SPLIT.exists():
    shutil.rmtree(OUT_SPLIT)
counts = defaultdict(Counter)
for s, imgs in subj_imgs.items():
    sp = split_of[s]
    for name, cls, fst in imgs:
        dest = OUT_SPLIT / sp / cls
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_index[name], dest / name)
        counts[sp][cls] += 1

# ---- write manifest (image -> class/label/subject/fst/split/fold) ----------
with open(ROOT / "manifest_clean.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["image_name", "class", "label", "subject", "fitzpatrick", "split", "fold"])
    for s, imgs in subj_imgs.items():
        for name, cls, fst in imgs:
            label = 0 if cls == "eczema_dermatitis" else 1
            w.writerow([name, cls, label, s, fst, split_of[s], fold_of[s]])

note("\nCleaned split (images):")
for sp in ("train", "val", "test"):
    e, p = counts[sp]["eczema_dermatitis"], counts[sp]["psoriasis_lichenoid"]
    note(f"  {sp:5} eczema={e:4} psoriasis={p:4} total={e+p}")

# ---- write report ----------------------------------------------------------
lines = ["# Data Cleaning Report\n",
         f"Original eczema/psoriasis images: **{start_n}** -> cleaned: **{len(rows)}** "
         f"(removed {start_n-len(rows)}).\n",
         "## Removals by step\n",
         f"1. **Corrupt/missing:** {len(removed['corrupt_image'])+len(removed['missing_file'])}",
         f"2. **Off-label** (Tinea/Balanitis): {len(removed['off_label'])}",
         f"3. **Low-confidence** (Confidence < {MIN_CONFIDENCE}): {len(removed['low_confidence'])}",
         f"4. **Near-duplicate** (pHash <= {PHASH_MAX_DISTANCE}): {len(dup_remove)} "
         f"from {len(dup_clusters)} clusters\n"]
for reason in ("off_label", "low_confidence", "duplicate"):
    if removed[reason]:
        lines.append(f"### {reason} ({len(removed[reason])})")
        lines += [f"- {x}" for x in removed[reason][:60]]
        if len(removed[reason]) > 60:
            lines.append(f"- ...and {len(removed[reason])-60} more")
        lines.append("")
lines.append("## Split")
lines.append(f"- Subject-level, **joint class x Fitzpatrick** stratified {RATIOS} split -> `{OUT_SPLIT.name}/`")
lines.append(f"- `manifest_clean.csv` written with per-image {KFOLDS}-fold assignment (subject-level, "
             "joint-stratified) for cross-validation.\n")
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
note(f"\nReport   -> {REPORT.name}")
note(f"Split    -> {OUT_SPLIT.name}/  (joint class x Fitzpatrick stratified)")
note(f"Manifest -> manifest_clean.csv  ({KFOLDS}-fold CV assignment)")
