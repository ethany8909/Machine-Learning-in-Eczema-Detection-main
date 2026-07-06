"""
Filter the skin metadata for Eczema/Dermatitis and Psoriasis/Lichenoid images,
then split them into train/val/test subfolders at the SUBJECT level
(so a patient's images never span multiple splits), stratified by class.
"""
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Skin_Metadata-1.csv"
SOURCE_DIRS = [ROOT / "DATASET_0", ROOT / "DATASET_1"]
OUT_DIR = ROOT / "dataset_split"
SEED = 42
RATIOS = (0.70, 0.15, 0.15)  # train, val, test

CLASS_MAP = {
    "Inflammatory skin diseases (Eczema and Dermatitis)": "eczema_dermatitis",
    "Inflammatory skin diseases (Psoriasis and Lichenoid disorders)": "psoriasis_lichenoid",
}

# --- Build an index of available image files ---------------------------------
file_index = {}
for d in SOURCE_DIRS:
    for f in d.iterdir():
        if f.is_file():
            file_index[f.name] = f

# --- Read + filter metadata --------------------------------------------------
records = []  # (image_name, subject_id, class_label, source_path)
missing = []
with open(CSV_PATH, encoding="utf-8-sig", newline="") as fh:
    for row in csv.DictReader(fh):
        cls = CLASS_MAP.get(row["Sub_class"])
        if cls is None:
            continue
        name = row["Image_name"]
        src = file_index.get(name)
        if src is None:
            missing.append(name)
            continue
        records.append((name, row["Subject_ID"], cls, src))

print(f"Selected metadata rows with a file on disk: {len(records)}")
if missing:
    print(f"WARNING: {len(missing)} selected images not found on disk, e.g. {missing[:5]}")

# --- Assign each subject to a single (dominant) class ------------------------
subj_images = defaultdict(list)          # subject -> list of records
subj_class_votes = defaultdict(Counter)  # subject -> class counts
for rec in records:
    _, sid, cls, _ = rec
    subj_images[sid].append(rec)
    subj_class_votes[sid][cls] += 1

subj_class = {sid: votes.most_common(1)[0][0] for sid, votes in subj_class_votes.items()}

# --- Stratified subject-level split -----------------------------------------
rng = random.Random(SEED)
subjects_by_class = defaultdict(list)
for sid, cls in subj_class.items():
    subjects_by_class[cls].append(sid)

split_of_subject = {}
for cls, sids in subjects_by_class.items():
    sids = sorted(sids)
    rng.shuffle(sids)
    n = len(sids)
    n_train = int(round(n * RATIOS[0]))
    n_val = int(round(n * RATIOS[1]))
    for i, sid in enumerate(sids):
        if i < n_train:
            split_of_subject[sid] = "train"
        elif i < n_train + n_val:
            split_of_subject[sid] = "val"
        else:
            split_of_subject[sid] = "test"

# --- Copy files --------------------------------------------------------------
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
counts = defaultdict(Counter)  # split -> class -> count
for sid, recs in subj_images.items():
    split = split_of_subject[sid]
    for name, _, cls, src in recs:
        dest_dir = OUT_DIR / split / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / name)
        counts[split][cls] += 1

# --- Report ------------------------------------------------------------------
print("\nImages per split / class:")
header = f"{'split':<8}{'eczema_dermatitis':>20}{'psoriasis_lichenoid':>22}{'total':>8}"
print(header)
grand = 0
for split in ("train", "val", "test"):
    e = counts[split]["eczema_dermatitis"]
    p = counts[split]["psoriasis_lichenoid"]
    grand += e + p
    print(f"{split:<8}{e:>20}{p:>22}{e+p:>8}")
print(f"{'TOTAL':<8}{'':>20}{'':>22}{grand:>8}")

n_subj = Counter(split_of_subject.values())
print(f"\nSubjects: train={n_subj['train']}, val={n_subj['val']}, test={n_subj['test']}, total={len(split_of_subject)}")
