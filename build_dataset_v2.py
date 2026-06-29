"""
build_dataset_v2.py
--------------------
Filters eczema and psoriasis images from the skin metadata dataset
and organizes them into dataset_v2 with a stratified train/test split
and class balancing (oversampling minority class in train set only).

Expected inputs:
  --csv       Path to metadata .tab or .csv file (e.g. Skin_Metadata.tab)
  --img_dir   Directory containing all downloaded images (DATASET_0 + DATASET_1 merged)
  --out_dir   Output root directory (default: ./dataset_v2)
  --test_size Fraction of data for test set (default: 0.2)
  --seed      Random seed for reproducibility (default: 42)
  --balance   Balance strategy: 'oversample', 'undersample', or 'none' (default: oversample)

Output structure:
  dataset_v2/
    train/
      eczema/
      psoriasis/
    test/
      eczema/
      psoriasis/
"""

import shutil
import hashlib
import argparse
import random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ── Label normalization ────────────────────────────────────────────────────────

TARGET_CONDITIONS = {
    "eczema": [
        "eczema", "atopic dermatitis", "contact dermatitis",
        "allergic contact dermatitis", "infected eczema",
        "ear eczema", "disseminated eczema", "dry discoid eczema",
        "crusted eczematous dermatitis", "chronic eczema",
        "seborrheic dermatitis", "eczemated tinea"
    ],
    "psoriasis": [
        "psoriasis", "chronic plaque psoriasis", "guttate psoriasis",
        "pustular psoriasis", "psoriasis vulgaris", "palmar psoriasis",
        "inverse psoriasis"
    ],
}

def normalize_label(label: str) -> str | None:
    """Map a raw label to 'eczema', 'psoriasis', or None."""
    label = str(label).strip().lower().strip('"')
    for canonical, variants in TARGET_CONDITIONS.items():
        if any(v in label for v in variants):
            return canonical
    return None


# ── Content-hash deduplication ─────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dedup_paths(paths: list[Path]) -> list[Path]:
    seen = {}
    unique = []
    for p in paths:
        h = file_hash(p)
        if h not in seen:
            seen[h] = p
            unique.append(p)
    removed = len(paths) - len(unique)
    if removed:
        print(f"  Removed {removed} duplicate image(s) by content hash.")
    return unique


# ── Class balancing (train set only) ──────────────────────────────────────────

def balance_classes(train_splits: dict, strategy: str, seed: int) -> dict:
    """Oversample minority or undersample majority in the train set."""
    counts = {label: len(df) for label, df in train_splits.items()}
    print(f"\n  Class counts before balancing: {counts}")

    if strategy == "none":
        return train_splits

    if strategy == "oversample":
        max_count = max(counts.values())
        balanced = {}
        for label, df in train_splits.items():
            if len(df) < max_count:
                n_extra = max_count - len(df)
                extra = df.sample(n=n_extra, replace=True, random_state=seed)
                balanced[label] = pd.concat([df, extra], ignore_index=True)
                print(f"  Oversampled {label}: {len(df)} → {len(balanced[label])}")
            else:
                balanced[label] = df
        return balanced

    if strategy == "undersample":
        min_count = min(counts.values())
        balanced = {}
        for label, df in train_splits.items():
            if len(df) > min_count:
                balanced[label] = df.sample(n=min_count, random_state=seed)
                print(f"  Undersampled {label}: {len(df)} → {len(balanced[label])}")
            else:
                balanced[label] = df
        return balanced

    raise ValueError(f"Unknown balance strategy: {strategy}")


# ── Core pipeline ──────────────────────────────────────────────────────────────

def build_dataset(csv_path: str, img_dir: str, out_dir: str,
                  test_size: float, seed: int, balance: str) -> None:

    csv_path = Path(csv_path)
    img_dir  = Path(img_dir)
    out_dir  = Path(out_dir)

    # 1. Load metadata (supports .tab and .csv)
    print(f"\n[1/6] Loading metadata from {csv_path} ...")
    sep = "\t" if csv_path.suffix.lower() == ".tab" else ","
    df = pd.read_csv(csv_path, sep=sep)
    # Strip quotes from column names
    df.columns = [c.strip().strip('"') for c in df.columns]
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Use known column names for this dataset
    label_col = "Disease_label"
    img_col   = "Image_name"

    if label_col not in df.columns:
        raise ValueError(f"Expected column '{label_col}' not found. Columns: {list(df.columns)}")
    if img_col not in df.columns:
        raise ValueError(f"Expected column '{img_col}' not found. Columns: {list(df.columns)}")

    # 2. Filter target conditions
    print(f"\n[2/6] Filtering eczema and psoriasis rows ...")
    df["_canonical"] = df[label_col].astype(str).apply(normalize_label)
    df_filtered = df[df["_canonical"].notna()].copy()
    print(f"  Eczema rows   : {(df_filtered['_canonical'] == 'eczema').sum()}")
    print(f"  Psoriasis rows: {(df_filtered['_canonical'] == 'psoriasis').sum()}")
    print(f"  Total filtered: {len(df_filtered)}")

    # 3. Resolve image paths (search across img_dir and subdirectories)
    print(f"\n[3/6] Resolving image paths in {img_dir} ...")

    # Build a lookup of all image filenames in img_dir for fast searching
    all_images = {}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            all_images[p.name.lower()] = p

    def find_image(img_name: str) -> Path | None:
        img_name = img_name.strip().strip('"')
        # Direct match
        if img_name.lower() in all_images:
            return all_images[img_name.lower()]
        # Try adding extensions
        for ext in (".jpg", ".jpeg", ".png"):
            key = (img_name + ext).lower()
            if key in all_images:
                return all_images[key]
        return None

    df_filtered["_path"] = df_filtered[img_col].astype(str).apply(find_image)
    missing = df_filtered["_path"].isna().sum()
    if missing:
        print(f"  Warning: {missing} image(s) not found on disk — skipping.")
    df_filtered = df_filtered[df_filtered["_path"].notna()].copy()
    print(f"  Images resolved: {len(df_filtered)}")

    # 4. Content-hash deduplication per class
    print(f"\n[4/6] Deduplicating by content hash ...")
    deduped_rows = []
    for label in ("eczema", "psoriasis"):
        subset = df_filtered[df_filtered["_canonical"] == label].copy()
        paths  = [Path(p) for p in subset["_path"].tolist()]
        print(f"  {label.capitalize()}: {len(paths)} images before dedup")
        unique_paths = dedup_paths(paths)
        path_set     = set(str(p) for p in unique_paths)
        subset       = subset[subset["_path"].apply(lambda p: str(p) in path_set)]
        deduped_rows.append(subset)
    df_final = pd.concat(deduped_rows, ignore_index=True)
    print(f"  Total after dedup: {len(df_final)}")

    # 5. Stratified train/test split
    print(f"\n[5/6] Splitting ({int((1-test_size)*100)}/{int(test_size*100)}) ...")
    train_splits = {}
    test_splits  = {}
    for label in ("eczema", "psoriasis"):
        subset = df_final[df_final["_canonical"] == label]
        train_df, test_df = train_test_split(
            subset, test_size=test_size, random_state=seed
        )
        train_splits[label] = train_df
        test_splits[label]  = test_df
        print(f"  {label.capitalize()}: {len(train_df)} train / {len(test_df)} test")

    # Balance train set only
    train_splits = balance_classes(train_splits, balance, seed)

    # 6. Copy files into dataset_v2
    print(f"\n[6/6] Copying files to {out_dir} ...")

    def copy_split(splits: dict, split_name: str):
        for label, df_part in splits.items():
            dst_dir = out_dir / split_name / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            seen_names = {}
            for _, row in df_part.iterrows():
                src  = Path(row["_path"])
                name = src.name
                # Handle duplicates from oversampling (same file copied twice)
                if name in seen_names:
                    seen_names[name] += 1
                    stem, suffix = src.stem, src.suffix
                    name = f"{stem}_{seen_names[src.name]}{suffix}"
                else:
                    seen_names[src.name] = 0
                shutil.copy2(src, dst_dir / name)

    copy_split(train_splits, "train")
    copy_split(test_splits,  "test")

    # Summary
    print(f"\n✅ Done! Dataset written to: {out_dir.resolve()}")
    print(f"{'Split':<10} {'Eczema':>10} {'Psoriasis':>12} {'Total':>8}")
    print("-" * 42)
    grand = 0
    for split_name in ("train", "test"):
        row_counts = []
        for label in ("eczema", "psoriasis"):
            d = out_dir / split_name / label
            row_counts.append(len(list(d.glob("*"))) if d.exists() else 0)
        total = sum(row_counts)
        grand += total
        print(f"{split_name:<10} {row_counts[0]:>10} {row_counts[1]:>12} {total:>8}")
    print("-" * 42)
    print(f"{'TOTAL':<10} {grand:>31}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset_v2 from skin metadata")
    parser.add_argument("--csv",       required=True,  help="Path to metadata .tab or .csv file")
    parser.add_argument("--img_dir",   required=True,  help="Directory of downloaded images")
    parser.add_argument("--out_dir",   default="dataset_v2", help="Output directory (default: dataset_v2)")
    parser.add_argument("--test_size", type=float, default=0.2,          help="Test fraction (default: 0.2)")
    parser.add_argument("--seed",      type=int,   default=42,           help="Random seed (default: 42)")
    parser.add_argument("--balance",   default="oversample",
                        choices=["oversample", "undersample", "none"],
                        help="Class balancing strategy for train set (default: oversample)")
    args = parser.parse_args()

    build_dataset(
        csv_path  = args.csv,
        img_dir   = args.img_dir,
        out_dir   = args.out_dir,
        test_size = args.test_size,
        seed      = args.seed,
        balance   = args.balance,
    )
