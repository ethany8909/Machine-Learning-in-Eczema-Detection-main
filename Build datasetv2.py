"""
build_dataset_v2.py
--------------------
Filters eczema and psoriasis images from the Fitzpatrick17k dataset
and organizes them into dataset_v2 with a stratified train/test split.

Expected inputs:
  --csv       Path to Fitzpatrick17k metadata CSV (e.g. fitzpatrick17k.csv)
  --img_dir   Directory containing all downloaded Fitzpatrick17k images
  --out_dir   Output root directory (default: ./dataset_v2)
  --test_size Fraction of data for test set (default: 0.2)
  --seed      Random seed for reproducibility (default: 42)

Output structure:
  dataset_v2/
    train/
      eczema/
      psoriasis/
    test/
      eczema/
      psoriasis/
"""

import os
import re
import shutil
import hashlib
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ── Label normalization ────────────────────────────────────────────────────────

TARGET_CONDITIONS = {
    "eczema": [
        "eczema", "atopic dermatitis", "contact dermatitis",
        "dyshidrotic eczema", "nummular eczema", "seborrheic dermatitis"
    ],
    "psoriasis": [
        "psoriasis", "plaque psoriasis", "guttate psoriasis",
        "pustular psoriasis", "psoriasis vulgaris"
    ],
}

def normalize_label(label: str) -> str | None:
    """Map a raw Fitzpatrick17k label to 'eczema', 'psoriasis', or None."""
    label = label.strip().lower()
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
    """Return a deduplicated list of image paths based on MD5 content hash."""
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


# ── Core pipeline ──────────────────────────────────────────────────────────────

def build_dataset(csv_path: str, img_dir: str, out_dir: str,
                  test_size: float, seed: int) -> None:

    csv_path = Path(csv_path)
    img_dir  = Path(img_dir)
    out_dir  = Path(out_dir)

    # 1. Load metadata
    print(f"\n[1/5] Loading metadata from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")

    # Detect label column (common names in Fitzpatrick17k)
    label_col = next(
        (c for c in df.columns if c.lower() in ("label", "condition", "three_partition_label")),
        None
    )
    if label_col is None:
        raise ValueError(f"Could not find a label column. Columns found: {list(df.columns)}")
    print(f"  Using label column: '{label_col}'")

    # Detect image filename column
    img_col = next(
        (c for c in df.columns if c.lower() in ("image_id", "md5hash", "filename", "file")),
        None
    )
    if img_col is None:
        raise ValueError(f"Could not find an image ID column. Columns found: {list(df.columns)}")
    print(f"  Using image column: '{img_col}'")

    # 2. Filter target conditions
    print(f"\n[2/5] Filtering eczema and psoriasis rows ...")
    df["_canonical"] = df[label_col].astype(str).apply(normalize_label)
    df_filtered = df[df["_canonical"].notna()].copy()
    print(f"  Eczema rows   : {(df_filtered['_canonical'] == 'eczema').sum()}")
    print(f"  Psoriasis rows: {(df_filtered['_canonical'] == 'psoriasis').sum()}")
    print(f"  Total filtered: {len(df_filtered)}")

    # 3. Resolve image paths and drop missing files
    print(f"\n[3/5] Resolving image paths in {img_dir} ...")

    def find_image(img_id: str) -> Path | None:
        # Try common extensions
        for ext in ("", ".jpg", ".jpeg", ".png", ".bmp"):
            p = img_dir / f"{img_id}{ext}"
            if p.exists():
                return p
        # Fuzzy: search for filename containing img_id
        matches = list(img_dir.glob(f"*{img_id}*"))
        return matches[0] if matches else None

    df_filtered["_path"] = df_filtered[img_col].astype(str).apply(find_image)
    missing = df_filtered["_path"].isna().sum()
    if missing:
        print(f"  Warning: {missing} image(s) not found on disk — skipping.")
    df_filtered = df_filtered[df_filtered["_path"].notna()].copy()
    print(f"  Images resolved: {len(df_filtered)}")

    # 4. Content-hash deduplication per class
    print(f"\n[4/5] Deduplicating by content hash ...")
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

    # 5. Stratified train/test split and copy files
    print(f"\n[5/5] Splitting ({int((1-test_size)*100)}/{int(test_size*100)}) and copying files ...")

    splits = {"train": [], "test": []}
    for label in ("eczema", "psoriasis"):
        subset = df_final[df_final["_canonical"] == label]
        train_df, test_df = train_test_split(
            subset, test_size=test_size, random_state=seed
        )
        splits["train"].append(train_df)
        splits["test"].append(test_df)
        print(f"  {label.capitalize()}: {len(train_df)} train / {len(test_df)} test")

    for split_name, dfs in splits.items():
        for df_part in dfs:
            for _, row in df_part.iterrows():
                label   = row["_canonical"]
                src     = Path(row["_path"])
                dst_dir = out_dir / split_name / label
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst     = dst_dir / src.name
                # Avoid overwriting with a suffix if name collision
                if dst.exists():
                    stem, suffix = src.stem, src.suffix
                    dst = dst_dir / f"{stem}_{file_hash(src)[:8]}{suffix}"
                shutil.copy2(src, dst)

    # Summary
    print(f"\n✅ Done! Dataset written to: {out_dir.resolve()}")
    for split_name in ("train", "test"):
        for label in ("eczema", "psoriasis"):
            d = out_dir / split_name / label
            count = len(list(d.glob("*"))) if d.exists() else 0
            print(f"  {split_name}/{label}: {count} images")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset_v2 from Fitzpatrick17k")
    parser.add_argument("--csv",       required=True,  help="Path to Fitzpatrick17k CSV")
    parser.add_argument("--img_dir",   required=True,  help="Directory of downloaded images")
    parser.add_argument("--out_dir",   default="dataset_v2", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction (default 0.2)")
    parser.add_argument("--seed",      type=int,   default=42,  help="Random seed (default 42)")
    args = parser.parse_args()

    build_dataset(
        csv_path  = args.csv,
        img_dir   = args.img_dir,
        out_dir   = args.out_dir,
        test_size = args.test_size,
        seed      = args.seed,
    )
