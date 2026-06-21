"""
Rebuild dataset/train_data and dataset/test_data with:
1. Content-based deduplication (catches same image under different filenames)
2. Clean 80/20 split with zero overlap between train and test
3. Original folders backed up before anything is touched

Run this from the repo root:
    python3 rebuild_dataset.py
"""

import os
import shutil
import hashlib
import random
from pathlib import Path

# ---- CONFIG ----
REPO_ROOT = Path(".")  # run from repo root
TRAIN_DIR = REPO_ROOT / "dataset" / "train_data"
TEST_DIR = REPO_ROOT / "dataset" / "test_data"
BACKUP_DIR = REPO_ROOT / "dataset_backup_before_rebuild"
CLASSES = ["Normal", "Eczema"]
TEST_RATIO = 0.20
RANDOM_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

random.seed(RANDOM_SEED)


def file_hash(path):
    """Return md5 hash of file content (not filename) for true dedup."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(folder):
    if not folder.exists():
        return []
    return [f for f in folder.iterdir() if f.suffix in VALID_EXTENSIONS]


def main():
    print("=" * 60)
    print("STEP 1: Backing up current dataset (safety net)")
    print("=" * 60)
    if BACKUP_DIR.exists():
        print(f"Backup already exists at {BACKUP_DIR}, skipping backup step.")
    else:
        shutil.copytree(REPO_ROOT / "dataset", BACKUP_DIR)
        print(f"Backed up dataset/ to {BACKUP_DIR}")

    for cls in CLASSES:
        print("\n" + "=" * 60)
        print(f"Processing class: {cls}")
        print("=" * 60)

        train_files = collect_files(TRAIN_DIR / cls)
        test_files = collect_files(TEST_DIR / cls)
        all_files = train_files + test_files
        print(f"Found {len(train_files)} in train, {len(test_files)} in test "
              f"({len(all_files)} total before dedup)")

        # Hash every file, keep only first occurrence of each unique hash
        seen_hashes = {}
        unique_files = []
        for f in all_files:
            try:
                h = file_hash(f)
            except Exception as e:
                print(f"  WARNING: could not hash {f}: {e}")
                continue
            if h not in seen_hashes:
                seen_hashes[h] = f
                unique_files.append(f)

        dup_count = len(all_files) - len(unique_files)
        print(f"Removed {dup_count} duplicate file(s) by content hash")
        print(f"{len(unique_files)} unique images remain for {cls}")

        # Shuffle and split
        random.shuffle(unique_files)
        n_test = int(len(unique_files) * TEST_RATIO)
        new_test = unique_files[:n_test]
        new_train = unique_files[n_test:]
        print(f"New split -> train: {len(new_train)}, test: {len(new_test)}")

        # Write to fresh folders (use temp names, swap in at the end)
        new_train_dir = REPO_ROOT / "dataset" / "train_data_NEW" / cls
        new_test_dir = REPO_ROOT / "dataset" / "test_data_NEW" / cls
        new_train_dir.mkdir(parents=True, exist_ok=True)
        new_test_dir.mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(new_train, start=1):
            ext = f.suffix.lower()
            dest = new_train_dir / f"{cls.lower()}{i}{ext}"
            shutil.copy2(f, dest)

        for i, f in enumerate(new_test, start=1):
            ext = f.suffix.lower()
            dest = new_test_dir / f"{cls.lower()}{i}{ext}"
            shutil.copy2(f, dest)

        print(f"Wrote new train/test files for {cls}")

    print("\n" + "=" * 60)
    print("STEP 2: Verifying zero overlap by content hash")
    print("=" * 60)
    for cls in CLASSES:
        train_hashes = {file_hash(f) for f in collect_files(REPO_ROOT / "dataset" / "train_data_NEW" / cls)}
        test_hashes = {file_hash(f) for f in collect_files(REPO_ROOT / "dataset" / "test_data_NEW" / cls)}
        overlap = train_hashes & test_hashes
        print(f"{cls}: overlap = {len(overlap)} (should be 0)")
        if overlap:
            print("  WARNING: overlap detected! Do not proceed until this is 0.")

    print("\n" + "=" * 60)
    print("DONE. Review dataset/train_data_NEW and dataset/test_data_NEW")
    print("If everything looks correct, swap them in manually:")
    print("  rm -rf dataset/train_data dataset/test_data")
    print("  mv dataset/train_data_NEW dataset/train_data")
    print("  mv dataset/test_data_NEW dataset/test_data")
    print("=" * 60)


if __name__ == "__main__":
    main()