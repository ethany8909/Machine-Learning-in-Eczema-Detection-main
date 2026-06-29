import shutil, hashlib, argparse, random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

TARGET_CONDITIONS = {
    "eczema": ["eczema","atopic dermatitis","contact dermatitis","allergic contact dermatitis","infected eczema","ear eczema","disseminated eczema","dry discoid eczema","crusted eczematous dermatitis","chronic eczema","seborrheic dermatitis","eczemated tinea"],
    "psoriasis": ["psoriasis","chronic plaque psoriasis","guttate psoriasis","pustular psoriasis","psoriasis vulgaris","palmar psoriasis","inverse psoriasis"],
}

def normalize_label(label):
    label = str(label).strip().lower().strip('"')
    for canonical, variants in TARGET_CONDITIONS.items():
        if any(v in label for v in variants):
            return canonical
    return None

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def dedup_paths(paths):
    seen, unique = {}, []
    for p in paths:
        h = file_hash(p)
        if h not in seen:
            seen[h] = p
            unique.append(p)
    removed = len(paths) - len(unique)
    if removed:
        print(f"  Removed {removed} duplicates by content hash.")
    return unique

def balance_classes(train_splits, strategy, seed):
    counts = {label: len(df) for label, df in train_splits.items()}
    print(f"\n  Class counts before balancing: {counts}")
    if strategy == "none":
        return train_splits
    if strategy == "oversample":
        max_count = max(counts.values())
        balanced = {}
        for label, df in train_splits.items():
            if len(df) < max_count:
                extra = df.sample(n=max_count - len(df), replace=True, random_state=seed)
                balanced[label] = pd.concat([df, extra], ignore_index=True)
                print(f"  Oversampled {label}: {len(df)} -> {len(balanced[label])}")
            else:
                balanced[label] = df
        return balanced
    if strategy == "undersample":
        min_count = min(counts.values())
        balanced = {}
        for label, df in train_splits.items():
            balanced[label] = df.sample(n=min_count, random_state=seed) if len(df) > min_count else df
            print(f"  {label}: {len(df)} -> {len(balanced[label])}")
        return balanced

def build_dataset(csv_path, img_dir, out_dir, test_size, seed, balance):
    csv_path, img_dir, out_dir = Path(csv_path), Path(img_dir), Path(out_dir)
    print(f"\n[1/6] Loading metadata from {csv_path} ...")
    df = pd.read_csv(csv_path, sep="\t", on_bad_lines="skip", engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    label_col, img_col = "Disease_label", "Image_name"
    print(f"\n[2/6] Filtering eczema and psoriasis ...")
    df["_canonical"] = df[label_col].astype(str).apply(normalize_label)
    df_filtered = df[df["_canonical"].notna()].copy()
    print(f"  Eczema: {(df_filtered['_canonical']=='eczema').sum()}")
    print(f"  Psoriasis: {(df_filtered['_canonical']=='psoriasis').sum()}")
    print(f"  Total: {len(df_filtered)}")
    print(f"\n[3/6] Resolving image paths in {img_dir} ...")
    all_images = {}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
            all_images[p.name.lower()] = p
    print(f"  Found {len(all_images)} images on disk.")
    def find_image(img_name):
        img_name = str(img_name).strip().strip('"')
        if img_name.lower() in all_images:
            return all_images[img_name.lower()]
        for ext in (".jpg",".jpeg",".png"):
            if (img_name+ext).lower() in all_images:
                return all_images[(img_name+ext).lower()]
        return None
    df_filtered["_path"] = df_filtered[img_col].astype(str).apply(find_image)
    missing = df_filtered["_path"].isna().sum()
    if missing:
        print(f"  Warning: {missing} images not found on disk.")
    df_filtered = df_filtered[df_filtered["_path"].notna()].copy()
    print(f"  Images resolved: {len(df_filtered)}")
    print(f"\n[4/6] Deduplicating ...")
    deduped_rows = []
    for label in ("eczema","psoriasis"):
        subset = df_filtered[df_filtered["_canonical"]==label].copy()
        paths = [Path(p) for p in subset["_path"].tolist()]
        print(f"  {label}: {len(paths)} before dedup")
        unique_paths = dedup_paths(paths)
        path_set = set(str(p) for p in unique_paths)
        subset = subset[subset["_path"].apply(lambda p: str(p) in path_set)]
        deduped_rows.append(subset)
    df_final = pd.concat(deduped_rows, ignore_index=True)
    print(f"  Total after dedup: {len(df_final)}")
    print(f"\n[5/6] Splitting {int((1-test_size)*100)}/{int(test_size*100)} ...")
    train_splits, test_splits = {}, {}
    for label in ("eczema","psoriasis"):
        subset = df_final[df_final["_canonical"]==label]
        train_df, test_df = train_test_split(subset, test_size=test_size, random_state=seed)
        train_splits[label], test_splits[label] = train_df, test_df
        print(f"  {label}: {len(train_df)} train / {len(test_df)} test")
    train_splits = balance_classes(train_splits, balance, seed)
    print(f"\n[6/6] Copying files to {out_dir} ...")
    for split_name, splits in [("train", train_splits), ("test", test_splits)]:
        for label, df_part in splits.items():
            dst_dir = out_dir / split_name / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            seen_names = {}
            for _, row in df_part.iterrows():
                src = Path(row["_path"])
                name = src.name
                if name in seen_names:
                    seen_names[name] += 1
                    name = f"{src.stem}_{seen_names[src.name]}{src.suffix}"
                else:
                    seen_names[src.name] = 0
                shutil.copy2(src, dst_dir / name)
    print(f"\nDone! Dataset written to: {out_dir.resolve()}")
    for split_name in ("train","test"):
        for label in ("eczema","psoriasis"):
            d = out_dir / split_name / label
            count = len(list(d.glob("*"))) if d.exists() else 0
            print(f"  {split_name}/{label}: {count} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_dir", default="dataset_v2")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance", default="oversample", choices=["oversample","undersample","none"])
    args = parser.parse_args()
    build_dataset(args.csv, args.img_dir, args.out_dir, args.test_size, args.seed, args.balance)
