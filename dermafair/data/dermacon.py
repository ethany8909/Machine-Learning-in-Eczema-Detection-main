"""DermaCon-IN dataset adapter.

Reference implementation. Adapt the column names in ``configs/dermacon.yaml`` and
the feature-encoding block below to match the exact DermaCon-IN metadata schema.

Provides ``build_dataloaders(cfg) -> (train, val, test, meta_dim)``.

Each batch is a dict:
    image        : FloatTensor [B, 3, H, W]
    meta         : FloatTensor [B, F]
    label        : LongTensor  [B]
    fitzpatrick  : ndarray     [B]   (kept as numpy for grouping)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class DermaConDataset(Dataset):
    def __init__(self, df, indices, image_root, image_col, meta_matrix,
                 sensitive_attr, transform):
        self.df = df.loc[indices].reset_index(drop=True)
        self.meta = meta_matrix[df.index.get_indexer(indices)]
        self.image_root = Path(image_root)
        self.image_col = image_col
        self.sensitive_attr = sensitive_attr
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = self.image_root / row[self.image_col]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return {
            "image": img,
            "meta": torch.tensor(self.meta[i], dtype=torch.float32),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "fitzpatrick": int(row[self.sensitive_attr]),
        }


def _build_transforms(cfg):
    from torchvision import transforms

    p = cfg["preprocess"]
    size = p["image_size"]
    norm = transforms.Normalize(p["normalize_mean"], p["normalize_std"])

    train_t = [transforms.Resize((size, size))]
    aug = p.get("augment", {})
    if aug.get("horizontal_flip"):
        train_t.append(transforms.RandomHorizontalFlip())
    if aug.get("rotation_degrees"):
        train_t.append(transforms.RandomRotation(aug["rotation_degrees"]))
    if aug.get("color_jitter"):
        cj = aug["color_jitter"]
        train_t.append(transforms.ColorJitter(cj, cj, cj))
    train_t += [transforms.ToTensor(), norm]

    eval_t = [transforms.Resize((size, size)), transforms.ToTensor(), norm]
    return transforms.Compose(train_t), transforms.Compose(eval_t)


def _encode_metadata(df, feature_cols):
    """One-hot encode categoricals, standardize numerics. Returns [N, F] matrix."""
    frames = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        s = df[col]
        if s.dtype.kind in "if":  # numeric
            vals = s.fillna(s.median())
            std = vals.std() or 1.0
            frames.append(((vals - vals.mean()) / std).to_frame(col))
        else:  # categorical
            frames.append(pd.get_dummies(s.fillna("missing"), prefix=col))
    if not frames:
        raise ValueError("No usable metadata features found.")
    mat = pd.concat(frames, axis=1).to_numpy(dtype=np.float32)
    return mat


def build_dataloaders(cfg):
    dcfg = cfg["data"]
    processed = Path(dcfg["processed_dir"])
    df = pd.read_parquet(processed / "dataset.parquet")
    with open(processed / "splits.pkl", "rb") as f:
        splits = pickle.load(f)

    meta_matrix = _encode_metadata(df, dcfg["metadata_features"])
    meta_dim = meta_matrix.shape[1]

    train_t, eval_t = _build_transforms(cfg)
    common = dict(
        df=df, image_root=dcfg["raw_dir"], image_col=dcfg["image_col"],
        meta_matrix=meta_matrix, sensitive_attr=dcfg["sensitive_attr"],
    )
    train_ds = DermaConDataset(indices=splits["train"], transform=train_t, **common)
    val_ds = DermaConDataset(indices=splits["val"], transform=eval_t, **common)
    test_ds = DermaConDataset(indices=splits["test"], transform=eval_t, **common)

    bs = cfg["train"]["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=False),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2),
        meta_dim,
    )
