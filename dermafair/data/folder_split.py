"""Dataset adapter for the folder-based eczema/psoriasis split.

Reads the ImageFolder-style layout produced by ``split_dataset.py``::

    dataset_split/
      train/{eczema_dermatitis, psoriasis_lichenoid}/*.jpg
      val/...
      test/...

and joins each image's Fitzpatrick skin-tone band from the metadata CSV so that
image-only models can still be evaluated for skin-tone fairness.

Label convention (alphabetical, matches torchvision ImageFolder):
    eczema_dermatitis   -> 0
    psoriasis_lichenoid -> 1   (positive class for the fairness metrics)

Each batch item is a dict:
    image        : FloatTensor [3, H, W]
    label        : LongTensor  []
    fitzpatrick  : int   (3-6, or -1 if missing)
    image_name   : str
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASS_TO_LABEL = {"eczema_dermatitis": 0, "psoriasis_lichenoid": 1}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_fitzpatrick_map(metadata_csv: Path) -> dict[str, int]:
    """image_name -> Fitzpatrick band int (3-6); -1 if missing/unparseable."""
    fmap: dict[str, int] = {}
    with open(metadata_csv, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("Fitzpatrick") or "").strip()  # e.g. "FST 4"
            band = -1
            for tok in raw.replace("FST", "").split():
                if tok.isdigit():
                    band = int(tok)
                    break
            fmap[row["Image_name"]] = band
    return fmap


def _build_transforms(image_size: int, augment: bool):
    from torchvision import transforms

    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1),  # mild — preserve skin-tone signal
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        norm,
    ])


class FolderSplitDataset(Dataset):
    def __init__(self, split_dir: Path, fitz_map: dict[str, int], transform):
        self.transform = transform
        self.samples: list[tuple[Path, int, int, str]] = []  # path, label, fitz, name
        for cls, label in CLASS_TO_LABEL.items():
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.is_file():
                    self.samples.append((f, label, fitz_map.get(f.name, -1), f.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, fitz, name = self.samples[i]
        img = self.transform(Image.open(path).convert("RGB"))
        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "fitzpatrick": int(fitz),
            "image_name": name,
        }


def _list_split_names(split_dir: Path) -> list[str]:
    names = []
    for cls in CLASS_TO_LABEL:
        d = split_dir / cls
        if d.exists():
            names += [f.name for f in sorted(d.iterdir()) if f.is_file()]
    return names


class MultimodalFolderDataset(Dataset):
    """Yields image + metadata + label + fitzpatrick.

    With ``load_images=False`` a 1x1 placeholder tensor is returned instead of
    decoding the JPEG — used to train the metadata-only model quickly.
    """

    def __init__(self, split_dir: Path, fitz_map, meta_lookup, transform,
                 load_images: bool = True):
        self.transform = transform
        self.meta_lookup = meta_lookup
        self.load_images = load_images
        self.meta_dim = len(next(iter(meta_lookup.values()))) if meta_lookup else 0
        self.samples = []
        for cls, label in CLASS_TO_LABEL.items():
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.is_file():
                    self.samples.append((f, label, fitz_map.get(f.name, -1), f.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, fitz, name = self.samples[i]
        if self.load_images:
            img = self.transform(Image.open(path).convert("RGB"))
        else:
            img = torch.zeros(1)
        meta = self.meta_lookup.get(name)
        meta = torch.tensor(meta if meta is not None else [0.0] * self.meta_dim,
                            dtype=torch.float32)
        return {
            "image": img,
            "meta": meta,
            "label": torch.tensor(label, dtype=torch.long),
            "fitzpatrick": int(fitz),
            "image_name": name,
        }


def build_multimodal_dataloaders(
    split_root: str | Path,
    metadata_csv: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    load_images: bool = True,
    regime: str | None = None,
):
    """Return (train, val, test, meta_dim) with image+metadata batches.

    Metadata vocabulary is fit on the TRAIN split only. ``regime`` selects the
    deployment-scenario feature set (autonomous | triage | expert); None = full set.
    """
    from dermafair.data.metadata_features import MetadataEncoder

    split_root = Path(split_root)
    fitz_map = _load_fitzpatrick_map(Path(metadata_csv))

    encoder = (MetadataEncoder.for_regime(regime) if regime else MetadataEncoder())
    encoder.load_rows(metadata_csv)
    encoder.fit(_list_split_names(split_root / "train"))
    meta_lookup = encoder.build_lookup()

    train_t = _build_transforms(image_size, augment=True)
    eval_t = _build_transforms(image_size, augment=False)

    def mk(split, tfm):
        return MultimodalFolderDataset(split_root / split, fitz_map, meta_lookup,
                                       tfm, load_images=load_images)

    train_ds, val_ds, test_ds = mk("train", train_t), mk("val", eval_t), mk("test", eval_t)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        encoder.dim,
    )


class ListImageDataset(Dataset):
    """Image dataset from an explicit list of (path, label, fitzpatrick, name)."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, fitz, name = self.samples[i]
        img = self.transform(Image.open(path).convert("RGB"))
        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "fitzpatrick": int(fitz),
            "image_name": name,
        }


class MultimodalListDataset(Dataset):
    """Image + metadata dataset from an explicit sample list (for fusion CV)."""

    def __init__(self, samples, meta_lookup, transform, load_images: bool = True):
        self.samples = samples
        self.meta_lookup = meta_lookup
        self.transform = transform
        self.load_images = load_images
        self.meta_dim = len(next(iter(meta_lookup.values()))) if meta_lookup else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, fitz, name = self.samples[i]
        img = self.transform(Image.open(path).convert("RGB")) if self.load_images else torch.zeros(1)
        meta = self.meta_lookup.get(name)
        meta = torch.tensor(meta if meta is not None else [0.0] * self.meta_dim, dtype=torch.float32)
        return {
            "image": img,
            "meta": meta,
            "label": torch.tensor(label, dtype=torch.long),
            "fitzpatrick": int(fitz),
            "image_name": name,
        }


def build_cv_multimodal_from_manifest(
    manifest_csv: str | Path,
    source_dirs,
    metadata_csv: str | Path,
    test_fold: int,
    k: int,
    regime: str | None = None,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    load_images: bool = True,
):
    """Fusion-CV fold loaders: image + metadata, with the metadata encoder fit on
    THIS fold's train split (regime-specific). test=fold, val=(fold+1)%k, train=rest.
    Returns (train, val, test, meta_dim).
    """
    import csv as _csv
    from dermafair.data.metadata_features import MetadataEncoder

    file_index = {}
    for d in source_dirs:
        for f in Path(d).iterdir():
            if f.is_file():
                file_index[f.name] = f

    val_fold = (test_fold + 1) % k
    buckets = {"train": [], "val": [], "test": []}
    with open(manifest_csv, encoding="utf-8-sig", newline="") as fh:
        for r in _csv.DictReader(fh):
            fold = int(r["fold"])
            where = "test" if fold == test_fold else ("val" if fold == val_fold else "train")
            p = file_index.get(r["image_name"])
            if p is None:
                continue
            buckets[where].append((p, int(r["label"]), int(r["fitzpatrick"]), r["image_name"]))

    enc = (MetadataEncoder.for_regime(regime) if regime else MetadataEncoder()).load_rows(metadata_csv)
    enc.fit([s[3] for s in buckets["train"]])           # fit on this fold's train only
    meta_lookup = enc.build_lookup()

    train_t = _build_transforms(image_size, augment=True)
    eval_t = _build_transforms(image_size, augment=False)

    def mk(split, tfm):
        return MultimodalListDataset(buckets[split], meta_lookup, tfm, load_images=load_images)

    return (
        DataLoader(mk("train", train_t), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(mk("val", eval_t), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(mk("test", eval_t), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        enc.dim,
    )


def build_cv_dataloaders_from_manifest(
    manifest_csv: str | Path,
    source_dirs,
    test_fold: int,
    k: int,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Cross-validation fold loaders from a fold-assigned manifest.

    For ``test_fold`` i: test = fold i, val = fold (i+1)%k, train = the rest.
    Images are located by name across ``source_dirs`` (e.g. DATASET_0/1).
    Returns (train_loader, val_loader, test_loader).
    """
    import csv as _csv

    file_index = {}
    for d in source_dirs:
        for f in Path(d).iterdir():
            if f.is_file():
                file_index[f.name] = f

    val_fold = (test_fold + 1) % k
    buckets = {"train": [], "val": [], "test": []}
    with open(manifest_csv, encoding="utf-8-sig", newline="") as fh:
        for row in _csv.DictReader(fh):
            fold = int(row["fold"])
            where = "test" if fold == test_fold else ("val" if fold == val_fold else "train")
            p = file_index.get(row["image_name"])
            if p is None:
                continue
            buckets[where].append((p, int(row["label"]), int(row["fitzpatrick"]), row["image_name"]))

    train_t = _build_transforms(image_size, augment=True)
    eval_t = _build_transforms(image_size, augment=False)
    train_ds = ListImageDataset(buckets["train"], train_t)
    val_ds = ListImageDataset(buckets["val"], eval_t)
    test_ds = ListImageDataset(buckets["test"], eval_t)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def build_dataloaders_from_folders(
    split_root: str | Path,
    metadata_csv: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Return (train_loader, val_loader, test_loader).

    ``num_workers`` defaults to 0 for safe single-process loading on Windows/CPU.
    """
    split_root = Path(split_root)
    fitz_map = _load_fitzpatrick_map(Path(metadata_csv))

    train_t = _build_transforms(image_size, augment=True)
    eval_t = _build_transforms(image_size, augment=False)

    train_ds = FolderSplitDataset(split_root / "train", fitz_map, train_t)
    val_ds = FolderSplitDataset(split_root / "val", fitz_map, eval_t)
    test_ds = FolderSplitDataset(split_root / "test", fitz_map, eval_t)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
