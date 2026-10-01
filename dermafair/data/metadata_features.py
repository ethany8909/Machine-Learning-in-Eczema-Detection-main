"""Clinical metadata encoding for the eczema/psoriasis task.

Turns the raw metadata CSV into a per-image feature vector for the metadata-only
model and the fusion branches. Encoded fields (matching the Week-3 plan):

    Age         -> one-hot of the age band string ("10 - 20", ...)
    Sex         -> one-hot (Male / Female / missing)
    Body_part   -> multi-hot over individual body regions (comma-separated)
    Descriptors -> multi-hot over individual morphologic descriptors

Deliberately EXCLUDED as features (to avoid target/skin-tone leakage):
    Fitzpatrick, Monk_skin_tone   -> sensitive attribute, not a predictor
    Disease_label, Main_class, Sub_class -> encode the label itself
    Confidence, Gradability, Quality     -> annotation-process fields

The vocabulary is fit on a supplied set of TRAIN image names only, then applied
to every image, so val/test categories never expand the feature space.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

MULTI_FIELDS = ["Body_part", "Descriptors"]
SINGLE_FIELDS = ["Age", "Sex"]

# Deployment-scenario spectrum (feature-ablation on the SAME images/split):
#   autonomous : demographics a patient app already knows
#   triage     : + body site a non-expert can report        (PRIMARY / recommended)
#   expert     : + morphologic descriptors a dermatologist logs (leaky ceiling)
# Each maps to (single_fields, multi_fields).
REGIMES = {
    "autonomous": (["Age", "Sex"], []),
    "triage": (["Age", "Sex"], ["Body_part"]),
    "expert": (["Age", "Sex"], ["Body_part", "Descriptors"]),
}


def regime_fields(name: str):
    if name not in REGIMES:
        raise KeyError(f"Unknown regime '{name}'. Options: {list(REGIMES)}")
    return REGIMES[name]


def _split_multi(value: str) -> list[str]:
    return [tok.strip() for tok in (value or "").split(",") if tok.strip()]


class MetadataEncoder:
    """Fit categorical vocab on train rows; transform any image into a fixed vector.

    ``single_fields``/``multi_fields`` select which metadata columns are visible —
    this is how the deployment-scenario spectrum is created (same data, different
    feature access). Defaults to the full field set for backward compatibility.
    """

    def __init__(self, single_fields=None, multi_fields=None):
        self.single_fields = list(SINGLE_FIELDS if single_fields is None else single_fields)
        self.multi_fields = list(MULTI_FIELDS if multi_fields is None else multi_fields)
        self.single_vocab: dict[str, list[str]] = {}
        self.multi_vocab: dict[str, list[str]] = {}
        self.feature_names: list[str] = []
        self._rows: dict[str, dict] = {}

    @classmethod
    def for_regime(cls, name: str) -> "MetadataEncoder":
        single, multi = regime_fields(name)
        return cls(single_fields=single, multi_fields=multi)

    def load_rows(self, metadata_csv: str | Path) -> "MetadataEncoder":
        with open(metadata_csv, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                self._rows[row["Image_name"]] = row
        return self

    def fit(self, train_image_names) -> "MetadataEncoder":
        train_names = [n for n in train_image_names if n in self._rows]
        for col in self.single_fields:
            vals = sorted({(self._rows[n].get(col) or "missing").strip() or "missing"
                           for n in train_names})
            self.single_vocab[col] = vals
        for col in self.multi_fields:
            toks = set()
            for n in train_names:
                toks.update(_split_multi(self._rows[n].get(col, "")))
            self.multi_vocab[col] = sorted(toks)

        names = []
        for col in self.single_fields:
            names += [f"{col}={v}" for v in self.single_vocab[col]]
        for col in self.multi_fields:
            names += [f"{col}:{v}" for v in self.multi_vocab[col]]
        self.feature_names = names
        return self

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def transform_one(self, image_name: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        row = self._rows.get(image_name)
        if row is None:
            return vec
        i = 0
        for col in self.single_fields:
            val = (row.get(col) or "missing").strip() or "missing"
            for v in self.single_vocab[col]:
                if v == val:
                    vec[i] = 1.0
                i += 1
        for col in self.multi_fields:
            present = set(_split_multi(row.get(col, "")))
            for v in self.multi_vocab[col]:
                if v in present:
                    vec[i] = 1.0
                i += 1
        return vec

    def build_lookup(self) -> dict[str, np.ndarray]:
        """image_name -> feature vector, for every row loaded."""
        return {name: self.transform_one(name) for name in self._rows}
