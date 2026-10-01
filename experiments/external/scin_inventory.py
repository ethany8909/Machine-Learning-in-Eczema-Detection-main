"""Inventory the SCIN dataset's METADATA ONLY (no images) to decide whether it is
viable as an external-validation cohort for the eczema vs psoriasis task.

Reports:
  - how many cases carry an eczema / psoriasis dermatologist label
  - which of our triage metadata fields (age, sex, body site) exist
  - the Fitzpatrick distribution (for fairness validation)
  - body-site vocabulary, for building a crosswalk to our encoder

Downloads two CSVs (a few MB) to the scratchpad; no image data is fetched.
"""

from __future__ import annotations

import ast
import csv
import io
import re
import sys
import urllib.request
from collections import Counter

from dermafair.paths import SCIN_METADATA_DIR

BASE = "https://storage.googleapis.com/dx-scin-public-data/dataset/"
CASES, LABELS = "scin_cases.csv", "scin_labels.csv"
CACHE = SCIN_METADATA_DIR
CACHE.mkdir(parents=True, exist_ok=True)

ECZEMA_TERMS = ["eczema", "atopic dermatitis", "dermatitis"]
PSORIASIS_TERMS = ["psoriasis"]


def fetch(name: str) -> list[dict]:
    dest = CACHE / name
    if not dest.exists():
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
    text = dest.read_text(encoding="utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def parse_conditions(raw: str) -> list[str]:
    """SCIN dermatologist labels are stored as a stringified list/dict of conditions."""
    if not raw or raw.strip() in {"", "[]", "{}"}:
        return []
    try:
        val = ast.literal_eval(raw)
    except Exception:
        return [t.strip().strip("'\"") for t in re.split(r"[,\[\]]", raw) if t.strip()]
    if isinstance(val, dict):
        return [str(k) for k in val]
    if isinstance(val, (list, tuple)):
        out = []
        for v in val:
            out.append(str(v[0]) if isinstance(v, (list, tuple)) and v else str(v))
        return out
    return [str(val)]


def classify(conds: list[str]) -> set[str]:
    tags = set()
    for c in conds:
        cl = c.lower()
        if any(t in cl for t in PSORIASIS_TERMS):
            tags.add("psoriasis")
        elif any(t in cl for t in ECZEMA_TERMS):
            tags.add("eczema")
    return tags


def main():
    print("downloading SCIN metadata (no images)...")
    cases = fetch(CASES)
    labels = fetch(LABELS)
    print(f"cases={len(cases)}  labels={len(labels)}\n")

    # ---- which of our triage fields exist? ----
    cols = set(cases[0])
    print("=== metadata field availability (our triage regime) ===")
    wanted = {
        "age": [c for c in cols if "age" in c.lower()],
        "sex": [c for c in cols if "sex" in c.lower() or "gender" in c.lower()],
        "body site": [c for c in cols if "body" in c.lower() or "part" in c.lower() or "location" in c.lower()],
        "skin tone": [c for c in cols if "fitzpatrick" in c.lower() or "monk" in c.lower()],
    }
    for k, v in wanted.items():
        print(f"  {k:10} -> {v if v else 'NOT FOUND'}")

    # ---- disease counts from dermatologist labels ----
    label_col = next((c for c in labels[0] if "skin_condition" in c and "name" in c), None)
    print(f"\n=== disease inventory (label column: {label_col}) ===")
    tally, examples = Counter(), Counter()
    case_tags = {}
    for r in labels:
        conds = parse_conditions(r.get(label_col, ""))
        for c in conds:
            examples[c] += 1
        tags = classify(conds)
        if tags:
            case_tags[r["case_id"]] = tags
        for t in tags:
            tally[t] += 1
    print(f"  cases with an ECZEMA-family label   : {tally['eczema']}")
    print(f"  cases with a PSORIASIS label        : {tally['psoriasis']}")
    both = sum(1 for t in case_tags.values() if len(t) > 1)
    print(f"  cases tagged with both (ambiguous)  : {both}")
    print("\n  top 15 conditions overall:")
    for c, n in examples.most_common(15):
        print(f"    {n:5}  {c}")

    # ---- Fitzpatrick distribution for the usable subset ----
    fst_col = next((c for c in cols if "fitzpatrick" in c.lower()), None)
    if fst_col:
        by_case = {r["case_id"]: r for r in cases}
        fst = Counter()
        for cid in case_tags:
            row = by_case.get(cid)
            if row:
                fst[(row.get(fst_col) or "missing").strip() or "missing"] += 1
        print(f"\n=== Fitzpatrick distribution among eczema/psoriasis cases ({fst_col}) ===")
        for k, n in sorted(fst.items(), key=lambda kv: str(kv[0])):
            print(f"  {k:28} {n}")

    # ---- body-site vocabulary ----
    site_cols = wanted["body site"]
    if site_cols:
        col = site_cols[0]
        vocab = Counter()
        by_case = {r["case_id"]: r for r in cases}
        for cid in case_tags:
            row = by_case.get(cid)
            if row:
                for tok in parse_conditions(row.get(col, "")) or [row.get(col, "")]:
                    if tok and tok.strip():
                        vocab[tok.strip()] += 1
        print(f"\n=== body-site vocabulary ({col}), top 20 of {len(vocab)} ===")
        for k, n in vocab.most_common(20):
            print(f"  {n:5}  {k}")

    print("\nVERDICT INPUTS: psoriasis count is the gating number for fusion validation.")


if __name__ == "__main__":
    sys.exit(main())
