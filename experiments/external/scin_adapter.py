"""Harmonize SCIN and our internal cohort into a SHARED canonical metadata space
so a model trained internally can be validated externally on SCIN.

Why a canonical space: SCIN records body site as 12 coarse binary flags and age in
adult decade bands, while our cohort uses 47 fine-grained site terms and different
age bands. Neither maps onto the other directly, so both are projected onto a
common coarse taxonomy. Models must be RETRAINED in this space before external
validation (mapping a fine-grained model onto coarse inputs is not valid).

Outputs (repo root):
  external_scin_manifest.csv       SCIN cases: label, canonical metadata, FST, image paths
  internal_canonical_manifest.csv  our cohort in the identical feature space
  HARMONIZATION_REPORT.md          counts, coverage, and the crosswalks used
"""

from __future__ import annotations

import ast
import csv
import io
import re
from collections import Counter

from dermafair.paths import DATA_DIR, SCIN_METADATA_DIR, output_dir

REPORT_DIR = output_dir("external")
CACHE = SCIN_METADATA_DIR

# --------------------------------------------------------------------------- #
# Canonical spaces
# --------------------------------------------------------------------------- #
SITES = ["head_neck", "arm", "hand", "torso_front", "torso_back", "genital_groin", "buttocks", "leg", "foot", "other"]
AGES = ["under_20", "20_39", "40_59", "60_plus", "unknown"]
SEXES = ["male", "female", "unknown"]

# SCIN's 12 binary columns -> canonical site
SCIN_SITE = {
    "body_parts_head_or_neck": "head_neck",
    "body_parts_arm": "arm",
    "body_parts_palm": "hand",
    "body_parts_back_of_hand": "hand",
    "body_parts_torso_front": "torso_front",
    "body_parts_torso_back": "torso_back",
    "body_parts_genitalia_or_groin": "genital_groin",
    "body_parts_buttocks": "buttocks",
    "body_parts_leg": "leg",
    "body_parts_foot_top_or_side": "foot",
    "body_parts_foot_sole": "foot",
    "body_parts_other": "other",
}

# our free-text site terms -> canonical site (matched by keyword, first hit wins)
OUR_SITE_RULES = [
    ("sole", "foot"),
    ("plantar", "foot"),
    ("feet", "foot"),
    ("pedal", "foot"),
    ("toe", "foot"),
    ("toenail", "foot"),
    ("palm", "hand"),
    ("hand", "hand"),
    ("manus", "hand"),
    ("finger", "hand"),
    ("fingernail", "hand"),
    ("armpit", "other"),
    ("axillary", "other"),
    ("forearm", "arm"),
    ("antebrachium", "arm"),
    ("upper arm", "arm"),
    ("brachium", "arm"),
    ("elbow", "arm"),
    ("shoulder", "arm"),
    ("wrist", "arm"),
    ("carpus", "arm"),
    ("buttock", "buttocks"),
    ("gluteal", "buttocks"),
    ("genital", "genital_groin"),
    ("pubic", "genital_groin"),
    ("groin", "genital_groin"),
    ("inguinal", "genital_groin"),
    ("perianal", "genital_groin"),
    ("lowerback", "torso_back"),
    ("lumbus", "torso_back"),
    ("posterior", "torso_back"),
    ("back (dorsum)", "torso_back"),
    ("chest", "torso_front"),
    ("thorax", "torso_front"),
    ("abdomen", "torso_front"),
    ("navel", "torso_front"),
    ("umbilicus", "torso_front"),
    ("breast", "torso_front"),
    ("mammary", "torso_front"),
    ("anterior", "torso_front"),
    ("thigh", "leg"),
    ("femoral", "leg"),
    ("knee", "leg"),
    ("patellar", "leg"),
    ("popliteal", "leg"),
    ("calf", "leg"),
    ("calves", "leg"),
    ("sural", "leg"),
    ("ankle", "leg"),
    ("tarsal", "leg"),
    ("lower leg", "leg"),
    ("crural", "leg"),
    ("head", "head_neck"),
    ("scalp", "head_neck"),
    ("cheek", "head_neck"),
    ("forehead", "head_neck"),
    ("temple", "head_neck"),
    ("nose", "head_neck"),
    ("ear", "head_neck"),
    ("chin", "head_neck"),
    ("lip", "head_neck"),
    ("eye", "head_neck"),
    ("oral", "head_neck"),
    ("neck", "head_neck"),
    ("back", "torso_back"),  # generic 'Back' after the specific rules
]

SCIN_AGE = {
    "AGE_18_TO_29": "20_39",
    "AGE_30_TO_39": "20_39",
    "AGE_40_TO_49": "40_59",
    "AGE_50_TO_59": "40_59",
    "AGE_60_TO_69": "60_plus",
    "AGE_70_TO_79": "60_plus",
    "AGE_80_OR_ABOVE": "60_plus",
    "AGE_UNKNOWN": "unknown",
    "": "unknown",
}
OUR_AGE = {
    "0 - 10": "under_20",
    "10 - 20": "under_20",
    "20 - 40": "20_39",
    "40 - 60": "40_59",
    "60 - 80": "60_plus",
    "80 - 100": "60_plus",
}

ECZEMA_KEYS = ["eczema", "dermatitis"]
PSORIASIS_KEYS = ["psoriasis"]


def canon_site_ours(term: str) -> str:
    t = term.lower()
    for key, site in OUR_SITE_RULES:
        if key in t:
            return site
    return "other"


def parse_conditions(raw: str) -> list[str]:
    if not raw or raw.strip() in {"", "[]", "{}"}:
        return []
    try:
        val = ast.literal_eval(raw)
    except Exception:
        return [t.strip().strip("'\"") for t in re.split(r"[,\[\]]", raw) if t.strip()]
    if isinstance(val, dict):
        return [str(k) for k in val]
    if isinstance(val, (list, tuple)):
        return [str(v[0]) if isinstance(v, (list, tuple)) and v else str(v) for v in val]
    return [str(val)]


def family_of(condition: str):
    cl = condition.lower()
    if any(k in cl for k in PSORIASIS_KEYS):
        return "psoriasis"
    if any(k in cl for k in ECZEMA_KEYS):
        return "eczema"
    return None


def label_of(weighted_raw: str, min_weight: float = 0.0):
    """Derive the label from SCIN's WEIGHTED consensus, not the raw rating list.

    ``weighted_skin_condition_label`` is a {condition: weight} dict aggregating the
    dermatologists' differentials. The raw ``..._on_label_name`` list is per-rater
    and NOT ordered by confidence, so taking its first element mislabels cases.

    Returns (label, top_weight, competing) where ``competing`` marks cases in which
    the other family also carries appreciable weight.
    """
    try:
        wl = ast.literal_eval(weighted_raw) if weighted_raw and weighted_raw.strip() else {}
    except Exception:
        return None, 0.0, False
    if not isinstance(wl, dict) or not wl:
        return None, 0.0, False

    top_cond, top_w = max(wl.items(), key=lambda kv: float(kv[1]))
    fam = family_of(str(top_cond))
    if fam is None or float(top_w) < min_weight:
        return None, float(top_w), False

    # weight carried by the opposing family
    other = "eczema" if fam == "psoriasis" else "psoriasis"
    other_w = sum(float(w) for c, w in wl.items() if family_of(str(c)) == other)
    return fam, float(top_w), other_w >= 0.25


def multihot(active: set[str], vocab: list[str]) -> list[int]:
    return [1 if v in active else 0 for v in vocab]


def build_scin():
    cases = list(csv.DictReader(io.StringIO((CACHE / "scin_cases.csv").read_text(encoding="utf-8", errors="replace"))))
    labels = list(
        csv.DictReader(io.StringIO((CACHE / "scin_labels.csv").read_text(encoding="utf-8", errors="replace")))
    )
    by_case = {r["case_id"]: r for r in cases}

    rows, stats = [], Counter()
    for lr in labels:
        lab, top_w, competing = label_of(lr.get("weighted_skin_condition_label", ""))
        if lab is None:
            continue
        stats["matched"] += 1
        if competing:
            stats["competing_differential"] += 1
        if top_w < 0.5:
            stats["low_confidence"] += 1
        cr = by_case.get(lr["case_id"])
        if cr is None:
            stats["no_case_row"] += 1
            continue
        sites = {SCIN_SITE[c] for c in SCIN_SITE if (cr.get(c) or "").strip().upper() == "YES"}
        fst_raw = (cr.get("fitzpatrick_skin_type") or "").strip()
        m = re.search(r"(\d)", fst_raw)
        fst = int(m.group(1)) if m else -1
        sex = (cr.get("sex_at_birth") or "").strip().upper()
        sex = "male" if sex == "MALE" else ("female" if sex == "FEMALE" else "unknown")
        imgs = [cr.get(f"image_{i}_path", "") for i in (1, 2, 3)]
        rows.append(
            {
                "case_id": lr["case_id"],
                "label": 0 if lab == "eczema" else 1,
                "class": lab,
                "age": SCIN_AGE.get((cr.get("age_group") or "").strip(), "unknown"),
                "sex": sex,
                "fitzpatrick": fst,
                "sites": "|".join(sorted(sites)) if sites else "",
                "label_weight": round(top_w, 3),
                "competing": int(competing),
                "images": "|".join(p for p in imgs if p),
            }
        )
        stats[f"kept_{lab}"] += 1
        if fst > 0:
            stats["kept_with_fst"] += 1
    return rows, stats


def build_internal():
    subs = {
        "Inflammatory skin diseases (Eczema and Dermatitis)": ("eczema", 0),
        "Inflammatory skin diseases (Psoriasis and Lichenoid disorders)": ("psoriasis", 1),
    }
    keep = {r["image_name"] for r in csv.DictReader(open(DATA_DIR / "manifest_clean.csv", encoding="utf-8-sig"))}
    rows, unmapped = [], Counter()
    for r in csv.DictReader(open(DATA_DIR / "Skin_Metadata-1.csv", encoding="utf-8-sig")):
        if r["Sub_class"] not in subs or r["Image_name"] not in keep:
            continue
        cls, lab = subs[r["Sub_class"]]
        sites = set()
        for term in (r.get("Body_part") or "").split(","):
            if term.strip():
                s = canon_site_ours(term.strip())
                sites.add(s)
                if s == "other":
                    unmapped[term.strip()] += 1
        m = re.search(r"(\d)", r.get("Fitzpatrick") or "")
        rows.append(
            {
                "image_name": r["Image_name"],
                "subject": r["Subject_ID"],
                "label": lab,
                "class": cls,
                "age": OUR_AGE.get((r.get("Age") or "").strip(), "unknown"),
                "sex": (r.get("Sex") or "").strip().lower() or "unknown",
                "fitzpatrick": int(m.group(1)) if m else -1,
                "sites": "|".join(sorted(sites)) if sites else "",
            }
        )
    return rows, unmapped


def main():
    scin, stats = build_scin()
    internal, unmapped = build_internal()

    hdr_common = ["age", "sex", "fitzpatrick", "sites", "label", "class"]
    with open(DATA_DIR / "external_scin_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id"] + hdr_common + ["label_weight", "competing", "images"])
        w.writeheader()
        w.writerows(scin)
    with open(DATA_DIR / "internal_canonical_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_name", "subject"] + hdr_common)
        w.writeheader()
        w.writerows(internal)

    def dist(rows, key):
        return dict(Counter(r[key] for r in rows).most_common())

    def site_cov(rows):
        c = Counter()
        for r in rows:
            for s in r["sites"].split("|") if r["sites"] else []:
                c[s] += 1
        return c

    L = [
        "# SCIN <-> internal harmonization report\n",
        "Both cohorts projected onto a shared canonical space so a model trained internally can be",
        "validated on SCIN. Models must be RETRAINED in this space (coarse sites, coarse age bands).\n",
        f"Canonical sites ({len(SITES)}): {', '.join(SITES)}",
        f"Canonical age bands: {', '.join(AGES)}\n",
        "## Cohort sizes\n",
        "| | Internal | SCIN (external) |",
        "|---|---|---|",
        f"| cases | {len(internal)} | {len(scin)} |",
        f"| eczema | {sum(1 for r in internal if r['label'] == 0)} | {sum(1 for r in scin if r['label'] == 0)} |",
        f"| psoriasis | {sum(1 for r in internal if r['label'] == 1)} | {sum(1 for r in scin if r['label'] == 1)} |",
        f"| with Fitzpatrick | {sum(1 for r in internal if r['fitzpatrick'] > 0)} | {sum(1 for r in scin if r['fitzpatrick'] > 0)} |",
        f"\nSCIN cases excluded as ambiguous (both families in differential): {stats['excluded_ambiguous']}\n",
        "## Fitzpatrick coverage\n",
        f"- internal: {dict(sorted(Counter(r['fitzpatrick'] for r in internal).items()))}",
        f"- SCIN:     {dict(sorted(Counter(r['fitzpatrick'] for r in scin).items()))}",
        "\n> SCIN supplies FST 1-2, absent from the internal cohort, and substantially more FST 6.\n",
        "## Age / sex coverage\n",
        f"- internal age: {dist(internal, 'age')}",
        f"- SCIN age:     {dist(scin, 'age')}",
        f"- internal sex: {dist(internal, 'sex')}",
        f"- SCIN sex:     {dist(scin, 'sex')}",
        "\n## Canonical body-site coverage\n",
        "| site | internal | SCIN |",
        "|---|---|---|",
    ]
    si, ss = site_cov(internal), site_cov(scin)
    for s in SITES:
        L.append(f"| {s} | {si.get(s, 0)} | {ss.get(s, 0)} |")
    if unmapped:
        L.append("\n## Internal site terms that fell through to 'other' (review these)\n")
        for k, n in unmapped.most_common():
            L.append(f"- {k} ({n})")
    (REPORT_DIR / "HARMONIZATION_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n".join(L[6:30]))
    print(
        f"\nwrote external_scin_manifest.csv ({len(scin)}), internal_canonical_manifest.csv ({len(internal)}), HARMONIZATION_REPORT.md"
    )


if __name__ == "__main__":
    main()
