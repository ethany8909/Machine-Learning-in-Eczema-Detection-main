"""Table S1 — dataset composition: class x Fitzpatrick (and Monk) contingency tables,
computed over the filtered eczema/psoriasis set and broken down by train/val/test split.
Pure stdlib so it runs without the ML stack installed.
"""
import csv
import os
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Skin_Metadata-1.csv"
SPLIT_DIR = ROOT / "dataset_split"

CLASS_MAP = {
    "Inflammatory skin diseases (Eczema and Dermatitis)": "eczema_dermatitis",
    "Inflammatory skin diseases (Psoriasis and Lichenoid disorders)": "psoriasis_lichenoid",
}
FST_ORDER = ["FST 3", "FST 4", "FST 5", "FST 6", "(missing)"]

# Which split each image landed in (from the folders already created)
split_of_image = {}
for split in ("train", "val", "test"):
    for cls in CLASS_MAP.values():
        d = SPLIT_DIR / split / cls
        if d.exists():
            for f in d.iterdir():
                split_of_image[f.name] = split

rows = []
with open(CSV_PATH, encoding="utf-8-sig", newline="") as fh:
    for r in csv.DictReader(fh):
        cls = CLASS_MAP.get(r["Sub_class"])
        if cls is None:
            continue
        fst = r["Fitzpatrick"].strip() or "(missing)"
        rows.append((r["Image_name"], cls, fst, split_of_image.get(r["Image_name"], "?")))


def contingency(records, out_lines, title):
    out_lines.append(f"\n### {title}\n")
    table = defaultdict(Counter)  # class -> fst -> count
    for _, cls, fst, _ in records:
        table[cls][fst] += 1
    fsts = [f for f in FST_ORDER if any(table[c][f] for c in CLASS_MAP.values())]
    header = "| Class | " + " | ".join(fsts) + " | **Total** |"
    sep = "|" + "---|" * (len(fsts) + 2)
    out_lines.append(header)
    out_lines.append(sep)
    col_tot = Counter()
    grand = 0
    for cls in CLASS_MAP.values():
        cells = [table[cls][f] for f in fsts]
        tot = sum(cells)
        grand += tot
        for f, c in zip(fsts, cells):
            col_tot[f] += c
        out_lines.append(f"| {cls} | " + " | ".join(str(c) for c in cells) + f" | **{tot}** |")
    out_lines.append("| **Total** | " + " | ".join(f"**{col_tot[f]}**" for f in fsts) + f" | **{grand}** |")


lines = ["# Table S1 — Dataset Composition (Eczema/Dermatitis vs Psoriasis/Lichenoid)\n"]
lines.append(f"Total images: **{len(rows)}**")
contingency(rows, lines, "Overall: Class x Fitzpatrick")
for split in ("train", "val", "test"):
    contingency([r for r in rows if r[3] == split], lines, f"{split.capitalize()} split: Class x Fitzpatrick")

out = ROOT / "TABLE_S1_composition.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nWritten to {out}")
