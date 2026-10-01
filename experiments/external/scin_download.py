"""Download SCIN images for the harmonized eczema/psoriasis subset.

Fetches the FIRST image per case (matching our one-image-per-record structure),
~1,128 files / roughly 1 GB. Resumable: existing files are skipped, so the script
can be re-run safely after an interruption.

    python experiments/external/scin_download.py [--all-images] [--workers 6]
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dermafair.paths import DATA_DIR

BUCKET = "https://storage.googleapis.com/dx-scin-public-data/"
OUT = DATA_DIR / "scin_images"
MANIFEST = DATA_DIR / "external_scin_manifest.csv"

_lock = threading.Lock()
_done = _skip = _fail = 0


def fetch(task):
    global _done, _skip, _fail
    case_id, path = task
    dest = OUT / f"{case_id}__{Path(path).name}"
    if dest.exists() and dest.stat().st_size > 0:
        with _lock:
            _skip += 1
        return
    try:
        req = urllib.request.Request(BUCKET + path, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
        dest.write_bytes(data)
        with _lock:
            _done += 1
    except Exception as e:
        with _lock:
            _fail += 1
        print(f"  FAIL {path}: {type(e).__name__}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-images", action="store_true", help="fetch every image, not just the first per case")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    tasks = []
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        paths = [p for p in r["images"].split("|") if p]
        for p in paths if args.all_images else paths[:1]:
            tasks.append((r["case_id"], p))

    print(f"SCIN download: {len(tasks)} images -> {OUT}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(fetch, tasks), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)}  downloaded={_done} skipped={_skip} failed={_fail}", flush=True)

    total = sum(f.stat().st_size for f in OUT.iterdir() if f.is_file())
    print(f"DONE downloaded={_done} skipped={_skip} failed={_fail} | {total / 1e9:.2f} GB in {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
