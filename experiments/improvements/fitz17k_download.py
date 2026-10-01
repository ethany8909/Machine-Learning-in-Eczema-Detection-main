"""Download the Fitzpatrick17k images needed for the second external test set.

Only the psoriasis and eczema/dermatitis-type labels are fetched (every label the SCIN
keyword rule could match), from the atlas URLs listed in fitzpatrick17k.csv
(github.com/mattgroh/fitzpatrick17k, CC BY-NC-SA 3.0). Some links are known to be dead:
in September 2026 dermaamin.com no longer resolved, leaving the Atlas Dermatologico images.
Failures are logged, not retried forever. Requests are throttled to stay polite.

Outputs (under DERMAFAIR_DATA_DIR): fitzpatrick17k/fitzpatrick17k.csv,
fitzpatrick17k/images/<md5hash>.jpg, fitzpatrick17k/download_log.csv

    python experiments/improvements/fitz17k_download.py
"""

from __future__ import annotations

import csv
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from common import ROOT
from PIL import Image

F17 = ROOT / "fitzpatrick17k"
IMG = F17 / "images"
IMG.mkdir(parents=True, exist_ok=True)
LABELS_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
PATTERN = "psoria|eczema|dermatitis"
HEADERS = {"User-Agent": "Mozilla/5.0 (research download of the Fitzpatrick17k dataset)"}
_lock = threading.Lock()
_last = [0.0]


def _throttle(gap=0.25):
    with _lock:
        wait = _last[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def fetch(row) -> tuple[str, str]:
    out = IMG / f"{row.md5hash}.jpg"
    if out.exists():
        return row.md5hash, "ok"
    for attempt in range(3):
        try:
            _throttle()
            r = requests.get(row.url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                status = f"http {r.status_code}"
                if r.status_code in (403, 404, 410):
                    return row.md5hash, status
                time.sleep(2 * (attempt + 1))
                continue
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            im.save(out, quality=95)
            return row.md5hash, "ok"
        except Exception as e:  # noqa: BLE001 - log and move on
            status = type(e).__name__
            if "getaddrinfo" in str(e) or "NameResolution" in str(e):
                return row.md5hash, "host offline"  # dermaamin.com no longer resolves (Sept 2026)
            time.sleep(2 * (attempt + 1))
    return row.md5hash, status


def main():
    labels = F17 / "fitzpatrick17k.csv"
    if not labels.exists():
        r = requests.get(LABELS_URL, headers=HEADERS, timeout=60)
        r.raise_for_status()
        labels.write_bytes(r.content)
    d = pd.read_csv(labels)
    d = d[d["label"].str.contains(PATTERN, case=False)].drop_duplicates("md5hash")
    print(f"{len(d)} images to fetch", flush=True)
    results = {}
    with ThreadPoolExecutor(4) as ex:
        for i, (h, s) in enumerate(ex.map(fetch, d.itertuples()), 1):
            results[h] = s
            if i % 100 == 0:
                ok = sum(v == "ok" for v in results.values())
                print(f"{time.strftime('%H:%M:%S')} {i}/{len(d)}  ok {ok}", flush=True)
    with open(F17 / "download_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["md5hash", "label", "status"])
        for r in d.itertuples():
            w.writerow([r.md5hash, r.label, results[r.md5hash]])
    ok = sum(v == "ok" for v in results.values())
    print(f"done: {ok}/{len(d)} downloaded", flush=True)
    print(pd.Series(results).value_counts().to_string())


if __name__ == "__main__":
    main()
