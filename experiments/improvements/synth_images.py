"""Image-space synthetic training data from a pretrained generator (SD-Turbo, Stability AI).

The earlier synthetic-data experiments generated FEATURES; these generate actual images,
aimed at the scarcest cell in the training data: psoriasis on darker skin (40 real images,
Fitzpatrick V-VI). Arms:

  tone     every real Fitzpatrick III-IV image re-rendered on darker skin: the skin pixels are
           darkened to a skin tone drawn from the real V-VI photos (CIELAB transform), then the
           generator makes a light image-to-image pass so the result looks photographic
  color    the same darkening with NO generator pass (ablation: what does the generator add?)
  neutral  the generator pass alone, with no darkening and a neutral prompt (control for
           "any re-rendered image")
  text     text-to-image from scratch: psoriasis / eczema on dark skin, equal numbers per class,
           which tests whether a general-purpose image model knows what these diseases look like

Pilot 1 (prompt-only image-to-image) erased the lesions even at the lowest usable strength
(healthy-looking skin carrying a disease label), so the darkening is now done explicitly and
the generator is used only at low noise (strength 0.2 = one denoising step from t=199).

The generator is pretrained and never trained on our photos, so there is no fold leakage:
a derived image inherits its source's fold and label. Derived images come from patient
photos and stay local (results/improvements/synthetic/ is git-ignored).

    python experiments/improvements/synth_images.py pilot2
    python experiments/improvements/synth_images.py tone color neutral text --strength 0.2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from common import CACHE, IMP, load_cohorts  # noqa: E402
from PIL import Image

from dermafair.data.skin_tone import darken, ita  # noqa: F401  (ita is re-exported to synth_eval)

SYN = IMP / "synthetic"
RES = 512
TONE_PROMPTS = [
    "clinical photograph of a skin rash on dark brown skin",
    "clinical photograph of a skin rash on deeply pigmented dark brown skin, Fitzpatrick skin type V",
]
TONE_PROMPT = "close-up clinical photograph of a skin rash on dark brown skin"
NEUTRAL_PROMPT = "close-up clinical photograph of a skin rash"
TEXT_DISEASE = {1: ["plaque psoriasis", "psoriasis"], 0: ["eczema", "atopic dermatitis"]}
TEXT_SITES = ["elbow", "knee", "scalp", "lower back", "hand", "forearm", "shin", "trunk", "neck", "foot"]
TEXT_TONES = ["dark brown", "black"]
TEXT_PER_CLASS = 240


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def seed_of(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def dark_ita_pool() -> np.ndarray:
    """Skin ITA of every real Fitzpatrick V-VI photo in the internal cohort: darkening
    targets are drawn from this, so synthetic skin tones match real darker-skin photos."""
    f = SYN / "dark_ita_pool.npy"
    if f.exists():
        return np.load(f)
    df = load_cohorts().internal
    cache = np.load(CACHE / "internal_256.npy")
    pool = np.array([ita(cache[i]) for i in np.where((df["fitzpatrick"] >= 5).to_numpy())[0]])
    SYN.mkdir(parents=True, exist_ok=True)
    np.save(f, pool)
    return pool


def target_for(name: str, pool: np.ndarray) -> float:
    return float(np.random.default_rng(seed_of(name)).choice(pool))


def load_src(path: str) -> Image.Image:
    im = Image.open(path)
    im.draft("RGB", (RES * 2, RES * 2))
    return im.convert("RGB").resize((RES, RES), Image.BICUBIC)


def pipelines():
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

    torch.set_num_threads(12)
    t2i = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", variant="fp16", torch_dtype=torch.float32)
    t2i.set_progress_bar_config(disable=True)
    i2i = AutoPipelineForImage2Image.from_pipe(t2i)
    return t2i, i2i


def img2img(i2i, src: Image.Image, prompt: str, strength: float, seed: int) -> Image.Image:
    steps = math.ceil(1 / strength)  # SD-Turbo needs steps * strength >= 1 (one denoising step)
    g = torch.Generator().manual_seed(seed)
    return i2i(
        prompt=prompt, image=src, num_inference_steps=steps, strength=strength, guidance_scale=0.0, generator=g
    ).images[0]


def sources(df):
    """Every harmonized-label image photographed on Fitzpatrick III-IV skin."""
    return df[df["harmonized"].notna() & df["fitzpatrick"].between(1, 4)]


# --------------------------------------------------------------------------- #
def pilot(strengths=(0.4, 0.5, 0.6, 0.75), n_per_class=6):
    """Pick the image-to-image strength from skin tone and content preservation only
    (no classifier is trained or scored here)."""
    from extract_features import dinov2, encode

    df = load_cohorts().internal
    cache = np.load(CACHE / "internal_256.npy")
    ref = {
        g: float(np.median([ita(cache[i]) for i in np.where(m)[0]]))
        for g, m in (
            ("real III-IV", df["fitzpatrick"].between(1, 4).to_numpy()),
            ("real V-VI", (df["fitzpatrick"] >= 5).to_numpy()),
        )
    }
    log("reference median ITA:", {k: round(v, 1) for k, v in ref.items()})
    src = sources(df)
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.where(src["harmonized"] == c)[0], n_per_class, replace=False) for c in (0, 1)])
    rows = src.iloc[pick]
    _, i2i = pipelines()
    f = dinov2("b")
    out_dir = SYN / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    srcs = [load_src(p) for p in rows["path"]]
    src256 = np.stack([np.asarray(s.resize((256, 256), Image.BICUBIC)) for s in srcs])
    e_src = encode(f, src256)
    e_src /= np.linalg.norm(e_src, axis=1, keepdims=True)
    table, grid = [], [src256]
    for pi, prompt in enumerate(TONE_PROMPTS):
        for s in strengths:
            outs = [img2img(i2i, im, prompt, s, seed_of(n)) for im, n in zip(srcs, rows["image_name"])]
            o256 = np.stack([np.asarray(o.resize((256, 256), Image.BICUBIC)) for o in outs])
            e = encode(f, o256)
            e /= np.linalg.norm(e, axis=1, keepdims=True)
            table.append(
                {
                    "prompt": pi,
                    "strength": s,
                    "median_ita": float(np.median([ita(x) for x in o256])),
                    "cos_to_source": float(np.mean(np.sum(e * e_src, 1))),
                }
            )
            log(table[-1])
            grid.append(o256)
    log("source median ITA:", round(float(np.median([ita(x) for x in src256])), 1))
    # local-only QA contact sheet: rows = source, then each (prompt, strength); columns = 6 examples
    cols = list(range(0, 2 * n_per_class, 2))
    sheet = np.concatenate([np.concatenate([g[c] for c in cols], 1) for g in grid], 0)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(
        out_dir / "pilot_grid_LOCAL_ONLY.jpg", quality=85
    )
    with open(out_dir / "pilot.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    # a few text-to-image samples (no patient data involved)
    t2i, _ = pipelines()
    ims = []
    for c in (1, 0):
        for j in range(4):
            p = f"clinical dermatology photograph, close-up of {TEXT_DISEASE[c][0]} on the {TEXT_SITES[j]} of a person with dark brown skin"
            ims.append(
                np.asarray(
                    t2i(
                        prompt=p,
                        num_inference_steps=1,
                        guidance_scale=0.0,
                        height=RES,
                        width=RES,
                        generator=torch.Generator().manual_seed(j),
                    )
                    .images[0]
                    .resize((256, 256))
                )
            )
    sheet = np.concatenate([np.concatenate(ims[:4], 1), np.concatenate(ims[4:], 1)], 0)
    Image.fromarray(sheet).save(out_dir / "text_pilot.jpg", quality=85)
    log("pilot done")


# --------------------------------------------------------------------------- #
def make_derived(arm: str, src: Image.Image, name: str, pool, i2i, strength: float) -> Image.Image:
    if arm == "neutral":
        return img2img(i2i, src, NEUTRAL_PROMPT, strength, seed_of(name))
    guide = Image.fromarray(darken(np.asarray(src), target_for(name, pool)))
    return guide if arm == "color" else img2img(i2i, guide, TONE_PROMPT, strength, seed_of(name))


def pilot2(strengths=(0.2, 0.34), n_per_class=6, tag="pilot2"):
    """Second pilot: explicit darkening + low-noise generator pass. Scored on skin tone and
    content preservation only (no classifier is trained or scored here)."""
    from extract_features import dinov2, encode

    df = load_cohorts().internal
    pool = dark_ita_pool()
    cache = np.load(CACHE / "internal_256.npy")
    light = np.array([ita(cache[i]) for i in np.where(df["fitzpatrick"].between(1, 4).to_numpy())[0]])
    log(
        f"real skin ITA: III-IV median {np.median(light):.1f}; V-VI median {np.median(pool):.1f} "
        f"(10-90%: {np.percentile(pool, 10):.1f} to {np.percentile(pool, 90):.1f})"
    )
    src = sources(df)
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.where(src["harmonized"] == c)[0], n_per_class, replace=False) for c in (0, 1)])
    rows = src.iloc[pick]
    _, i2i = pipelines()
    f = dinov2("b")
    srcs = [load_src(p) for p in rows["path"]]
    to256 = lambda ims: np.stack([np.asarray(im.resize((256, 256), Image.BICUBIC)) for im in ims])  # noqa: E731
    unit = lambda e: e / np.linalg.norm(e, axis=1, keepdims=True)  # noqa: E731
    s256 = to256(srcs)
    e_src = unit(encode(f, s256))
    variants = (
        [("color only", "color", None)]
        + [(f"color + generator s={s}", "tone", s) for s in strengths]
        + [(f"generator only (neutral) s={s}", "neutral", s) for s in strengths]
    )
    table, grid = [], [s256]
    for label, arm, s in variants:
        outs = [make_derived(arm, im, n, pool, i2i, s or 0.2) for im, n in zip(srcs, rows["image_name"])]
        o = to256(outs)
        table.append(
            {
                "variant": label,
                "median_skin_ita": float(np.median([ita(x) for x in o])),
                "cos_to_source": float(np.mean(np.sum(unit(encode(f, o)) * e_src, 1))),
            }
        )
        log(table[-1])
        grid.append(o)
    log(f"source median skin ITA {np.median([ita(x) for x in s256]):.1f}")
    out_dir = SYN / "pilot"
    cols = list(range(0, 2 * n_per_class, 2))
    sheet = np.concatenate([np.concatenate([g[c] for c in cols], 1) for g in grid], 0)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(
        out_dir / f"{tag}_grid_LOCAL_ONLY.jpg", quality=85
    )
    with open(out_dir / f"{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def run_derived(arm: str, strength: float):
    df = load_cohorts().internal
    src = sources(df)
    pool = dark_ita_pool()
    out_dir = SYN / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    i2i = None if arm == "color" else pipelines()[1]
    rows = []
    t0 = time.time()
    for i, r in enumerate(src.itertuples(), 1):
        out = out_dir / f"{Path(r.image_name).stem}.jpg"
        if not out.exists():
            make_derived(arm, load_src(r.path), r.image_name, pool, i2i, strength).save(out, quality=95)
        rows.append(
            {
                "file": out.name,
                "source": r.image_name,
                "label": int(r.harmonized),
                "fold": int(r.fold),
                "source_fst": int(r.fitzpatrick),
                "target_ita": "" if arm == "neutral" else round(target_for(r.image_name, pool), 2),
                "prompt": {"tone": TONE_PROMPT, "neutral": NEUTRAL_PROMPT}.get(arm, ""),
                "strength": "" if arm == "color" else strength,
                "seed": seed_of(r.image_name),
            }
        )
        if i % 25 == 0:
            log(f"{arm} {i}/{len(src)}  {(time.time() - t0) / i:.1f} s/image")
    write_manifest(out_dir, rows)


def run_text():
    out_dir = SYN / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    t2i, _ = pipelines()
    rows = []
    t0 = time.time()
    for c in (1, 0):
        for j in range(TEXT_PER_CLASS):
            dz = TEXT_DISEASE[c][j % 2]
            site = TEXT_SITES[(j // 2) % len(TEXT_SITES)]
            tone = TEXT_TONES[(j // 20) % 2]
            prompt = f"clinical dermatology photograph, close-up of {dz} on the {site} of a person with {tone} skin"
            seed = 10_000 * c + j
            out = out_dir / f"text_{c}_{j:03d}.jpg"
            if not out.exists():
                t2i(
                    prompt=prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    height=RES,
                    width=RES,
                    generator=torch.Generator().manual_seed(seed),
                ).images[0].save(out, quality=95)
            rows.append(
                {
                    "file": out.name,
                    "source": "",
                    "label": c,
                    "fold": -1,
                    "source_fst": "",
                    "prompt": prompt,
                    "strength": "",
                    "seed": seed,
                }
            )
            if len(rows) % 40 == 0:
                log(f"text {len(rows)}/{2 * TEXT_PER_CLASS}  {(time.time() - t0) / len(rows):.1f} s/image")
    write_manifest(out_dir, rows)


def write_manifest(out_dir: Path, rows: list[dict]):
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log(f"{out_dir.name}: {len(rows)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", choices=["pilot", "pilot2", "pilot3", "tone", "color", "neutral", "text"])
    ap.add_argument("--strength", type=float, default=0.2)
    a = ap.parse_args()
    for arm in a.arms:
        if arm == "pilot":
            pilot()
        elif arm == "pilot2":
            pilot2()
        elif arm == "pilot3":
            pilot2(strengths=(0.1, 0.05), tag="pilot3")
        elif arm == "text":
            run_text()
        else:
            run_derived(arm, a.strength)


if __name__ == "__main__":
    main()
