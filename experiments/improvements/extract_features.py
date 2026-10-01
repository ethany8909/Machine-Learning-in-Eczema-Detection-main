"""Build the feature bank every improvement experiment reads from.

1. Decode each image once and cache it at 256x256 (the originals are up to ~2,600 px,
   so decoding would otherwise dominate run time).
2. Extract penultimate-layer features with:
     dinov2_b14  - DINOv2 ViT-B/14 foundation model, frozen  (CLS + mean patch token, 1536-d)
     rn50_in     - ImageNet ResNet-50, frozen                 (2048-d)
     rn50_ft     - the paper's five fine-tuned ResNet-50 fold models (cv_clean)  (2048-d)
3. For augmentation experiments (point 3), extract features of V augmented views per
   internal image under three policies (mild = current training policy, strong, tone),
   plus V mild views of each external image for test-time augmentation.

Outputs (git-ignored): results/improvements/cache/*.npy, results/improvements/features/*.npz

    python experiments/improvements/extract_features.py
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision.transforms.v2 as T
from common import CACHE, FEAT, load_cohorts  # noqa: E402
from PIL import Image

from dermafair.paths import RESULTS_DIR

torch.set_num_threads(12)
SIZE = 256
V = 4  # augmented views per image per policy
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- #
# 1. image cache
# --------------------------------------------------------------------------- #
def _load(path: str) -> np.ndarray:
    im = Image.open(path)
    im.draft("RGB", (SIZE * 2, SIZE * 2))  # fast JPEG downscale while decoding
    return np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC), dtype=np.uint8)


def image_cache(name: str, paths: list[str]) -> np.ndarray:
    f = CACHE / f"{name}_{SIZE}.npy"
    if f.exists():
        return np.load(f)
    log(f"decoding {len(paths)} {name} images")
    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(_load, paths)))
    np.save(f, arr)
    return arr


# --------------------------------------------------------------------------- #
# 2. augmentation policies (point 3)
# --------------------------------------------------------------------------- #
norm = [T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)]
POLICIES = {
    "clean": T.Compose([T.Resize((224, 224), antialias=True), *norm]),
    # the policy the paper's models were trained with
    "mild": T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(0.1, 0.1, 0.1),
            *norm,
        ]
    ),
    # geometric + photometric: crops/scale, both flips, larger rotation, blur
    "strong": T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.25), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(0.25, 0.25, 0.15, 0.02),
            T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2),
            *norm,
        ]
    ),
    # lighting/exposure variation: phone photos of darker skin are often under- or
    # over-exposed, so vary brightness, contrast, saturation and gamma widely
    "tone": T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=(0.55, 1.25), contrast=(0.7, 1.3), saturation=(0.75, 1.25)),
            T.RandomApply(
                [T.Lambda(lambda x: T.functional.adjust_gamma(x, float(np.random.uniform(0.7, 1.4))))], p=0.6
            ),
            *norm,
        ]
    ),
}


POLICY_SEED = {"clean": 0, "mild": 1, "strong": 2, "tone": 3}  # fixed, so views are reproducible


def batches(arr: np.ndarray, policy: str, seed: int, bs: int = 32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf = POLICIES[policy]
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i : i + bs]).permute(0, 3, 1, 2)  # N,3,H,W uint8
        yield torch.stack([tf(img) for img in x])


# --------------------------------------------------------------------------- #
# 3. backbones
# --------------------------------------------------------------------------- #
def dinov2(size: str = "b"):
    m = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{size}14", trust_repo=True).eval()

    def f(x):
        o = m.forward_features(x)
        return torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], 1)

    return f


def resnet_imagenet():
    import torchvision

    m = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
    m.fc = torch.nn.Identity()
    return m


def resnet_finetuned(fold: int):
    from dermafair.models import build_image_model

    m = build_image_model("resnet50", num_classes=2, pretrained=False)
    sd = torch.load(RESULTS_DIR / f"cv_clean/checkpoints/resnet50_fold{fold}.pt", map_location="cpu")
    m.load_state_dict(sd)
    m.eval()
    return m.features, m.classifier


@torch.no_grad()
def encode(fn, arr, policy="clean", seed=0):
    return np.concatenate([fn(x).float().numpy() for x in batches(arr, policy, seed)]).astype(np.float32)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--sizes":
        # backbone-size study (point 1): clean features only for DINOv2 ViT-S and ViT-L
        c = load_cohorts()
        img_int = image_cache("internal", c.internal["path"].tolist())
        img_ext = image_cache("external", c.external["path"].tolist())
        for size in sys.argv[2:]:
            out = FEAT / f"dinov2_{size}14_clean.npz"
            if not out.exists():
                f = dinov2(size)
                log(f"dinov2 vit{size}14 clean")
                np.savez(out, int=encode(f, img_int), ext=encode(f, img_ext))
        log("done")
        return
    c = load_cohorts()
    img_int = image_cache("internal", c.internal["path"].tolist())
    img_ext = image_cache("external", c.external["path"].tolist())
    log(f"cache ready: internal {img_int.shape}, external {img_ext.shape}")

    # --- clean features for every backbone (experiments can start as soon as these exist) ---
    out = FEAT / "dinov2_b14_clean.npz"
    if not out.exists():
        f = dinov2()
        log("dinov2 clean")
        np.savez(out, int=encode(f, img_int), ext=encode(f, img_ext))
    out = FEAT / "rn50_in_clean.npz"
    if not out.exists():
        m = resnet_imagenet()
        log("resnet50 imagenet clean")
        np.savez(out, int=encode(m, img_int), ext=encode(m, img_ext))
    out = FEAT / "rn50_ft_clean.npz"
    if not out.exists():
        logits_i, logits_e = [], []
        fi, fe = [], []
        for k in range(5):
            feat, head = resnet_finetuned(k)
            log(f"fine-tuned resnet50 fold {k}")
            a, b = encode(feat, img_int), encode(feat, img_ext)
            fi.append(a)
            fe.append(b)
            with torch.no_grad():
                logits_i.append(head(torch.from_numpy(a)).numpy())
                logits_e.append(head(torch.from_numpy(b)).numpy())
        np.savez(out, int=np.stack(fi), ext=np.stack(fe), int_logit=np.stack(logits_i), ext_logit=np.stack(logits_e))

    # --- augmented views (DINOv2) ---
    f = None
    for policy in ("mild", "strong", "tone"):
        out = FEAT / f"dinov2_b14_{policy}.npz"
        if out.exists():
            continue
        f = f or dinov2()
        views = []
        for v in range(V):
            log(f"dinov2 {policy} view {v + 1}/{V}")
            views.append(encode(f, img_int, policy, seed=1000 * v + POLICY_SEED[policy]))
        np.savez(out, int=np.stack(views))
    out = FEAT / "dinov2_b14_ext_tta.npz"
    if not out.exists():
        f = f or dinov2()
        views = []
        for v in range(V):
            log(f"dinov2 external TTA view {v + 1}/{V}")
            views.append(encode(f, img_ext, "mild", seed=77 + v))
        np.savez(out, ext=np.stack(views))
    log("done")


if __name__ == "__main__":
    main()
