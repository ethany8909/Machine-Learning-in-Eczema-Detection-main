#!/usr/bin/env python
"""Grad-CAM atlas — where each image architecture looks, by skin tone.

Grid: rows = original + conv architectures, columns = one representative correctly-
classified test image per Fitzpatrick band. Grad-CAM is computed by hooking each
model's ``cam_target_layer``. Interpretability figure (Week-5 Fig 8).

NOTE: contains real patient skin photos -> LOCAL ONLY, do not publish without consent.
ViT-B/16 is omitted (its token layer needs a reshape transform, not standard Grad-CAM).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dermafair.data.folder_split import FolderSplitDataset, _load_fitzpatrick_map, _build_transforms
from dermafair.models import build_image_model

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT.parent / "dataset_split_clean"
META = ROOT.parent / "Skin_Metadata-1.csv"
CKPT = ROOT / "results/image_models_clean/checkpoints"
OUT = ROOT / "results/week5_gradcam"
OUT.mkdir(parents=True, exist_ok=True)
BANDS = [3, 4, 5, 6]
ARCHS = ["cnn", "resnet50", "custom_resnet50", "hybrid"]   # conv-based (Grad-CAM standard)
CLASS = {0: "Ecz", 1: "Pso"}


def gradcam(model, x, cls):
    acts, grads = {}, {}

    def fh(m, i, o):
        acts["v"] = o                     # returns None -> does not replace output

    def bh(m, gi, go):
        grads["v"] = go[0].detach()       # returns None -> does not modify gradient

    h1 = model.cam_target_layer.register_forward_hook(fh)
    h2 = model.cam_target_layer.register_full_backward_hook(bh)
    model.zero_grad()
    out = model(x)
    out[0, cls].backward()
    a, g = acts["v"][0].detach(), grads["v"][0]           # [C,H,W]
    cam = F.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    h1.remove(); h2.remove()
    return cam.detach().cpu().numpy()


def main():
    fitz_map = _load_fitzpatrick_map(META)
    ds = FolderSplitDataset(SPLIT / "test", fitz_map, _build_transforms(224, augment=False))
    samples = ds.samples                                  # (path, label, fitz, name)
    tfm = _build_transforms(224, augment=False)

    # pick one example image per band (prefer one resnet50 gets right)
    ref = build_image_model("resnet50", num_classes=2, pretrained=True)
    ref.load_state_dict(torch.load(CKPT / "resnet50.pt", map_location="cpu")); ref.eval()
    examples = {}
    for b in BANDS:
        cands = [(p, l, n) for (p, l, f, n) in samples if f == b]
        chosen = None
        for p, l, n in cands:
            x = tfm(Image.open(p).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                if int(ref(x).argmax(1)) == l:
                    chosen = (p, l); break
        examples[b] = chosen or (cands[0][0], cands[0][1]) if cands else None

    # preload models
    models = {}
    for a in ARCHS:
        m = build_image_model(a, num_classes=2, pretrained=True)
        m.load_state_dict(torch.load(CKPT / f"{a}.pt", map_location="cpu")); m.eval()
        models[a] = m

    nrows, ncols = len(ARCHS) + 1, len(BANDS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    for j, b in enumerate(BANDS):
        if examples[b] is None:
            continue
        path, label = examples[b]
        disp = np.asarray(Image.open(path).convert("RGB").resize((224, 224)))
        x = tfm(Image.open(path).convert("RGB")).unsqueeze(0)
        axes[0, j].imshow(disp); axes[0, j].set_title(f"FST {b} — true {CLASS[label]}", fontsize=10)
        for i, a in enumerate(ARCHS, start=1):
            m = models[a]
            with torch.no_grad():
                pred = int(m(x).argmax(1))
            cam = gradcam(m, x.clone().requires_grad_(True), pred)
            axes[i, j].imshow(disp); axes[i, j].imshow(cam, cmap="jet", alpha=0.45)
            axes[i, j].set_title(f"{a} → {CLASS[pred]}", fontsize=9)
    for i, name in enumerate(["Original"] + ARCHS):
        axes[i, 0].set_ylabel(name, fontsize=11)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("Figure 8 — Grad-CAM atlas: where each architecture attends, by skin tone\n"
                 "(local only — patient photos)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / "gradcam_atlas.png", dpi=140)
    plt.close()
    print(f"wrote {OUT}/gradcam_atlas.png")


if __name__ == "__main__":
    main()
