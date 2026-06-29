"""Grad-CAM explainability, organized for the per-tone attention atlas.

Wraps pytorch-grad-cam with a reshape transform for ViT and a simple atlas
generator that samples N images per Fitzpatrick band per model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _vit_reshape_transform(tensor, height: int = 14, width: int = 14):
    """Reshape ViT token sequence [B, 1+H*W, C] -> [B, C, H, W] for CAM."""
    # drop CLS token, reshape patch tokens to spatial grid
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def make_cam(model, arch_name: str):
    """Build a GradCAM object for a given backbone."""
    from pytorch_grad_cam import GradCAM

    target_layer = model.cam_target_layer
    reshape = _vit_reshape_transform if arch_name == "vit_b16" else None
    return GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape)


def generate_atlas(
    model,
    arch_name: str,
    images: np.ndarray,
    fitzpatrick: np.ndarray,
    raw_images: np.ndarray,
    out_dir: Path,
    samples_per_tone: int = 5,
    bands=(3, 4, 5, 6),
    seed: int = 42,
):
    """Generate a Grad-CAM atlas: ``out_dir/<arch>/fst<band>_<i>.png``.

    Parameters
    ----------
    images : preprocessed tensor-ready array [N, 3, H, W]
    raw_images : 0-1 float RGB images [N, H, W, 3] for overlay
    """
    import torch
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from PIL import Image

    rng = np.random.default_rng(seed)
    cam = make_cam(model, arch_name)
    out_dir = Path(out_dir) / arch_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for band in bands:
        idx = np.where(fitzpatrick == band)[0]
        if idx.size == 0:
            continue
        chosen = rng.choice(idx, size=min(samples_per_tone, idx.size), replace=False)
        for j, i in enumerate(chosen):
            inp = torch.tensor(images[i : i + 1], dtype=torch.float32)
            grayscale = cam(input_tensor=inp)[0]  # [H, W]
            overlay = show_cam_on_image(
                raw_images[i].astype(np.float32), grayscale, use_rgb=True
            )
            Image.fromarray(overlay).save(out_dir / f"fst{band}_{j}.png")

    return out_dir
