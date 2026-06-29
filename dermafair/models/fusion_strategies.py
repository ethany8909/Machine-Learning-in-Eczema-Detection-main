"""Multimodal fusion strategies.

LateFusion       : fixed 0.5/0.5 logit averaging — the ablation/control.
GateNetwork      : learns per-sample [w_image, w_metadata] from concatenated
                   features; the novelty core. Exposes ``last_gate_weights`` so
                   training/eval loops can log what it learned per sample.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusion(nn.Module):
    """Fixed-weight late fusion of two branches at the logit level."""

    def __init__(self, image_model: nn.Module, metadata_model: nn.Module,
                 w_image: float = 0.5):
        super().__init__()
        self.image_model = image_model
        self.metadata_model = metadata_model
        self.w_image = w_image
        self.w_meta = 1.0 - w_image

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        li = self.image_model(image)
        lm = self.metadata_model(meta)
        return self.w_image * li + self.w_meta * lm


class GateNetwork(nn.Module):
    """Learned adaptive fusion.

    A gate head consumes concatenated image+metadata embeddings and emits a
    softmax weighting over the two modalities, applied at the logit level.

    The per-sample weights are stored in ``last_gate_weights`` (shape [B, 2],
    columns = [image, metadata]) on every forward pass — this is what powers the
    per-Fitzpatrick weighting analysis (the paper's signature figure).
    """

    def __init__(self, image_model: nn.Module, metadata_model: nn.Module,
                 num_classes: int = 2, hidden_dim: int = 128,
                 freeze_image_backbone: bool = False):
        super().__init__()
        self.image_model = image_model
        self.metadata_model = metadata_model

        if freeze_image_backbone:
            for p in self.image_model.parameters():
                p.requires_grad = False

        img_dim = image_model.feature_dim
        meta_dim = metadata_model.feature_dim
        self.gate = nn.Sequential(
            nn.Linear(img_dim + meta_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # -> [logit_image, logit_metadata]
        )
        # independent classifier heads per branch so logits are comparable
        self.image_head = nn.Linear(img_dim, num_classes)
        self.meta_head = nn.Linear(meta_dim, num_classes)

        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        fi = self.image_model.features(image)     # [B, img_dim]
        fm = self.metadata_model.features(meta)   # [B, meta_dim]

        gate_logits = self.gate(torch.cat([fi, fm], dim=1))   # [B, 2]
        weights = F.softmax(gate_logits, dim=1)               # [B, 2]
        self.last_gate_weights = weights.detach()

        li = self.image_head(fi)
        lm = self.meta_head(fm)
        w_img = weights[:, 0:1]
        w_meta = weights[:, 1:2]
        return w_img * li + w_meta * lm

    @torch.no_grad()
    def gate_weights(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Return per-sample [w_image, w_metadata] without classification."""
        fi = self.image_model.features(image)
        fm = self.metadata_model.features(meta)
        return F.softmax(self.gate(torch.cat([fi, fm], dim=1)), dim=1)


def build_fusion(strategy: str, image_model: nn.Module, metadata_model: nn.Module,
                 num_classes: int = 2, **kwargs) -> nn.Module:
    if strategy == "late_fusion":
        return LateFusion(image_model, metadata_model)
    if strategy == "gate_network":
        return GateNetwork(image_model, metadata_model, num_classes=num_classes, **kwargs)
    raise KeyError(f"Unknown fusion '{strategy}'. Options: late_fusion, gate_network")
