"""Metadata branch: encodes tabular clinical features (age, body region, sex,
morphologic descriptors) for standalone prediction and for fusion."""
from __future__ import annotations

import torch
import torch.nn as nn


class MetadataMLP(nn.Module):
    """Shallow MLP encoder + classifier head.

    Exposes ``features(x)`` returning an embedding for fusion, mirroring the
    image-model interface.
    """

    def __init__(self, in_features: int, num_classes: int = 2,
                 hidden_dim: int = 128, feature_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MetadataLogReg(nn.Module):
    """Plain logistic regression — interpretable baseline."""

    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.feature_dim = in_features
        self.linear = nn.Linear(in_features, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_metadata_model(kind: str, in_features: int, num_classes: int = 2) -> nn.Module:
    if kind == "mlp":
        return MetadataMLP(in_features, num_classes)
    if kind == "logistic":
        return MetadataLogReg(in_features, num_classes)
    raise KeyError(f"Unknown metadata model '{kind}'. Options: mlp, logistic")
