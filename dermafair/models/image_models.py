"""Image backbones with a shared interface.

Every model exposes:
  - forward(x) -> logits  [B, num_classes]
  - features(x) -> embedding  [B, feature_dim]   (for fusion)
  - feature_dim : int
  - cam_target_layer : nn.Module                 (for Grad-CAM)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from dermafair.models.custom_resnet50 import CustomResNet50

class _Backbone(nn.Module):
    feature_dim: int
    cam_target_layer: nn.Module

    def features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class ScratchCNN(_Backbone):
    """Small 4-block CNN trained from scratch — the baseline floor."""

    def __init__(self, num_classes: int = 2, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.conv = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.cam_target_layer = self.conv[-1][0]  # last conv2d

    def features(self, x):
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return torch.relu(self.proj(h))

    def forward(self, x):
        return self.classifier(self.features(x))


class ResNet50(_Backbone):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = tvm.resnet50(weights=weights)
        self.feature_dim = net.fc.in_features  # 2048
        self.cam_target_layer = net.layer4[-1]
        net.fc = nn.Identity()
        self.backbone = net
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.features(x))


class ViTB16(_Backbone):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.vit_b_16(weights=weights)
        self.feature_dim = net.heads.head.in_features  # 768
        net.heads = nn.Identity()
        self.backbone = net
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        # Grad-CAM on ViT targets the last block's LayerNorm; reshape handled in explainability
        self.cam_target_layer = net.encoder.layers[-1].ln_1

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.features(x))


class HybridCNNTransformer(_Backbone):
    """ResNet-50 conv stem -> transformer encoder over the spatial feature grid."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 d_model: int = 256, nhead: int = 8, nlayers: int = 2):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = tvm.resnet50(weights=weights)
        # keep through layer4 -> [B, 2048, 7, 7]
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.cam_target_layer = resnet.layer4[-1]
        self.proj = nn.Conv2d(2048, d_model, 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.feature_dim = d_model
        self.classifier = nn.Linear(d_model, num_classes)

    def features(self, x):
        h = self.stem(x)                 # [B, 2048, 7, 7]
        h = self.proj(h)                 # [B, d, 7, 7]
        b, d, hh, ww = h.shape
        tokens = h.flatten(2).transpose(1, 2)             # [B, 49, d]
        cls = self.cls_token.expand(b, -1, -1)            # [B, 1, d]
        tokens = torch.cat([cls, tokens], dim=1)          # [B, 50, d]
        out = self.transformer(tokens)
        return out[:, 0]                                  # CLS embedding

    def forward(self, x):
        return self.classifier(self.features(x))


_REGISTRY = {
    "cnn": ScratchCNN,
    "resnet50": ResNet50,                    # torchvision pre-trained (original)
    "custom_resnet50": CustomResNet50,       # YOUR TensorFlow-converted model
    "vit_b16": ViTB16,
    "hybrid": HybridCNNTransformer,
}


def build_image_model(name: str, num_classes: int = 2, pretrained: bool = True) -> _Backbone:
    """Factory: build an image backbone by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown architecture '{name}'. Options: {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    if name == "cnn":
        return cls(num_classes=num_classes)
    if name == "custom_resnet50":
        return cls(num_classes=num_classes, pretrained=pretrained, freeze_backbone=True)
    return cls(num_classes=num_classes, pretrained=pretrained)
