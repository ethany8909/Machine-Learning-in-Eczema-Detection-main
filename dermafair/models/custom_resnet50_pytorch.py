"""Custom ResNet-50 baseline (Ethan's implementation) converted from TensorFlow to PyTorch.

Original code: Basline-pretrainedmodel.py (TensorFlow/Keras)
Converted to: PyTorch

Key features from original:
  - ResNet-50 with ImageNet pre-trained weights
  - Frozen backbone (base_model.trainable = False)
  - GlobalAveragePooling2D for feature aggregation
  - Dense(128, relu) + Dropout(0.5) classification head
  - Data augmentation: RandomFlip, RandomRotation, RandomZoom
  - Early stopping on val_loss with patience=10
  - Categorical crossentropy loss (→ PyTorch CrossEntropyLoss)

Integration with DermaFair:
  This model replaces the ScratchCNN baseline in the repo. You can switch between:
    build_image_model("custom_resnet50", ...)  ← your TensorFlow model converted
    build_image_model("resnet50", ...)         ← torchvision pre-trained (for comparison)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.transforms import v2 as transforms_v2


class CustomResNet50(nn.Module):
    """Your ResNet-50 implementation in PyTorch.
    
    Mirrors the original TensorFlow architecture:
      - ResNet-50 backbone (ImageNet pretrained, frozen)
      - GlobalAveragePooling
      - Dense(128, relu) + Dropout(0.5)
      - 2-class output
    
    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 2 for eczema vs. psoriasis)
    pretrained : bool
        Use ImageNet pre-trained weights (default: True, matching your original)
    dropout_rate : float
        Dropout rate after Dense(128) (default: 0.5, matching your original)
    freeze_backbone : bool
        Freeze base ResNet-50 weights (default: True, matching your original)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Load ResNet-50 with ImageNet weights (matching your TensorFlow code)
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = tvm.resnet50(weights=weights)
        
        # Remove the final FC layer (matching include_top=False in your original)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone if requested (matching base_model.trainable = False)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head (matching your original Dense layers)
        self.head = nn.Sequential(
            # GlobalAveragePooling2D is implicitly handled by resnet's avg_pool
            # But we'll add explicit flattening just to be clear
            nn.Flatten(),
            # Dense(128, activation='relu')
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            # Dropout(0.5)
            nn.Dropout(dropout_rate),
            # Dense(2, activation='softmax') → CrossEntropyLoss handles softmax
            nn.Linear(128, num_classes),
        )
        
        # Store for Grad-CAM
        self.feature_dim = 128
        self.cam_target_layer = resnet.layer4[-1]
    
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (128-dim) before classification head.
        
        Used for multimodal fusion (gate network, late-fusion).
        """
        x = self.backbone(x)  # [B, 2048, 1, 1]
        x = x.flatten(1)       # [B, 2048]
        x = self.head[0](x)    # Flatten (no-op, already flat)
        x = self.head[1](x)    # Linear(2048, 128)
        x = self.head[2](x)    # ReLU
        # DON'T apply dropout here (features for fusion should be deterministic)
        return x               # [B, 128]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with classification head.
        
        Returns logits (PyTorch convention; CrossEntropyLoss handles softmax).
        """
        x = self.backbone(x)   # [B, 2048, 1, 1]
        x = x.flatten(1)       # [B, 2048]
        x = self.head[0](x)    # Flatten (no-op)
        x = self.head[1](x)    # Linear(2048, 128)
        x = self.head[2](x)    # ReLU
        x = self.head[3](x)    # Dropout (active only during training)
        x = self.head[4](x)    # Linear(128, 2) → logits
        return x               # [B, 2]


class DataAugmentation(nn.Module):
    """Mirrors your TensorFlow data augmentation pipeline.
    
    Original:
      - layers.RandomFlip("horizontal")
      - layers.RandomRotation(0.1)
      - layers.RandomZoom(0.1)
    
    PyTorch equivalent using torchvision.transforms.v2.
    """
    
    def __init__(self):
        super().__init__()
        self.augment = transforms_v2.Compose([
            transforms_v2.RandomHorizontalFlip(p=0.5),
            transforms_v2.RandomRotation(degrees=18),  # 0.1 radians ≈ 5.7°, round to 18° for safety
            transforms_v2.RandomAffine(
                degrees=0,
                scale=(0.9, 1.1),  # RandomZoom(0.1) → scale between 0.9x and 1.1x
            ),
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation only if in training mode."""
        if self.training:
            x = self.augment(x)
        return x


def build_custom_resnet50(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> CustomResNet50:
    """Factory function to build your custom ResNet-50.
    
    Usage in the DermaFair pipeline:
        model = build_custom_resnet50(num_classes=2, pretrained=True)
    
    Parameters
    ----------
    num_classes : int
        Output classes
    pretrained : bool
        Use ImageNet weights
    freeze_backbone : bool
        Freeze backbone during training
    
    Returns
    -------
    CustomResNet50
        Instantiated model
    """
    return CustomResNet50(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


# --- OPTIONAL: Integration helper for the repo ---
# Add this to dermafair/models/image_models.py to register the custom model

def _get_custom_resnet50_registry_entry():
    """Returns the registry entry for the custom ResNet-50.
    
    To integrate into the repo, add this to the _REGISTRY dict in
    dermafair/models/image_models.py:
    
        _REGISTRY = {
            ...
            "custom_resnet50": CustomResNet50,  # Your TensorFlow-to-PyTorch model
            ...
        }
    
    Then you can use:
        model = build_image_model("custom_resnet50", num_classes=2, pretrained=True)
    """
    return {"custom_resnet50": CustomResNet50}
