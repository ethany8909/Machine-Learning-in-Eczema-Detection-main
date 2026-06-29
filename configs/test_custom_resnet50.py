# Quick test script — save as test_custom_resnet50.py
from dermafair.models import build_image_model
import torch

# Build your custom model
model = build_image_model("custom_resnet50", num_classes=2, pretrained=True)
print(model)

# Test forward pass
x = torch.randn(2, 3, 224, 224)  # 2 images, 224x224, RGB
logits = model(x)
print(f"Output shape: {logits.shape}")  # should be [2, 2]

# Test features extraction (for fusion)
features = model.features(x)
print(f"Features shape: {features.shape}")  # should be [2, 128]

print("✓ Integration successful!")