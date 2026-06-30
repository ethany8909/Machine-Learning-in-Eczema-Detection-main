"""
train_all_models.py
--------------------
Trains all 4 baseline models (cnn, resnet50, vit_b16, hybrid) sequentially
on dataset_v2 (eczema vs psoriasis) and saves results + best checkpoints.

Usage:
    python train_all_models.py --data_dir dataset_v2

Output:
    models/cnn_best.pt
    models/resnet50_best.pt
    models/vit_b16_best.pt
    models/hybrid_best.pt
    models/results_summary.csv
"""

import argparse
import os
import csv
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as tvm

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS     = 20
LR         = 1e-4
BATCH_SIZE = 32
NUM_CLASSES = 2
IMG_SIZE   = 224
CLASSES    = ["eczema", "psoriasis"]

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Model definitions (self-contained, no external imports) ───────────────────

class _Backbone(nn.Module):
    feature_dim: int
    cam_target_layer: nn.Module

    def features(self, x):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ScratchCNN(_Backbone):
    def __init__(self, num_classes=2, feature_dim=512):
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
        self.cam_target_layer = self.conv[-1][0]

    def features(self, x):
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return torch.relu(self.proj(h))

    def forward(self, x):
        return self.classifier(self.features(x))


class ResNet50(_Backbone):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = tvm.resnet50(weights=weights)
        self.feature_dim = net.fc.in_features
        self.cam_target_layer = net.layer4[-1]
        net.fc = nn.Identity()
        self.backbone = net
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.features(x))


class ViTB16(_Backbone):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.vit_b_16(weights=weights)
        self.feature_dim = net.heads.head.in_features
        net.heads = nn.Identity()
        self.backbone = net
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.cam_target_layer = net.encoder.layers[-1].ln_1

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.features(x))


class HybridCNNTransformer(_Backbone):
    def __init__(self, num_classes=2, pretrained=True,
                 d_model=256, nhead=8, nlayers=2):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = tvm.resnet50(weights=weights)
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
        h = self.stem(x)
        h = self.proj(h)
        b, d, hh, ww = h.shape
        tokens = h.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        return out[:, 0]

    def forward(self, x):
        return self.classifier(self.features(x))


MODELS = {
    "cnn":      lambda: ScratchCNN(num_classes=NUM_CLASSES),
    "resnet50": lambda: ResNet50(num_classes=NUM_CLASSES, pretrained=True),
    "vit_b16":  lambda: ViTB16(num_classes=NUM_CLASSES, pretrained=True),
    "hybrid":   lambda: HybridCNNTransformer(num_classes=NUM_CLASSES, pretrained=True),
}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Per-class accuracy
    per_class = {}
    for i, cls in enumerate(CLASSES):
        idxs = [j for j, l in enumerate(all_labels) if l == i]
        if idxs:
            per_class[cls] = sum(all_preds[j] == all_labels[j] for j in idxs) / len(idxs)
        else:
            per_class[cls] = 0.0

    return total_loss / total, correct / total, per_class


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: str):
    data_dir = Path(data_dir)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load datasets
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=test_tf)
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_ds)} images | Classes: {train_ds.classes}")
    print(f"  Test:  {len(test_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    summary_rows = []

    for model_name, model_fn in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Training: {model_name.upper()}")
        print(f"{'='*60}")

        model = model_fn().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_acc = 0.0
        best_weights = None
        start = time.time()

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, per_class = evaluate(model, test_loader, criterion)
            scheduler.step()

            # Save best weights
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

            print(f"  Epoch {epoch:02d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
                  f"Eczema: {per_class['eczema']:.3f} Psoriasis: {per_class['psoriasis']:.3f}")

        elapsed = time.time() - start

        # Save best checkpoint
        ckpt_path = models_dir / f"{model_name}_best.pt"
        torch.save({
            "model_name": model_name,
            "state_dict": best_weights,
            "val_acc": best_val_acc,
            "classes": CLASSES,
        }, ckpt_path)
        print(f"\n  Best val acc: {best_val_acc:.4f} | Saved to {ckpt_path}")
        print(f"  Training time: {elapsed/60:.1f} min")

        # Final eval with best weights
        model.load_state_dict(best_weights)
        _, final_acc, final_per_class = evaluate(model, test_loader, criterion)

        summary_rows.append({
            "model":            model_name,
            "best_val_acc":     f"{best_val_acc:.4f}",
            "eczema_acc":       f"{final_per_class['eczema']:.4f}",
            "psoriasis_acc":    f"{final_per_class['psoriasis']:.4f}",
            "train_time_min":   f"{elapsed/60:.1f}",
        })

    # Save summary CSV
    summary_path = models_dir / "results_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Val Acc':>8} {'Eczema':>8} {'Psoriasis':>10} {'Time(min)':>10}")
    print("-" * 52)
    for row in summary_rows:
        print(f"{row['model']:<12} {row['best_val_acc']:>8} {row['eczema_acc']:>8} "
              f"{row['psoriasis_acc']:>10} {row['train_time_min']:>10}")
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset_v2",
                        help="Path to dataset_v2 folder (default: dataset_v2)")
    args = parser.parse_args()
    main(args.data_dir)
