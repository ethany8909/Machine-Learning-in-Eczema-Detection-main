# ViT-B/16 (ImageNet-21K) baseline for Eczema vs. Psoriasis differential diagnosis
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
CLASS_NAMES = ["Eczema", "Psoriasis"]
# Folder names in dataset must match these keys (case-insensitive checked below)
LABEL_MAPPING = {"eczema": 0, "psoriasis": 1}

DATASET_DIR = "/Users/mzhong/Documents/Stage 2 - Eczema - Psoriasis Differentiation/Eczema ML Extension/dataset"
train_dir = os.path.join(DATASET_DIR, "train_data")
test_dir = os.path.join(DATASET_DIR, "test_data")


class DermViT(nn.Module):
    """ViT-B/16 pretrained on ImageNet-21K with a custom 2-class head."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


class DermDataset(Dataset):
    """Loads Eczema / Psoriasis images from a folder-per-class directory."""

    def __init__(self, root_dir, processor, label_mapping=LABEL_MAPPING):
        self.processor = processor
        self.images = []
        self.labels = []

        for label_name, label_id in label_mapping.items():
            # Accept both lowercase and title-case folder names
            for folder in [label_name, label_name.title()]:
                label_dir = os.path.join(root_dir, folder)
                if os.path.isdir(label_dir):
                    for img_name in sorted(os.listdir(label_dir)):
                        img_path = os.path.join(label_dir, img_name)
                        if os.path.isfile(img_path):
                            self.images.append(img_path)
                            self.labels.append(label_id)
                    break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        encoded = self.processor(images=image, return_tensors='pt')
        pixel_values = encoded['pixel_values'].squeeze(0)
        return pixel_values, torch.tensor(self.labels[idx], dtype=torch.long)


def evaluate_model(model, loader, device, model_name="ViT-B/16"):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            preds = model(pixel_values).argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\n===== {model_name} =====")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix_vit.png", dpi=150)
    plt.show()


# --- Setup ---
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = DermDataset(train_dir, processor)
test_dataset = DermDataset(test_dir, processor)

# 80/20 train/val split
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_subset, val_subset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Model, Loss, Optimizer ---
model = DermViT(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Training Loop with Early Stopping ---
print(f"\nTraining ViT-B/16 (ImageNet-21K) for up to {EPOCHS} epochs...")
best_val_loss = float('inf')
epochs_without_improvement = 0
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for pixel_values, labels in train_loader:
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(pixel_values), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            val_loss += criterion(model(pixel_values), labels).item()
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_vit_checkpoint.pt")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Restore best checkpoint
model.load_state_dict(torch.load("best_vit_checkpoint.pt", map_location=device))

# --- Training Curve ---
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ViT-B/16 — Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_curve_vit.png', dpi=150)
plt.show()

# --- Evaluate ---
evaluate_model(model, test_loader, device, "ViT-B/16")

# --- Cleanup ---
try:
    del train_loader, val_loader, test_loader, train_dataset, test_dataset
except NameError:
    pass
try:
    torch.cuda.empty_cache()
except Exception:
    pass
import gc
gc.collect()
model = None
