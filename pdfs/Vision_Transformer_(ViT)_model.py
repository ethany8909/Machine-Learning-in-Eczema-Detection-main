#Important ViT Model Libraries
from transformers import ViTModel, ViTImageProcessor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set image/batch sizes for ViT preprocessing
img_height = 224
img_width = 224
batch_size = 32

class EczemViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ViT but WITHOUT the classification head
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        # Your own binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),  # 768 is ViT-base's hidden size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),    # Single output for binary
            nn.Sigmoid()
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token = global representation
        return self.classifier(cls_token)


# Custom Dataset for loading images from folder structure
class EczemDataset(Dataset):
    def __init__(self, root_dir, processor, label_mapping={'Normal': 0, 'Eczema': 1}):
        self.root_dir = root_dir
        self.processor = processor
        self.images = []
        self.labels = []
        
        # Load images from subdirectories
        for label_name, label_id in label_mapping.items():
            label_dir = os.path.join(root_dir, label_name)
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    if os.path.isfile(img_path):
                        self.images.append(img_path)
                        self.labels.append(label_id)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        encoded = self.processor(image, return_tensors='pt')
        pixel_values = encoded['pixel_values'].squeeze()
        return pixel_values, torch.tensor(label, dtype=torch.float32)


# Initialize processor and device
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset paths
train_dir = "/Users/nolanyu/Machine-Learning-in-Eczema-Detection-main/dataset/train_data"
test_dir = "/Users/nolanyu/Machine-Learning-in-Eczema-Detection-main/dataset/test_data"

from torch.utils.data import random_split

# Create datasets
full_train_dataset = EczemDataset(train_dir, processor)
test_dataset = EczemDataset(test_dir, processor)

val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = EczemViT().to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for pixel_values, labels in train_loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(pixel_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for pixel_values, labels in val_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Training loop with early stopping
np.random.seed(42)
torch.manual_seed(42)

max_epochs = 30
patience = 5
best_val_loss = float('inf')
patience_counter = 0

print(f"\nTraining ViT model for up to {max_epochs} epochs...")
for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_vit_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(torch.load('best_vit_model.pt'))

# Evaluation function
def evaluate_model(model, test_loader, device, model_name="ViT-B/16"):
    model.eval()
    y_true = []
    y_pred_probs = []
    
    with torch.no_grad():
        for pixel_values, labels in test_loader:
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values)
            
            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(outputs.cpu().numpy().flatten())
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    print(f"\n===== {model_name} =====")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, y_pred_probs):.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Eczema"]))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Eczema"], 
                yticklabels=["Normal", "Eczema"],
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

# Evaluate on test set
evaluate_model(model, test_loader, device, "ViT-B/16")