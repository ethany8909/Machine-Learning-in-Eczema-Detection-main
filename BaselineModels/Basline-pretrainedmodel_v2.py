# ============================================================
# Basline-pretrainedmodel.py — EfficientNetB0 Transfer Learning
# Task: Eczema vs Psoriasis (harder task than eczema vs normal)
# Dataset: dataset_v2 (Fitzpatrick17k, patient-level split)
# Key changes from v1:
#   - New dataset paths (dataset_v2/)
#   - Separate val_data/ folder instead of splitting from train
#   - Top 30 EfficientNet layers unfrozen for fine-tuning
#   - Lower learning rate (1e-5) for fine-tuning stability
#   - More dropout (0.5) for small dataset
#   - Class labels updated to Eczema/Psoriasis
#   - ReduceLROnPlateau added
# ============================================================

import tensorflow as tf
keras = tf.keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)

EfficientNetB0 = tf.keras.applications.EfficientNetB0
layers  = tf.keras.layers
models  = tf.keras.models

np.random.seed(42)
tf.random.set_seed(42)

# ---- Config ----
img_height = 224
img_width  = 224
batch_size = 16  # Small dataset — smaller batch helps generalization

# Class order: alphabetical (Eczema=0, Psoriasis=1)
CLASS_NAMES = ["Eczema", "Psoriasis"]

# ---- Paths ----
BASE_DIR  = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset_v2" / "train_data"
VAL_DIR   = BASE_DIR / "dataset_v2" / "val_data"
TEST_DIR  = BASE_DIR / "dataset_v2" / "test_data"

# ---- Augmentation ----
# More aggressive for small dataset — helps prevent overfitting
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
], name="augmentation")


# ---- Model ----
def build_transfer_model(input_shape=(img_height, img_width, 3)):
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # Unfreeze the top 30 layers for fine-tuning
    # EfficientNetB0 has ~237 layers total
    # Freezing the lower layers preserves general ImageNet features
    # Unfreezing the top layers lets the model adapt to skin texture specifics
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"Trainable EfficientNet layers: {trainable_count} / {len(base_model.layers)}")

    model = models.Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name="efficientnet_transfer")

    # Lower learning rate critical for fine-tuning pretrained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


model = build_transfer_model()
model.summary()

# ---- Load datasets ----
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary',
    shuffle=True
)

# Use dedicated val folder — not split from train
val_ds = tf.keras.utils.image_dataset_from_directory(
    str(VAL_DIR),
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary',
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary',
    shuffle=False
)

# ---- Optimize pipelines ----
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---- Callbacks ----
CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_efficientnet_psoriasis.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# ---- Phase 1: Train with frozen base (5 epochs to warm up classifier head) ----
print("\n" + "="*60)
print("PHASE 1: Warming up classifier head (frozen base)")
print("="*60)

# Temporarily freeze everything for warmup
for layer in model.layers:
    if hasattr(layer, 'layers'):  # it's the base model
        for l in layer.layers:
            l.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    verbose=1
)

# ---- Phase 2: Unfreeze top 30 layers and fine-tune ----
print("\n" + "="*60)
print("PHASE 2: Fine-tuning top 30 EfficientNet layers")
print("="*60)

# Re-enable top 30 layers
base_model = model.layers[1]  # index 1 = EfficientNetB0 (after augmentation)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=CALLBACKS,
    verbose=1
)

# ---- Evaluation ----
def evaluate_model(model, dataset, model_name):
    y_true = np.concatenate([y.numpy() for x, y in dataset], axis=0)
    y_pred_probs = model.predict(dataset).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    print(f"\n===== {model_name} =====")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, y_pred_probs):.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

    return accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_pred_probs)


print("\nEvaluating on validation set:")
evaluate_model(model, val_ds, "EfficientNetB0 - Val")

print("\nEvaluating on test set (held out):")
acc, auc = evaluate_model(model, test_ds, "EfficientNetB0 - Test (Eczema vs Psoriasis)")

# ---- Training history plot ----
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("Results/efficientnet_training_history.png")
plt.show()

print(f"\nFinal Results — EfficientNetB0 (Eczema vs Psoriasis)")
print(f"  Test Accuracy: {acc:.4f}")
print(f"  Test AUC-ROC:  {auc:.4f}")
print(f"Best model saved to: best_efficientnet_psoriasis.keras")