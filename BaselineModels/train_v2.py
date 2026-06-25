# ============================================================
# train.py — Scratch CNN: Eczema vs Psoriasis
# Dataset: dataset_v2 (Fitzpatrick17k, patient-level split)
# ============================================================

import tensorflow as tf
from keras.src.utils import image_dataset_from_directory
layers = tf.keras.layers
models = tf.keras.models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

keras = tf.keras
layers = tf.keras.layers
models = tf.keras.models

np.random.seed(42)
tf.random.set_seed(42)

# ---- Image / batch config ----
img_height = 224
img_width  = 224
batch_size = 16  # Reduced from 32 — dataset is small, smaller batch = more gradient updates

# ---- Class config ----
# image_dataset_from_directory assigns labels alphabetically:
# Eczema = 0, Psoriasis = 1
CLASS_NAMES = ["Eczema", "Psoriasis"]

# ---- Dataset paths (relative to repo root) ----
BASE_DIR   = Path(__file__).resolve().parent.parent
TRAIN_DIR  = BASE_DIR / "dataset_v2" / "train_data"
VAL_DIR    = BASE_DIR / "dataset_v2" / "val_data"
TEST_DIR   = BASE_DIR / "dataset_v2" / "test_data"

# ---- Data augmentation ----
# More aggressive than before — small dataset benefits from stronger augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
], name="augmentation")


# ---- Model definition ----
def build_eczema_psoriasis_cnn(input_shape=(img_height, img_width, 3)):
    model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name="scratch_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


model = build_eczema_psoriasis_cnn()
model.summary()

# ---- Load datasets ----
# Training: load from dedicated train_data folder
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary',
    shuffle=True
)

# Validation: load from dedicated val_data folder (NOT split from train)
val_ds = tf.keras.utils.image_dataset_from_directory(
    str(VAL_DIR),
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary',
    shuffle=False
)

# Test: completely held out
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
        monitor='val_loss', patience=8,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-7, verbose=1
    )
]

# ---- Train ----
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # More epochs — early stopping will cut it short if needed
    callbacks=CALLBACKS
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
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()


evaluate_model(model, test_ds, "Scratch CNN (Eczema vs Psoriasis)")

# ---- Misclassified image visualization ----
y_true_list, images = [], []
for img_batch, label_batch in val_ds:
    images.extend(img_batch.numpy())
    y_true_list.extend(label_batch.numpy())

y_true = np.array(y_true_list)
y_pred_probs = model.predict(val_ds)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

misclassified_indices = np.where(y_true != y_pred)[0]
misclassified_images  = [images[i] for i in misclassified_indices]
misclassified_labels  = [y_true[i] for i in misclassified_indices]
misclassified_preds   = [y_pred[i] for i in misclassified_indices]

num_images = len(misclassified_images)
if num_images > 0:
    cols = 5
    rows = (num_images // cols) + 1
    plt.figure(figsize=(15, rows * 3))
    for i in range(num_images):
        true_name = CLASS_NAMES[int(misclassified_labels[i])]
        pred_name = CLASS_NAMES[int(misclassified_preds[i])]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(misclassified_images[i].astype("uint8"))
        plt.title(f"True: {true_name}\nPred: {pred_name}", fontsize=8)
        plt.axis("off")

    show = input("See incorrectly predicted images? !Warning! very graphic (y/n): ")
    if show.lower() == 'y':
        import os
        os.makedirs("Results/wrong_predictions", exist_ok=True)
        plt.savefig("Results/wrong_predictions/scratch_cnn_misclassified.png")
        plt.tight_layout()
        plt.show()
else:
    print("No misclassified images in validation set.")