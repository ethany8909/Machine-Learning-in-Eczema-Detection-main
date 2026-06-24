# ResNet-50 transfer learning baseline for Eczema vs. Psoriasis differential diagnosis
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ResNet50 = tf.keras.applications.ResNet50
preprocess_input = tf.keras.applications.resnet50.preprocess_input
layers = tf.keras.layers
models = tf.keras.models

# --- Config ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
CLASS_NAMES = ["Eczema", "Psoriasis"]

DATASET_DIR = "/Users/mzhong/Documents/Stage 2 - Eczema - Psoriasis Differentiation/Eczema ML Extension/dataset"
train_dir = os.path.join(DATASET_DIR, "train_data")
test_dir = os.path.join(DATASET_DIR, "test_data")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


def build_resnet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """ResNet-50 with ImageNet weights, frozen backbone, fine-tune classification head."""
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    # ResNet-50 preprocess_input expects [0, 255] and performs channel-wise mean subtraction
    x = layers.Lambda(lambda img: preprocess_input(img))(x)

    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model, base_model


def evaluate_model(model, dataset, model_name):
    y_true = np.concatenate([y.numpy() for x, y in dataset], axis=0)
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_probs = model.predict(dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"\n===== {model_name} =====")
    print(f"Accuracy:  {accuracy_score(y_true_idx, y_pred):.4f}")
    print(classification_report(y_true_idx, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true_idx, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix_resnet50.png", dpi=150)
    plt.show()


# --- Load Data ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=CLASS_NAMES,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    class_names=CLASS_NAMES,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_names=CLASS_NAMES,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Build & Train ---
model, base_model = build_resnet50()
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

# --- Training Curve ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ResNet-50 — Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('ResNet-50 — Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_curve_resnet50.png', dpi=150)
plt.show()

# --- Evaluate ---
evaluate_model(model, test_ds, "ResNet-50")

# --- Cleanup ---
try:
    del train_ds, val_ds, test_ds
except NameError:
    pass
tf.keras.backend.clear_session()
import gc
gc.collect()
model = None
