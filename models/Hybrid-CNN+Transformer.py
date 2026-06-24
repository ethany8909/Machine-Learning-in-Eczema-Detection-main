# Hybrid CNN-Transformer baseline for Eczema vs. Psoriasis differential diagnosis
# Architecture: ResNet-50 backbone → spatial tokens → Transformer encoder → 2-class head
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


def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
    """Single Transformer encoder block with multi-head self-attention."""
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(inputs.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    return x


def build_hybrid_model(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    num_heads=4,
    ff_dim=256,
    num_transformer_blocks=2,
):
    """ResNet-50 CNN backbone → spatial tokens → Transformer encoder → 2-class head."""
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Lambda(lambda img: preprocess_input(img))(x)

    # ResNet-50 backbone: output shape (batch, 7, 7, 2048) for 224×224 input
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    cnn_features = base_model(x, training=False)

    # Treat each 7×7 spatial location as a token
    tokens = layers.Reshape((7 * 7, 2048))(cnn_features)  # (batch, 49, 2048)
    tokens = layers.Dense(256)(tokens)                      # (batch, 49, 256)

    for _ in range(num_transformer_blocks):
        tokens = transformer_encoder(tokens, num_heads=num_heads, ff_dim=ff_dim)

    x = layers.GlobalAveragePooling1D()(tokens)
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
    plt.savefig("confusion_matrix_hybrid.png", dpi=150)
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

# --- Phase 1: Train classification head (frozen backbone) ---
model, base_model = build_hybrid_model()
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
)

print("\nPhase 1: Training classification head with frozen ResNet-50 backbone...")
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping],
)

# --- Phase 2: Fine-tune top 20 layers of ResNet-50 ---
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

print("\nPhase 2: Fine-tuning top 20 ResNet-50 layers...")
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

# --- Training Curves (combined phases) ---
n1 = len(history_phase1.history['loss'])
combined_loss = history_phase1.history['loss'] + history_phase2.history['loss']
combined_val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
combined_acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
combined_val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(combined_loss, label='Train Loss')
plt.plot(combined_val_loss, label='Val Loss')
plt.axvline(x=n1 - 1, color='gray', linestyle='--', label='Fine-tune start')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Hybrid CNN-Transformer — Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(combined_acc, label='Train Accuracy')
plt.plot(combined_val_acc, label='Val Accuracy')
plt.axvline(x=n1 - 1, color='gray', linestyle='--', label='Fine-tune start')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Hybrid CNN-Transformer — Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_curve_hybrid.png', dpi=150)
plt.show()

# --- Evaluate ---
evaluate_model(model, test_ds, "Hybrid CNN-Transformer")

# --- Cleanup ---
try:
    del train_ds, val_ds, test_ds
except NameError:
    pass
tf.keras.backend.clear_session()
import gc
gc.collect()
model = None
