#Import necessary libraries
import keras
import tensorflow as tf

#Import Scikit-learn for evaluation metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Set image/batch sizes
img_height = 224
img_width = 224
batch_size = 32

EfficientNetB0 = tf.keras.applications.EfficientNetB0
layers = tf.keras.layers
models = tf.keras.models

def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
    """Single Transformer encoder block."""
    # Multi-head self-attention
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)  # Residual connection

    # Feed-forward network
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(inputs.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)  # Residual connection

    return x

def build_hybrid_model(input_shape=(img_height, img_width, 3), num_heads=4, ff_dim=256, num_transformer_blocks=2):
    inputs = layers.Input(shape=input_shape)

    # --- CNN Backbone (feature extractor) ---
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    base_model.trainable = False  # Freeze initially, unfreeze later for fine-tuning

    cnn_features = base_model.output  # Shape: (batch, 7, 7, 1280) for 224x224 input

    # --- Reshape for Transformer ---
    # Treat each spatial location as a "token"
    batch_size, h, w, c = cnn_features.shape
    tokens = layers.Reshape((h * w, c))(cnn_features)  # Shape: (batch, 49, 1280)

    # Project to smaller dimension for efficiency
    tokens = layers.Dense(256)(tokens)  # Shape: (batch, 49, 256)

    # --- Transformer Encoder Blocks ---
    for _ in range(num_transformer_blocks):
        tokens = transformer_encoder(tokens, num_heads=num_heads, ff_dim=ff_dim)

    # --- Classification Head ---
    x = layers.GlobalAveragePooling1D()(tokens)  # Aggregate token representations
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def evaluate_model(model, dataset, model_name):
    y_true = np.concatenate([y.numpy() for x, y in dataset], axis=0)
    y_pred_probs = model.predict(dataset).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    print(f"\n===== {model_name} =====")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, y_pred_probs):.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Eczema"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Eczema"], yticklabels=["Normal", "Eczema"], cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

# Build and summarize
hybrid_model = build_hybrid_model()
hybrid_model.summary()

# ========== LOAD TRAINING AND VALIDATION DATA ==========
# Set image and batch sizes
img_height = 224
img_width = 224
batch_size = 32

# Load training and validation data
train_dir = "/Users/mzhong/Downloads/Machine-Learning-in-Eczema-Detection-main-main/dataset/train_data"

train_ds = tf.keras.utils.image_da taset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Optimize datasets
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Load test dataset for final evaluation
test_dir = "/Users/mzhong/Downloads/Machine-Learning-in-Eczema-Detection-main-main/dataset/test_data"
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(img_height, img_width),
    batch_size=batch_size
)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Phase 1: Train with frozen CNN (~5 epochs)
hybrid_model.fit(train_ds, validation_data=val_ds, epochs=5)

# Phase 2: Unfreeze top layers of EfficientNet and fine-tune
base_model = hybrid_model.layers[1]  # EfficientNetB0
base_model.trainable = True

# Only unfreeze the last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a much lower learning rate
hybrid_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # 10x lower than default
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Phase 2: Fine-tune (~10 epochs)
hybrid_model.fit(train_ds, validation_data=val_ds, epochs=10)

evaluate_model(hybrid_model, test_ds, "Hybrid")