#Import necessary libraries
import keras
import tensorflow as tf

#Import Scikit-learn for evaluation metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

EfficientNetB0 = tf.keras.applications.EfficientNetB0
layers = tf.keras.layers
models = tf.keras.models

# Set image/batch sizes
img_height = 224
img_width = 224
batch_size = 32

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def build_transfer_cnn(input_shape=(img_height, img_width, 3)):
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze base initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and display the model
model = build_transfer_cnn()
model.summary()

# ========== TESTING SECTION ==========
# Load test data (same as train.py)
test_dir = "/Users/mzhong/Downloads/Machine-Learning-in-Eczema-Detection-main-main/dataset/test_data"

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Optimize test dataset
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Normal", "Eczema"], 
                yticklabels=["Normal", "Eczema"],
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

evaluate_model(model, test_ds, "EfficientNetB0")