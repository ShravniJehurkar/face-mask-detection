"""
Face Mask Detection - Model Training Script
==========================================
Trains a CNN-based classifier to detect whether a person is
wearing a face mask or not using transfer learning with MobileNetV2.

Author: [Your Name]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI use
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    AveragePooling2D, Dropout, Flatten, Dense, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)   # MobileNetV2 default
BATCH_SIZE    = 32
INIT_LR       = 1e-4
EPOCHS        = 20
DATASET_PATH  = "dataset"
MODEL_PATH    = "model/mask_detector.h5"
PLOT_PATH     = "logs/training_plot.png"
LOG_PATH      = "logs/training_log.csv"
# ──────────────────────────────────────────────────────────────────────────────


def build_model(num_classes: int = 2) -> Model:
    """
    Build a transfer-learning model using MobileNetV2 as a frozen base
    with a custom classification head on top.
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))
    )
    # Freeze the base — we only train the head
    base_model.trainable = False

    head = base_model.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten()(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(num_classes, activation="softmax")(head)

    return Model(inputs=base_model.input, outputs=head)


def build_generators(dataset_path: str):
    """Create augmented training and validation data generators."""
    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20,
    )
    val_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.20,
    )

    train_gen = train_aug.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_gen = val_aug.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )
    return train_gen, val_gen


def plot_training(history, output_path: str):
    """Save accuracy/loss curves to disk."""
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="train_acc")
    axes[0].plot(history.history["val_accuracy"], label="val_acc")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[INFO] Training plot saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Face Mask Detector")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--model",   default=MODEL_PATH)
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--lr",      type=float, default=INIT_LR)
    args = parser.parse_args()

    os.makedirs("model", exist_ok=True)
    os.makedirs("logs",  exist_ok=True)

    print("[INFO] Loading dataset ...")
    train_gen, val_gen = build_generators(args.dataset)
    class_names = list(train_gen.class_indices.keys())
    print(f"[INFO] Classes found: {class_names}")

    print("[INFO] Building model ...")
    model = build_model(num_classes=len(class_names))
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(
            args.model,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        CSVLogger(LOG_PATH),
    ]

    print("[INFO] Training ...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("[INFO] Evaluating on validation set ...")
    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = val_gen.classes

    print("\n" + classification_report(
        true_labels, pred_labels, target_names=class_names
    ))

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("logs/confusion_matrix.png", dpi=150)
    print("[INFO] Confusion matrix saved → logs/confusion_matrix.png")

    plot_training(history, PLOT_PATH)
    print(f"[INFO] Best model saved → {args.model}")


if __name__ == "__main__":
    main()
