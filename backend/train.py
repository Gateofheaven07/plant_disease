#!/usr/bin/env python
"""Training script for the plant leaf disease classifier."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DATASET_CANDIDATES = [
    os.environ.get("DATASET_NAME", "DATASET_DAUN"),
    "DATASET-DAUN",
]
IMG_SIZE = (160, 160)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
SEED = 1337
EPOCHS_STAGE1 = int(os.environ.get("EPOCHS_STAGE1", 10))
EPOCHS_STAGE2 = int(os.environ.get("EPOCHS_STAGE2", 10))

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = Path("model.keras")
RF_PATH = Path("rf.joblib")


def find_dataset_dir(candidates: Iterable[str]) -> Path:
    """Locate the dataset directory by checking common locations."""
    start = Path.cwd()
    for base in (start, *start.parents):
        for name in candidates:
            candidate = base / name
            if candidate.is_dir():
                return candidate.resolve()
    for name in candidates:
        matches = list(start.rglob(name))
        for match in matches:
            if match.is_dir():
                return match.resolve()
    raise FileNotFoundError(
        "Dataset directory not found. Checked candidates: "
        f"{list(candidates)}. Set DATASET_NAME or move the dataset into the project."
    )


def create_datasets(data_dir: Path):
    """Create training and validation datasets from the image directory."""
    print("Searching dataset directory...")
    abs_dir = str(data_dir)
    train_ds = keras.utils.image_dataset_from_directory(
        abs_dir,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        abs_dir,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    return train_ds, val_ds


def compute_weights(train_ds: tf.data.Dataset, num_classes: int) -> dict[int, float]:
    """Compute class weights to mitigate imbalance."""
    labels = []
    for _, batch_labels in train_ds:
        labels.append(batch_labels.numpy())
    y_all = np.concatenate(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_all,
    )
    return {i: float(w) for i, w in enumerate(weights)}


def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    """Construct the CNN classifier."""
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="preds")(x)

    model = keras.Model(inputs, outputs, name="leaf_disease_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def fine_tune_model(model: keras.Model, base_model: keras.Model):
    """Unfreeze the deeper layers of the base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def dataset_to_embeddings(extractor: keras.Model, dataset: tf.data.Dataset):
    """Convert an image dataset to embeddings and labels."""
    features, labels = [], []
    for batch_images, batch_labels in dataset:
        emb = extractor.predict(batch_images, verbose=0)
        features.append(emb)
        labels.append(batch_labels.numpy())
    return np.vstack(features), np.concatenate(labels)


def ensure_disease_info(class_names: list[str]) -> Path:
    """Create a skeleton disease info file if none exists yet."""
    info_path = ARTIFACTS_DIR / "disease_info.json"
    if not info_path.exists():
        template = {
            name: {
                "nama": name.replace("_", " "),
                "deskripsi": "",
                "pengobatan": "",
                "perawatan": "",
                "pencegahan": "",
            }
            for name in class_names
        }
        info_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Created disease info skeleton at", info_path)
    return info_path


def save_class_indices(class_names: list[str]) -> None:
    mapping = {i: name for i, name in enumerate(class_names)}
    target = ARTIFACTS_DIR / "class_indices.json"
    target.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved class indices to", target)


def main():
    tf.random.set_seed(SEED)

    data_dir = find_dataset_dir(DATASET_CANDIDATES)
    print("DATA_DIR =", data_dir)

    raw_train_ds, raw_val_ds = create_datasets(data_dir)
    class_names = raw_train_ds.class_names
    num_classes = len(class_names)
    print("Found", num_classes, "classes")
    save_class_indices(class_names)

    class_weight = compute_weights(raw_train_ds, num_classes)
    print("Class weights:", class_weight)

    autotune = tf.data.AUTOTUNE
    train_ds = raw_train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = raw_val_ds.cache().prefetch(autotune)

    model, base_model = build_model(num_classes)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
    )
    early_cb = keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)

    print("\n=== Stage 1: training (backbone frozen) ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        class_weight=class_weight,
        callbacks=[checkpoint_cb, early_cb],
    )

    print("\n=== Stage 2: fine-tuning (partial unfreeze) ===")
    fine_tune_model(model, base_model)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        class_weight=class_weight,
        callbacks=[checkpoint_cb, early_cb],
    )

    print("\nEvaluating best model...")
    best_model = keras.models.load_model(str(MODEL_PATH))
    _, val_acc = best_model.evaluate(val_ds, verbose=0)
    print(f"Best validation accuracy: {val_acc:.4f}")

    print("\n=== Train RandomForest on CNN embeddings ===")
    embedding_model = keras.Model(best_model.input, best_model.get_layer("gap").output)
    emb_train, y_train = dataset_to_embeddings(embedding_model, train_ds)
    emb_val, y_val = dataset_to_embeddings(embedding_model, val_ds)
    rf = RandomForestClassifier(
        n_estimators=400,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=SEED,
    )
    rf.fit(emb_train, y_train)
    print("RandomForest validation accuracy:", rf.score(emb_val, y_val))
    joblib.dump(rf, RF_PATH)
    print("Saved RandomForest model to", RF_PATH)

    info_path = ensure_disease_info(class_names)

    print("\nTraining complete. Artifacts saved:")
    print(" -", MODEL_PATH.resolve())
    print(" -", RF_PATH.resolve())
    print(" -", (ARTIFACTS_DIR / "class_indices.json").resolve())
    print(" -", info_path.resolve())


if __name__ == "__main__":
    main()
