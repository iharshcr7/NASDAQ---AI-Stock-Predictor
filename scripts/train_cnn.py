"""
train_cnn.py
============
Train CNN classifier on candlestick images for bullish/bearish classes.
"""

from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = PROJECT_ROOT / "data" / "candlestick_image_dataset.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "cnn_candlestick_model.keras"
META_FILE = PROJECT_ROOT / "models" / "cnn_candlestick_metadata.json"


def load_dataset(dataset_csv: Path, img_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(dataset_csv)
    required = {"image_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset CSV missing columns: {sorted(missing)}")

    x_rows = []
    y_rows = []
    label_map = {"bearish": 0, "bullish": 1, 0: 0, 1: 1}
    for _, row in df.iterrows():
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / row["image_path"]
        if not img_path.exists():
            continue
        img = tf.keras.utils.load_img(img_path, target_size=img_size)
        arr = tf.keras.utils.img_to_array(img) / 255.0
        label_val = label_map.get(row["label"])
        if label_val is None:
            continue
        x_rows.append(arr)
        y_rows.append(label_val)

    X = np.array(x_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    return X, y


def build_cnn(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN on candlestick images")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    X, y = load_dataset(DATASET_CSV, (args.img_size, args.img_size))
    if len(X) == 0:
        raise RuntimeError("No valid images loaded from dataset.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info("CNN dataset: train=%d val=%d", len(X_train), len(X_val))

    model = build_cnn((args.img_size, args.img_size, 3))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    y_prob = model.predict(X_val, batch_size=args.batch_size, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_val, y_pred, target_names=["bearish", "bullish"])
    cm = confusion_matrix(y_val, y_pred).tolist()

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)
    meta = {
        "model_type": "CNN_Candlestick",
        "model_file": str(MODEL_FILE),
        "epochs_ran": len(hist.history.get("loss", [])),
        "val_accuracy_last": float(hist.history.get("val_accuracy", [0])[-1]),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(report)
    print("Confusion Matrix:", cm)
    logger.info("CNN training complete. Model -> %s", MODEL_FILE)


if __name__ == "__main__":
    main()

