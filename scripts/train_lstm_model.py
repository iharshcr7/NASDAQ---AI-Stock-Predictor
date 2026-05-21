"""
train_lstm_model.py
===================
LSTM model training for stock movement prediction using time-series sequences.

Unlike Random Forest (which uses a single-row snapshot of 21 features),
LSTM learns from a SEQUENCE of the last SEQUENCE_LENGTH days to capture
temporal patterns, trends, and momentum over time.

Architecture:
    Input: (batch, 60, 21) — 60 days × 21 features per day
    → LSTM(128, return_sequences=True)
    → Dropout(0.2)
    → LSTM(64)
    → Dropout(0.2)
    → Dense(32, relu)
    → Dense(1, sigmoid)
    Output: probability of UP (>0.5 = UP, <=0.5 = DOWN)

Usage:
    python scripts/train_lstm_model.py
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "final_featured_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"

LSTM_MODEL_FILE = MODELS_DIR / "lstm_model.keras"
LSTM_SCALER_FILE = MODELS_DIR / "lstm_scaler.pkl"
LSTM_METADATA_FILE = MODELS_DIR / "lstm_metadata.json"
LSTM_HISTORY_FILE = MODELS_DIR / "lstm_history.json"   # training curves for dashboard

# Same 21 features used by Random Forest — keeps pipelines consistent
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA10", "MA20", "EMA12",
    "RSI", "MACD", "MACD_Signal",
    "Daily_Returns", "Volatility", "Price_Change_Pct",
    "Weekly_Momentum", "Trend_Strength", "BB_Width",
    "Avg_5D_Volume_Trend", "Lag_1", "Lag_3",
]

TARGET_COLUMN = "Target"
SYMBOL_COLUMN = "Symbol"
DATE_COLUMN = "Date"

# Stocks to train on (same as RF stable symbols)
PREFERRED_SYMBOLS = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"}

# LSTM hyperparameters
SEQUENCE_LENGTH = 60    # Number of past days used as input sequence
TEST_SIZE = 0.2         # 80/20 chronological split
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ---------------------------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        logger.error("Input file not found: %s", INPUT_FILE)
        logger.error("Run scripts/feature_engineering.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    logger.info("Loaded dataset: %d rows × %d columns", len(df), len(df.columns))
    return df


def filter_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    required = {SYMBOL_COLUMN, DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing columns: %s", sorted(missing))
        sys.exit(1)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df[df[SYMBOL_COLUMN].isin(PREFERRED_SYMBOLS)].copy()
    df = df.sort_values([DATE_COLUMN, SYMBOL_COLUMN]).reset_index(drop=True)

    # Remove NaN / Inf
    invalid = df[FEATURE_COLUMNS + [TARGET_COLUMN]].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    df = df[~invalid].reset_index(drop=True)

    logger.info(
        "Filtered dataset: %d rows | symbols=%s | %s → %s",
        len(df),
        sorted(df[SYMBOL_COLUMN].unique()),
        df[DATE_COLUMN].min().date(),
        df[DATE_COLUMN].max().date(),
    )
    return df


def build_sequences(df: pd.DataFrame, scaler) -> tuple:
    """
    Build (X, y) sequences per stock symbol, then concatenate.

    For each symbol:
        - Scale features using the fitted scaler
        - Slide a window of SEQUENCE_LENGTH days
        - X[i] = features[i : i+SEQUENCE_LENGTH]  shape: (SEQ_LEN, n_features)
        - y[i] = target[i+SEQUENCE_LENGTH]

    Returns
    -------
    X : np.ndarray  shape (n_samples, SEQUENCE_LENGTH, n_features)
    y : np.ndarray  shape (n_samples,)
    """
    X_all, y_all = [], []

    for symbol, group in df.groupby(SYMBOL_COLUMN):
        group = group.sort_values(DATE_COLUMN).reset_index(drop=True)

        features = group[FEATURE_COLUMNS].values.astype(np.float32)
        targets = group[TARGET_COLUMN].values.astype(np.int32)

        # Scale features
        features_scaled = scaler.transform(features)

        # Build sliding windows
        for i in range(len(features_scaled) - SEQUENCE_LENGTH):
            X_all.append(features_scaled[i : i + SEQUENCE_LENGTH])
            y_all.append(targets[i + SEQUENCE_LENGTH])

        logger.info("  %s: %d sequences built", symbol, len(group) - SEQUENCE_LENGTH)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    logger.info("Total sequences: X=%s  y=%s", X.shape, y.shape)
    return X, y


def chronological_split(X: np.ndarray, y: np.ndarray):
    split = int(len(X) * (1 - TEST_SIZE))
    return X[:split], X[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------------

def build_lstm_model(n_features: int, seq_len: int):
    """Build and compile the LSTM model."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        logger.error("TensorFlow not installed. Run: pip install tensorflow==2.15.0")
        sys.exit(1)

    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),

        # First LSTM layer — returns sequences for stacking
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),

        # Second LSTM layer — returns final hidden state
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),

        # Dense head
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # Binary: UP probability
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)
    return model


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train(model, X_train, y_train, X_test, y_test):
    """Train with early stopping and learning rate reduction."""
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    except ImportError:
        logger.error("TensorFlow not installed.")
        sys.exit(1)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(LSTM_MODEL_FILE),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    logger.info("Training LSTM: epochs=%d  batch=%d  seq_len=%d", EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics dict."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, classification_report,
    )

    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["DOWN", "UP"]
        ),
    }

    confidence = np.maximum(y_proba, 1.0 - y_proba)
    metrics["prediction_confidence"] = {
        "mean":   float(np.mean(confidence)),
        "median": float(np.median(confidence)),
        "p90":    float(np.percentile(confidence, 90)),
    }

    logger.info("=" * 60)
    logger.info("  LSTM EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info("  Accuracy:  %.4f (%.2f%%)", metrics["accuracy"], metrics["accuracy"] * 100)
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall:    %.4f", metrics["recall"])
    logger.info("  F1-Score:  %.4f", metrics["f1_score"])
    logger.info("  ROC AUC:   %.4f", metrics["roc_auc"])
    logger.info("  Classification Report:\n%s", metrics["classification_report"])
    logger.info("=" * 60)

    return metrics


def save_artifacts(model, scaler, metrics: dict, history) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Model already saved by ModelCheckpoint callback — just log size
    if LSTM_MODEL_FILE.exists():
        size_mb = LSTM_MODEL_FILE.stat().st_size / (1024 * 1024)
        logger.info("LSTM model saved → %s (%.2f MB)", LSTM_MODEL_FILE, size_mb)

    # Save scaler
    joblib.dump(scaler, LSTM_SCALER_FILE, compress=3)
    logger.info("Scaler saved → %s", LSTM_SCALER_FILE)

    # Save training history (for dashboard graphs)
    history_data = {
        "accuracy":     [float(v) for v in history.history.get("accuracy", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        "loss":         [float(v) for v in history.history.get("loss", [])],
        "val_loss":     [float(v) for v in history.history.get("val_loss", [])],
    }
    with open(LSTM_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    logger.info("Training history saved → %s", LSTM_HISTORY_FILE)

    # Save metadata
    val_acc = history.history.get("val_accuracy", [])
    val_loss = history.history.get("val_loss", [])

    metadata = {
        "model_type": "LSTM",
        "model_file": str(LSTM_MODEL_FILE),
        "scaler_file": str(LSTM_SCALER_FILE),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "stable_symbols": sorted(PREFERRED_SYMBOLS),
        "sequence_length": SEQUENCE_LENGTH,
        "architecture": {
            "layers": [
                "LSTM(128, return_sequences=True)",
                "Dropout(0.2)",
                "LSTM(64)",
                "Dropout(0.2)",
                "Dense(32, relu)",
                "Dense(1, sigmoid)",
            ],
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "binary_crossentropy",
        },
        "training": {
            "epochs_configured": EPOCHS,
            "epochs_trained": len(val_loss),
            "batch_size": BATCH_SIZE,
            "split_strategy": "chronological_80_20",
            "early_stopping": True,
        },
        "metrics": {
            "accuracy":  metrics["accuracy"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1_score":  metrics["f1_score"],
            "roc_auc":   metrics["roc_auc"],
            "prediction_confidence": metrics["prediction_confidence"],
            "best_val_accuracy": float(max(val_acc)) if val_acc else None,
            "best_val_loss":     float(min(val_loss)) if val_loss else None,
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "prediction_config": {
            "labels": {"0": "DOWN", "1": "UP"},
            "threshold": 0.5,
            "confidence_source": "sigmoid_output_probability",
        },
        "deployment_notes": {
            "streamlit_compatible": True,
            "live_mode_only": True,
            "reason": (
                "LSTM requires a sequence of SEQUENCE_LENGTH past days. "
                "Manual mode provides only a single snapshot, so LSTM is "
                "only available in Live Data mode."
            ),
        },
        "trained_at": datetime.now().isoformat(),
    }

    with open(LSTM_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved → %s", LSTM_METADATA_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("LSTM TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Load & filter data
    df = load_dataset()
    df = filter_and_validate(df)

    # 2. Fit scaler on ALL training data (before sequence split)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_features = df[FEATURE_COLUMNS].values.astype(np.float32)
    scaler.fit(all_features)
    logger.info("MinMaxScaler fitted on %d rows", len(all_features))

    # 3. Build sequences
    logger.info("Building sequences (seq_len=%d)...", SEQUENCE_LENGTH)
    X, y = build_sequences(df, scaler)

    # 4. Chronological split
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))

    # Class distribution
    up_train = int(y_train.sum())
    down_train = int(len(y_train) - up_train)
    logger.info("Train target: UP=%d  DOWN=%d", up_train, down_train)

    # 5. Build model
    n_features = len(FEATURE_COLUMNS)
    model = build_lstm_model(n_features, SEQUENCE_LENGTH)

    # 6. Train
    history = train(model, X_train, y_train, X_test, y_test)

    # 7. Evaluate
    metrics = evaluate(model, X_test, y_test)

    # 8. Save
    save_artifacts(model, scaler, metrics, history)

    logger.info("=" * 60)
    logger.info("LSTM TRAINING COMPLETE")
    logger.info("  Model:    %s", LSTM_MODEL_FILE)
    logger.info("  Scaler:   %s", LSTM_SCALER_FILE)
    logger.info("  Metadata: %s", LSTM_METADATA_FILE)
    logger.info("=" * 60)
    logger.info("Next step: streamlit run app.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("LSTM training failed: %s", exc)
        sys.exit(1)
