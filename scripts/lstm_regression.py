"""
lstm_regression.py
==================
Production-ready LSTM regression pipeline for next-day return prediction.
"""

import sys
import json
import argparse
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "cleaned_stock_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODELS_DIR / "lstm_regression_model.h5"
SCALER_FILE = MODELS_DIR / "lstm_regression_scaler.pkl"
METADATA_FILE = MODELS_DIR / "lstm_regression_metadata.json"
CHECKPOINT_FILE = MODELS_DIR / "lstm_regression_best_checkpoint.h5"
BASELINE_METADATA_FILE = MODELS_DIR / "model_metadata.json"

SYMBOL_COL = "Symbol"
DATE_COL = "Date"
TARGET_COL = "Future_Return_Pct"
TRAIN_RATIO = 0.8
DEFAULT_SEQUENCE_LENGTH = 30

USE_STABLE_STOCKS_ONLY = True
PREFERRED_SYMBOLS = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"}

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA10", "MA20", "EMA12", "RSI", "MACD", "MACD_Signal",
    "Daily_Returns", "Volatility", "Weekly_Momentum",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class SequenceDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_dates: np.ndarray
    val_dates: np.ndarray
    test_dates: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM regression model.")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH, choices=[30, 60, 90])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--target-scaling", action="store_true")
    return parser.parse_args()


def load_raw_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        logger.error("Input file not found: %s", filepath)
        sys.exit(1)
    df = pd.read_csv(filepath)
    logger.info("Loaded raw data: %d rows x %d columns", len(df), len(df.columns))
    return df


def compute_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Open", "High", "Low", "Close", "Volume", SYMBOL_COL, DATE_COL}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", sorted(missing))
        sys.exit(1)

    data = df.copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])

    if USE_STABLE_STOCKS_ONLY:
        before = len(data)
        data = data[data[SYMBOL_COL].isin(PREFERRED_SYMBOLS)].copy()
        logger.info("Stable stock filter kept %d/%d rows", len(data), before)

    data = data.sort_values([SYMBOL_COL, DATE_COL]).reset_index(drop=True)
    processed = []

    for symbol, g in data.groupby(SYMBOL_COL):
        g = g.copy().sort_values(DATE_COL).reset_index(drop=True)

        g["MA10"] = g["Close"].rolling(window=10, min_periods=10).mean()
        g["MA20"] = g["Close"].rolling(window=20, min_periods=20).mean()
        g["EMA12"] = g["Close"].ewm(span=12, min_periods=12, adjust=False).mean()
        g["Daily_Returns"] = g["Close"].pct_change() * 100
        g["Volatility"] = g["High"] - g["Low"]
        g["Weekly_Momentum"] = g["Close"].pct_change(periods=5) * 100

        delta = g["Close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        g["RSI"] = 100.0 - (100.0 / (1.0 + rs))

        ema12 = g["Close"].ewm(span=12, min_periods=12, adjust=False).mean()
        ema26 = g["Close"].ewm(span=26, min_periods=26, adjust=False).mean()
        g["MACD"] = ema12 - ema26
        g["MACD_Signal"] = g["MACD"].ewm(span=9, min_periods=9, adjust=False).mean()

        # Regression target: next-day return percentage
        g[TARGET_COL] = ((g["Close"].shift(-1) - g["Close"]) / g["Close"]) * 100

        processed.append(g)

    out = pd.concat(processed, ignore_index=True)
    before_drop = len(out)
    out = out.dropna(subset=FEATURE_COLUMNS + [TARGET_COL]).reset_index(drop=True)
    logger.info("Dropped %d rows due to feature/target NaN warmup", before_drop - len(out))
    logger.info("Final dataset: %d rows | symbols=%d", len(out), out[SYMBOL_COL].nunique())
    return out


def compute_train_cutoff_date(data: pd.DataFrame) -> pd.Timestamp:
    unique_dates = np.sort(data[DATE_COL].unique())
    cutoff_idx = max(0, int(len(unique_dates) * TRAIN_RATIO) - 1)
    return pd.Timestamp(unique_dates[cutoff_idx])


def fit_feature_scaler(data: pd.DataFrame, train_cutoff_date: pd.Timestamp) -> MinMaxScaler:
    train_rows = data[data[DATE_COL] <= train_cutoff_date]
    scaler = MinMaxScaler()
    scaler.fit(train_rows[FEATURE_COLUMNS].values)
    logger.info("Feature scaler fitted on %d training rows only", len(train_rows))
    return scaler


def create_sequences(
    data: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    sequence_length: int,
    train_cutoff_date: pd.Timestamp,
    validation_split: float,
) -> SequenceDataset:
    X_train_all, y_train_all, d_train_all = [], [], []
    X_test, y_test, d_test = [], [], []

    for symbol, g in data.groupby(SYMBOL_COL):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        X_scaled = feature_scaler.transform(g[FEATURE_COLUMNS].values)
        y = g[TARGET_COL].values
        d = g[DATE_COL].values

        for i in range(sequence_length, len(g)):
            seq_x = X_scaled[i - sequence_length:i]
            seq_y = y[i]
            seq_d = d[i]
            if pd.Timestamp(seq_d) <= train_cutoff_date:
                X_train_all.append(seq_x)
                y_train_all.append(seq_y)
                d_train_all.append(seq_d)
            else:
                X_test.append(seq_x)
                y_test.append(seq_y)
                d_test.append(seq_d)

    if len(X_train_all) == 0 or len(X_test) == 0:
        logger.error("Failed to build train/test sequences.")
        sys.exit(1)

    X_train_all = np.asarray(X_train_all, dtype=np.float32)
    y_train_all = np.asarray(y_train_all, dtype=np.float32)
    d_train_all = np.asarray(d_train_all)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    d_test = np.asarray(d_test)

    val_size = max(1, int(len(X_train_all) * validation_split))
    if val_size >= len(X_train_all):
        logger.error("Validation split too large for training set.")
        sys.exit(1)

    X_train = X_train_all[:-val_size]
    y_train = y_train_all[:-val_size]
    d_train = d_train_all[:-val_size]
    X_val = X_train_all[-val_size:]
    y_val = y_train_all[-val_size:]
    d_val = d_train_all[-val_size:]

    logger.info(
        "Sequence dataset -> train=%d val=%d test=%d | seq_len=%d",
        len(y_train), len(y_val), len(y_test), sequence_length
    )

    return SequenceDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_dates=d_train,
        val_dates=d_val,
        test_dates=d_test,
    )


def fit_target_scaler(y_train: np.ndarray, enabled: bool):
    if not enabled:
        return None, y_train
    scaler = MinMaxScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    return scaler, y_train_scaled


def transform_target(y: np.ndarray, scaler):
    if scaler is None:
        return y
    return scaler.transform(y.reshape(-1, 1)).reshape(-1)


def inverse_target(y_scaled: np.ndarray, scaler):
    if scaler is None:
        return y_scaled
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)


def build_regression_model(input_shape: tuple[int, int], learning_rate: float) -> Model:
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.3)(x)
    x = GRU(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_directional(y_true_return: np.ndarray, y_pred_return: np.ndarray) -> dict:
    y_true_cls = (y_true_return > 0).astype(int)
    y_pred_cls = (y_pred_return > 0).astype(int)
    cm = confusion_matrix(y_true_cls, y_pred_cls).tolist()
    confidence = np.abs(y_pred_return)
    return {
        "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
        "precision": float(precision_score(y_true_cls, y_pred_cls, zero_division=0)),
        "recall": float(recall_score(y_true_cls, y_pred_cls, zero_division=0)),
        "f1_score": float(f1_score(y_true_cls, y_pred_cls, zero_division=0)),
        "confusion_matrix": cm,
        "confidence": {
            "mean_abs_return_pct": float(np.mean(confidence)),
            "median_abs_return_pct": float(np.median(confidence)),
            "p90_abs_return_pct": float(np.percentile(confidence, 90)),
        },
    }


def print_reports(reg_metrics: dict, cls_metrics: dict) -> None:
    cm = np.asarray(cls_metrics["confusion_matrix"])
    print("\n" + "=" * 60)
    print("  LSTM REGRESSION REPORT")
    print("=" * 60)
    print(f"  MAE:             {reg_metrics['mae']:.4f}")
    print(f"  MSE:             {reg_metrics['mse']:.4f}")
    print(f"  RMSE:            {reg_metrics['rmse']:.4f}")
    print(f"  R2:              {reg_metrics['r2']:.4f}")
    print("-" * 60)
    print("  Converted Direction Metrics (return > 0 => UP)")
    print(f"  Accuracy:        {cls_metrics['accuracy']:.4f}")
    print(f"  Precision:       {cls_metrics['precision']:.4f}")
    print(f"  Recall:          {cls_metrics['recall']:.4f}")
    print(f"  F1-score:        {cls_metrics['f1_score']:.4f}")
    print(f"  Confusion:       TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    print("  Confidence (|pred return %|):")
    print(f"    Mean:          {cls_metrics['confidence']['mean_abs_return_pct']:.4f}")
    print(f"    Median:        {cls_metrics['confidence']['median_abs_return_pct']:.4f}")
    print(f"    P90:           {cls_metrics['confidence']['p90_abs_return_pct']:.4f}")
    print("=" * 60)


def load_baselines() -> list[dict]:
    if not BASELINE_METADATA_FILE.exists():
        return []
    try:
        meta = json.loads(BASELINE_METADATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

    cmp_rows = meta.get("model_comparison")
    if isinstance(cmp_rows, list) and cmp_rows:
        return [
            {
                "model": row.get("model", "Unknown"),
                "accuracy": float(row.get("accuracy", np.nan)),
                "f1_score": float(row.get("f1_score", np.nan)),
                "roc_auc": float(row.get("roc_auc", np.nan)),
            }
            for row in cmp_rows
        ]
    return []


def print_model_comparison(lstm_cls_metrics: dict) -> list[dict]:
    rows = load_baselines()
    rows.append(
        {
            "model": "LSTM_Regression(direction)",
            "accuracy": lstm_cls_metrics["accuracy"],
            "f1_score": lstm_cls_metrics["f1_score"],
            "roc_auc": np.nan,
        }
    )
    rows_sorted = sorted(rows, key=lambda r: (r["f1_score"], r["accuracy"]), reverse=True)

    print("\n" + "=" * 60)
    print("  FINAL MODEL COMPARISON REPORT")
    print("=" * 60)
    for r in rows_sorted:
        roc_txt = "nan" if np.isnan(r["roc_auc"]) else f"{r['roc_auc']:.4f}"
        print(f"{r['model']:<30s} | Acc={r['accuracy']:.4f} | F1={r['f1_score']:.4f} | ROC AUC={roc_txt}")
    print("=" * 60)
    return rows_sorted


def save_artifacts(
    model: Model,
    feature_scaler: MinMaxScaler,
    target_scaler,
    args: argparse.Namespace,
    reg_metrics: dict,
    cls_metrics: dict,
    comparison_rows: list[dict],
    train_info: dict,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_FILE)
    joblib.dump({"feature_scaler": feature_scaler, "target_scaler": target_scaler}, SCALER_FILE, compress=3)

    metadata = {
        "model_type": "LSTM_Regression",
        "model_file": str(MODEL_FILE),
        "scaler_file": str(SCALER_FILE),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COL,
        "target_formula": "((Close[t+1]-Close[t])/Close[t])*100",
        "sequence_length": args.sequence_length,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "validation_split": args.validation_split,
            "learning_rate": args.learning_rate,
            "loss": "Huber",
            "target_scaling": bool(args.target_scaling),
        },
        "dataset_info": train_info,
        "regression_metrics": reg_metrics,
        "directional_metrics": cls_metrics,
        "model_comparison": comparison_rows,
        "trained_at": datetime.now().isoformat(),
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model -> %s", MODEL_FILE)
    logger.info("Saved scaler bundle -> %s", SCALER_FILE)
    logger.info("Saved metadata -> %s", METADATA_FILE)


def main() -> None:
    args = parse_args()
    logger.info("=" * 60)
    logger.info("LSTM REGRESSION PIPELINE")
    logger.info("=" * 60)
    logger.info("Config: seq=%d epochs=%d batch=%d val=%.2f", args.sequence_length, args.epochs, args.batch_size, args.validation_split)

    df_raw = load_raw_data(INPUT_FILE)
    df = compute_features_and_target(df_raw)
    cutoff = compute_train_cutoff_date(df)
    logger.info("Chronological split cutoff date: %s", cutoff.date())

    x_scaler = fit_feature_scaler(df, cutoff)
    seq = create_sequences(df, x_scaler, args.sequence_length, cutoff, args.validation_split)

    y_scaler, y_train_fit = fit_target_scaler(seq.y_train, enabled=args.target_scaling)
    y_val_fit = transform_target(seq.y_val, y_scaler)
    y_test_fit = transform_target(seq.y_test, y_scaler)

    model = build_regression_model(
        input_shape=(args.sequence_length, len(FEATURE_COLUMNS)),
        learning_rate=args.learning_rate,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1),
        ModelCheckpoint(filepath=str(CHECKPOINT_FILE), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        seq.X_train,
        y_train_fit,
        validation_data=(seq.X_val, y_val_fit),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    logger.info("Training completed in %d epochs", len(history.history.get("loss", [])))

    pred_test_fit = model.predict(seq.X_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    pred_test = inverse_target(pred_test_fit, y_scaler)
    y_test = seq.y_test

    reg_metrics = evaluate_regression(y_test, pred_test)
    cls_metrics = evaluate_directional(y_test, pred_test)
    print_reports(reg_metrics, cls_metrics)
    comparison_rows = print_model_comparison(cls_metrics)

    train_info = {
        "rows_after_prep": int(len(df)),
        "symbols": sorted(df[SYMBOL_COL].unique().tolist()),
        "date_range": {
            "start": str(df[DATE_COL].min().date()),
            "end": str(df[DATE_COL].max().date()),
            "train_end": str(cutoff.date()),
        },
        "sequence_counts": {
            "train": int(len(seq.y_train)),
            "val": int(len(seq.y_val)),
            "test": int(len(seq.y_test)),
        },
        "history": {
            "epochs_ran": int(len(history.history.get("loss", []))),
            "best_val_loss": float(np.min(history.history.get("val_loss", [np.nan]))),
        },
    }

    save_artifacts(
        model=model,
        feature_scaler=x_scaler,
        target_scaler=y_scaler,
        args=args,
        reg_metrics=reg_metrics,
        cls_metrics=cls_metrics,
        comparison_rows=comparison_rows,
        train_info=train_info,
    )
    logger.info("LSTM regression pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)
