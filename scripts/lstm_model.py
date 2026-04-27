"""
lstm_model.py
=============
Collapse-resistant deep sequence model for stock movement prediction.
Implements leak-safe preprocessing, dynamic target relabeling, threshold tuning,
and strict quality gates before saving the model.
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dropout, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "final_featured_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
LSTM_MODEL_FILE = MODELS_DIR / "lstm_model.h5"
LSTM_SCALER_FILE = MODELS_DIR / "lstm_scaler.pkl"
LSTM_METADATA_FILE = MODELS_DIR / "lstm_metadata.json"
BASELINE_METADATA_FILE = MODELS_DIR / "model_metadata.json"
CHECKPOINT_FILE = MODELS_DIR / "lstm_best_checkpoint.h5"

SYMBOL_COLUMN = "Symbol"
DATE_COLUMN = "Date"
CLOSE_COLUMN = "Close"
TARGET_COLUMN = "Target"
FUTURE_RETURN_COLUMN = "Future_Return_Pct"

TRAIN_RATIO = 0.8
DEFAULT_SEQUENCE_LENGTH = 30
THRESHOLD_GRID = [0.50, 0.52, 0.55, 0.58, 0.60]
TARGET_UPPER_PCT = 2.0
TARGET_LOWER_PCT = -2.0
MIN_ACCEPTABLE_ROC_AUC = 0.53
MIN_ACCEPTABLE_F1 = 0.50

USE_STABLE_STOCKS_ONLY = True
PREFERRED_SYMBOLS = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"}

# Reduced core feature set to avoid noisy inputs and collapse.
FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "MA10",
    "MA20",
    "EMA12",
    "RSI",
    "MACD",
    "MACD_Signal",
    "Daily_Returns",
    "Weekly_Momentum",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(description="Train collapse-resistant LSTM/GRU model.")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH, choices=[30, 60, 90])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma-focal", type=float, default=1.5)
    parser.add_argument("--alpha-focal", type=float, default=0.35)
    return parser.parse_args()


def load_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        logger.error("Input dataset not found: %s", filepath)
        logger.error("Run feature_engineering.py first.")
        sys.exit(1)
    df = pd.read_csv(filepath)
    logger.info("Loaded dataset: %d rows x %d columns", len(df), len(df.columns))
    return df


def relabel_target_with_stronger_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relabel target using ±2%% threshold and remove noisy labels.
    Uses Future_Return_Pct if available to avoid recomputation ambiguity.
    """
    data = df.copy()

    if FUTURE_RETURN_COLUMN not in data.columns:
        logger.error(
            "Required column '%s' not found. Re-run feature engineering to include future returns.",
            FUTURE_RETURN_COLUMN,
        )
        sys.exit(1)

    future_ret = data[FUTURE_RETURN_COLUMN].astype(float)
    relabeled_target = np.select(
        [future_ret > TARGET_UPPER_PCT, future_ret < TARGET_LOWER_PCT],
        [1, 0],
        default=np.nan,
    )
    data[TARGET_COLUMN] = relabeled_target

    noisy_count = int(np.isnan(relabeled_target).sum())
    data = data[data[TARGET_COLUMN].notna()].copy()
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    logger.info(
        "Applied stronger target filter (>%+.1f%% / <%+.1f%%). Removed %d noisy rows.",
        TARGET_UPPER_PCT,
        TARGET_LOWER_PCT,
        noisy_count,
    )
    return data


def validate_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {DATE_COLUMN, SYMBOL_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN, FUTURE_RETURN_COLUMN}
    missing = required - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", sorted(missing))
        sys.exit(1)

    data = df.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])

    if USE_STABLE_STOCKS_ONLY:
        before = len(data)
        data = data[data[SYMBOL_COLUMN].isin(PREFERRED_SYMBOLS)].copy()
        logger.info(
            "Stable-stock filter enabled: kept %d/%d rows for %s",
            len(data), before, sorted(PREFERRED_SYMBOLS),
        )

    data = relabel_target_with_stronger_filter(data)
    data = data.sort_values([SYMBOL_COLUMN, DATE_COLUMN]).reset_index(drop=True)

    nan_count = int(data[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any(axis=1).sum())
    if nan_count > 0:
        logger.warning("Dropping %d rows with NaN in required columns", nan_count)
        data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    if data.empty:
        logger.error("No valid rows after filtering/cleaning.")
        sys.exit(1)

    class_dist = data[TARGET_COLUMN].value_counts().to_dict()
    logger.info(
        "Prepared dataset: %d rows, symbols=%d, date range %s -> %s, class dist=%s",
        len(data),
        data[SYMBOL_COLUMN].nunique(),
        data[DATE_COLUMN].min().date(),
        data[DATE_COLUMN].max().date(),
        class_dist,
    )
    return data


def compute_train_cutoff_date(data: pd.DataFrame) -> pd.Timestamp:
    unique_dates = np.sort(data[DATE_COLUMN].unique())
    cutoff_idx = int(len(unique_dates) * TRAIN_RATIO) - 1
    cutoff_idx = max(0, min(cutoff_idx, len(unique_dates) - 1))
    return pd.Timestamp(unique_dates[cutoff_idx])


def fit_leak_safe_scaler(data: pd.DataFrame, train_cutoff_date: pd.Timestamp) -> MinMaxScaler:
    train_rows = data[data[DATE_COLUMN] <= train_cutoff_date]
    if train_rows.empty:
        logger.error("No rows found in training window before %s", train_cutoff_date.date())
        sys.exit(1)

    scaler = MinMaxScaler()
    scaler.fit(train_rows[FEATURE_COLUMNS].values)
    logger.info("MinMaxScaler fitted on training window only (%d rows)", len(train_rows))
    return scaler


def create_sequences(
    data: pd.DataFrame,
    scaler: MinMaxScaler,
    sequence_length: int,
    train_cutoff_date: pd.Timestamp,
    val_split: float,
) -> SequenceDataset:
    X_train_all, y_train_all, train_dates_all = [], [], []
    X_test, y_test, test_dates = [], [], []

    for symbol, group in data.groupby(SYMBOL_COLUMN):
        group = group.sort_values(DATE_COLUMN).reset_index(drop=True)
        if len(group) <= sequence_length:
            logger.warning("Skipping %s due to insufficient rows (%d)", symbol, len(group))
            continue

        scaled = scaler.transform(group[FEATURE_COLUMNS].values)
        targets = group[TARGET_COLUMN].astype(int).values
        dates = group[DATE_COLUMN].values

        symbol_train = 0
        symbol_test = 0
        for i in range(sequence_length, len(group)):
            seq_x = scaled[i - sequence_length:i]
            seq_y = targets[i]
            seq_date = dates[i]
            if pd.Timestamp(seq_date) <= train_cutoff_date:
                X_train_all.append(seq_x)
                y_train_all.append(seq_y)
                train_dates_all.append(seq_date)
                symbol_train += 1
            else:
                X_test.append(seq_x)
                y_test.append(seq_y)
                test_dates.append(seq_date)
                symbol_test += 1

        logger.info("Sequences for %s -> train=%d test=%d", symbol, symbol_train, symbol_test)

    if len(X_train_all) == 0 or len(X_test) == 0:
        logger.error("No train/test sequences generated.")
        sys.exit(1)

    X_train_all = np.asarray(X_train_all, dtype=np.float32)
    y_train_all = np.asarray(y_train_all, dtype=np.int32)
    train_dates_all = np.asarray(train_dates_all)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)
    test_dates = np.asarray(test_dates)

    val_size = max(1, int(len(X_train_all) * val_split))
    if val_size >= len(X_train_all):
        logger.error("Validation size too large for train set.")
        sys.exit(1)

    X_train = X_train_all[:-val_size]
    y_train = y_train_all[:-val_size]
    train_dates = train_dates_all[:-val_size]
    X_val = X_train_all[-val_size:]
    y_val = y_train_all[-val_size:]
    val_dates = train_dates_all[-val_size:]

    logger.info(
        "Sequence split -> train=%d val=%d test=%d | features=%d",
        len(y_train), len(y_val), len(y_test), X_train.shape[-1],
    )
    return SequenceDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
    )


def binary_focal_loss(gamma: float = 1.5, alpha: float = 0.35):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        cross_entropy = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        modulating = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_factor * modulating * cross_entropy)

    return loss


def build_model(input_shape: tuple[int, int], learning_rate: float, gamma: float, alpha: float) -> Model:
    """
    Bidirectional LSTM + GRU hybrid, designed to avoid trivial constant-output collapse.
    """
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.3)(x)
    x = GRU(32, return_sequences=False)(x)
    x = Dropout(0.25)(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=binary_focal_loss(gamma=gamma, alpha=alpha),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def evaluate_predictions(y_true: np.ndarray, prob_up: np.ndarray, threshold: float) -> dict:
    y_pred = (prob_up >= threshold).astype(int)
    conf = np.maximum(prob_up, 1.0 - prob_up)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob_up)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["DOWN", "UP"]),
        "prediction_distribution": {
            "pred_down": int((y_pred == 0).sum()),
            "pred_up": int((y_pred == 1).sum()),
        },
        "confidence": {
            "mean": float(np.mean(conf)),
            "median": float(np.median(conf)),
            "p90": float(np.percentile(conf, 90)),
        },
    }


def select_best_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> tuple[float, list[dict]]:
    threshold_scores = []
    for th in THRESHOLD_GRID:
        m = evaluate_predictions(y_val, prob_val, th)
        threshold_scores.append(
            {
                "threshold": th,
                "f1_score": m["f1_score"],
                "roc_auc": m["roc_auc"],
                "pred_down": m["prediction_distribution"]["pred_down"],
                "pred_up": m["prediction_distribution"]["pred_up"],
            }
        )

    def score(row: dict) -> tuple:
        predicts_both = int(row["pred_down"] > 0 and row["pred_up"] > 0)
        return (predicts_both, row["f1_score"], row["roc_auc"])

    best = sorted(threshold_scores, key=score, reverse=True)[0]
    logger.info(
        "Selected threshold %.2f using validation F1/ROC (pred_down=%d, pred_up=%d)",
        best["threshold"], best["pred_down"], best["pred_up"],
    )
    return float(best["threshold"]), threshold_scores


def passes_quality_gate(metrics: dict) -> tuple[bool, str]:
    pred_down = metrics["prediction_distribution"]["pred_down"]
    pred_up = metrics["prediction_distribution"]["pred_up"]
    predicts_both = pred_down > 0 and pred_up > 0

    if not predicts_both:
        return False, "Collapsed model detected (single-class predictions)."
    if metrics["roc_auc"] < MIN_ACCEPTABLE_ROC_AUC:
        return False, f"ROC AUC below gate ({metrics['roc_auc']:.4f} < {MIN_ACCEPTABLE_ROC_AUC:.2f})."
    if metrics["f1_score"] < MIN_ACCEPTABLE_F1:
        return False, f"F1-score below gate ({metrics['f1_score']:.4f} < {MIN_ACCEPTABLE_F1:.2f})."
    return True, "Quality gate passed."


def print_report(metrics: dict) -> None:
    cm = np.asarray(metrics["confusion_matrix"])
    print("\n" + "=" * 60)
    print("  LSTM COLLAPSE-RESISTANT EVALUATION")
    print("=" * 60)
    print(f"  Threshold:       {metrics['threshold']:.2f}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1-score:        {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:         {metrics['roc_auc']:.4f}")
    print(f"  Pred DOWN/UP:    {metrics['prediction_distribution']['pred_down']} / {metrics['prediction_distribution']['pred_up']}")
    print("  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print("  Classification Report:")
    print(metrics["classification_report"])
    print("=" * 60)


def load_baseline_model_scores() -> list[dict]:
    if not BASELINE_METADATA_FILE.exists():
        return []
    try:
        data = json.loads(BASELINE_METADATA_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not parse baseline metadata: %s", exc)
        return []

    comparison = data.get("model_comparison")
    if isinstance(comparison, list) and comparison:
        return [
            {
                "model": row.get("model", "Unknown"),
                "accuracy": float(row.get("accuracy", np.nan)),
                "f1_score": float(row.get("f1_score", np.nan)),
                "roc_auc": float(row.get("roc_auc", np.nan)),
            }
            for row in comparison
        ]

    metrics = data.get("metrics", {})
    if metrics:
        return [
            {
                "model": data.get("selected_model") or data.get("model_type", "BaselineModel"),
                "accuracy": float(metrics.get("accuracy", np.nan)),
                "f1_score": float(metrics.get("f1_score", np.nan)),
                "roc_auc": float(metrics.get("roc_auc", np.nan)),
            }
        ]
    return []


def print_model_comparison_report(lstm_metrics: dict) -> list[dict]:
    rows = load_baseline_model_scores()
    rows.append(
        {
            "model": "LSTM",
            "accuracy": lstm_metrics["accuracy"],
            "f1_score": lstm_metrics["f1_score"],
            "roc_auc": lstm_metrics["roc_auc"],
        }
    )
    rows = sorted(rows, key=lambda x: (x["roc_auc"], x["f1_score"], x["accuracy"]), reverse=True)
    print("\n" + "=" * 60)
    print("  FINAL MODEL COMPARISON REPORT")
    print("=" * 60)
    for row in rows:
        print(
            f"{row['model']:<24s} | "
            f"Acc={row['accuracy']:.4f} | F1={row['f1_score']:.4f} | ROC AUC={row['roc_auc']:.4f}"
        )
    print("=" * 60)
    print(f"Best model by ranking: {rows[0]['model']}")
    return rows


def save_artifacts(
    model: Model,
    scaler: MinMaxScaler,
    metrics: dict,
    threshold_scores: list[dict],
    comparison_rows: list[dict],
    train_info: dict,
    args: argparse.Namespace,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(LSTM_MODEL_FILE)
    joblib.dump(scaler, LSTM_SCALER_FILE, compress=3)

    metadata = {
        "model_type": "LSTM-BiLSTM-GRU",
        "feature_columns": FEATURE_COLUMNS,
        "target_logic": {
            "upper_pct": TARGET_UPPER_PCT,
            "lower_pct": TARGET_LOWER_PCT,
            "ignored_band": [TARGET_LOWER_PCT, TARGET_UPPER_PCT],
        },
        "sequence_length": args.sequence_length,
        "threshold_grid": THRESHOLD_GRID,
        "selected_threshold": metrics["threshold"],
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "validation_split": args.validation_split,
            "learning_rate": args.learning_rate,
            "focal_gamma": args.gamma_focal,
            "focal_alpha": args.alpha_focal,
        },
        "dataset_info": train_info,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "prediction_distribution": metrics["prediction_distribution"],
            "prediction_confidence": metrics["confidence"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "threshold_tuning": threshold_scores,
        "model_comparison": comparison_rows,
        "trained_at": datetime.now().isoformat(),
    }

    with open(LSTM_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved -> %s", LSTM_MODEL_FILE)
    logger.info("Scaler saved -> %s", LSTM_SCALER_FILE)
    logger.info("Metadata saved -> %s", LSTM_METADATA_FILE)


def main() -> None:
    args = parse_args()
    logger.info("=" * 60)
    logger.info("LSTM COLLAPSE-FIX TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(
        "Config: seq_len=%d, epochs=%d, batch=%d, val_split=%.2f",
        args.sequence_length, args.epochs, args.batch_size, args.validation_split,
    )

    data_raw = load_data(INPUT_FILE)
    data = validate_and_prepare_dataframe(data_raw)
    train_cutoff_date = compute_train_cutoff_date(data)
    logger.info("Chronological train cutoff: %s", train_cutoff_date.date())

    scaler = fit_leak_safe_scaler(data, train_cutoff_date)
    seq = create_sequences(data, scaler, args.sequence_length, train_cutoff_date, args.validation_split)

    classes = np.array([0, 1], dtype=np.int32)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=seq.y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    logger.info("Class weights (train): %s", class_weight_dict)

    model = build_model(
        input_shape=(args.sequence_length, len(FEATURE_COLUMNS)),
        learning_rate=args.learning_rate,
        gamma=args.gamma_focal,
        alpha=args.alpha_focal,
    )
    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1),
        ModelCheckpoint(filepath=str(CHECKPOINT_FILE), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        seq.X_train,
        seq.y_train,
        validation_data=(seq.X_val, seq.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    logger.info("Training finished in %d epochs", len(history.history.get("loss", [])))

    prob_val = model.predict(seq.X_val, batch_size=args.batch_size, verbose=0).reshape(-1)
    best_threshold, threshold_scores = select_best_threshold(seq.y_val, prob_val)

    prob_test = model.predict(seq.X_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    metrics = evaluate_predictions(seq.y_test, prob_test, best_threshold)
    print_report(metrics)

    ok, reason = passes_quality_gate(metrics)
    if not ok:
        logger.error("Model rejected by quality gate: %s", reason)
        logger.error("Artifacts were NOT saved to avoid deploying collapsed model.")
        sys.exit(2)
    logger.info("Quality gate passed: %s", reason)

    comparison_rows = print_model_comparison_report(metrics)
    train_info = {
        "num_rows_after_filtering": int(len(data)),
        "symbols": sorted(data[SYMBOL_COLUMN].unique().tolist()),
        "date_range": {
            "start": str(data[DATE_COLUMN].min().date()),
            "end": str(data[DATE_COLUMN].max().date()),
            "train_end": str(train_cutoff_date.date()),
        },
        "sequence_counts": {"train": int(len(seq.y_train)), "val": int(len(seq.y_val)), "test": int(len(seq.y_test))},
        "target_distribution": {
            "train_up_pct": float(seq.y_train.mean() * 100),
            "val_up_pct": float(seq.y_val.mean() * 100),
            "test_up_pct": float(seq.y_test.mean() * 100),
        },
    }

    save_artifacts(
        model=model,
        scaler=scaler,
        metrics=metrics,
        threshold_scores=threshold_scores,
        comparison_rows=comparison_rows,
        train_info=train_info,
        args=args,
    )
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("LSTM training failed: %s", exc)
        sys.exit(1)
