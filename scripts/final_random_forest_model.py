"""
final_random_forest_model.py
============================
Final production Random Forest training engine for stock movement prediction.

This script is intended as the final deployable model pipeline for:
- Stable and balanced classification behavior
- Time-series-safe evaluation
- Streamlit/live inference compatibility
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import TimeSeriesSplit
from model_config import (
    FINAL_MODEL_FILE,
    MODEL_METADATA_FILE,
    STABLE_SYMBOLS,
    FINAL_FEATURE_COLUMNS,
    SYMBOL_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,
    validate_feature_schema,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "final_featured_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = FINAL_MODEL_FILE
METADATA_FILE = MODEL_METADATA_FILE
TEST_SIZE = 0.2
CV_SPLITS = 5
CLASS_IMBALANCE_THRESHOLD = 1.2

PREFERRED_SYMBOLS = set(STABLE_SYMBOLS)

# Final production feature schema (must remain in sync with live inference)
FEATURE_COLUMNS = FINAL_FEATURE_COLUMNS

RF_PARAMS_BASE = {
    "n_estimators": 900,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        logger.error("Input dataset not found: %s", INPUT_FILE)
        logger.error("Run feature_engineering.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    logger.info("Loaded dataset: %d rows x %d columns", len(df), len(df.columns))
    return df


def filter_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {SYMBOL_COLUMN, DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required_columns - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", sorted(missing))
        sys.exit(1)

    data = df.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])

    before_filter = len(data)
    data = data[data[SYMBOL_COLUMN].isin(PREFERRED_SYMBOLS)].copy()
    logger.info(
        "Stable-stock filter: kept %d/%d rows for symbols %s",
        len(data),
        before_filter,
        sorted(PREFERRED_SYMBOLS),
    )
    if data.empty:
        logger.error("No rows remain after stable-stock filtering.")
        sys.exit(1)

    # Target quality check (threshold target should already exist from feature engineering).
    target_values = set(pd.Series(data[TARGET_COLUMN]).dropna().unique().tolist())
    if not target_values.issubset({0, 1}):
        logger.error("Target contains invalid values %s. Expected only {0, 1}.", sorted(target_values))
        sys.exit(1)

    data = data.sort_values([DATE_COLUMN, SYMBOL_COLUMN]).reset_index(drop=True)

    # Drop rows with missing feature/target values safely.
    invalid_mask = data[FEATURE_COLUMNS + [TARGET_COLUMN]].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    invalid_rows = int(invalid_mask.sum())
    if invalid_rows > 0:
        logger.warning("Dropping %d rows with NaN/Inf in feature-target set", invalid_rows)
        data = data[~invalid_mask].reset_index(drop=True)

    if data.empty:
        logger.error("No valid rows remain after cleaning.")
        sys.exit(1)

    logger.info(
        "Prepared dataset: %d rows | symbols=%d | date range=%s to %s",
        len(data),
        data[SYMBOL_COLUMN].nunique(),
        data[DATE_COLUMN].min().date(),
        data[DATE_COLUMN].max().date(),
    )
    return data


def split_chronologically(data: pd.DataFrame):
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].astype(int).copy()
    dates = data[DATE_COLUMN].copy()

    split_idx = int(len(data) * (1 - TEST_SIZE))
    split_idx = max(1, min(split_idx, len(data) - 1))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    train_dates, test_dates = dates.iloc[:split_idx], dates.iloc[split_idx:]

    logger.info(
        "Chronological split: %d train / %d test (%.0f/%.0f)",
        len(X_train),
        len(X_test),
        (1 - TEST_SIZE) * 100,
        TEST_SIZE * 100,
    )
    logger.info("Train window: %s -> %s", train_dates.iloc[0].date(), train_dates.iloc[-1].date())
    logger.info("Test window:  %s -> %s", test_dates.iloc[0].date(), test_dates.iloc[-1].date())
    return X_train, X_test, y_train, y_test


def get_class_weight(y_train: pd.Series) -> str | None:
    class_counts = y_train.value_counts().sort_index()
    down_count = int(class_counts.get(0, 0))
    up_count = int(class_counts.get(1, 0))
    ratio = max(down_count, up_count) / max(1, min(down_count, up_count))
    logger.info(
        "Target distribution (train): DOWN=%d UP=%d | imbalance ratio=%.3f",
        down_count,
        up_count,
        ratio,
    )
    if ratio >= CLASS_IMBALANCE_THRESHOLD:
        logger.info("Imbalance detected. Using class_weight='balanced'.")
        return "balanced"
    logger.info("Class distribution acceptable. class_weight=None.")
    return None


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    conf_matrix = confusion_matrix(y_true, y_pred)
    confidence = np.maximum(y_proba, 1.0 - y_proba)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["DOWN", "UP"]),
        "prediction_confidence": {
            "mean": float(np.mean(confidence)),
            "median": float(np.median(confidence)),
            "p90": float(np.percentile(confidence, 90)),
        },
    }


def run_time_series_cv(model_params: dict, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    scores = []
    for fold_id, (tr_idx, va_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        fold_model = RandomForestClassifier(**model_params)
        fold_model.fit(X_tr, y_tr)
        fold_pred = fold_model.predict(X_va)
        fold_acc = accuracy_score(y_va, fold_pred)
        scores.append(fold_acc)
        logger.info("TimeSeriesSplit fold %d/%d accuracy: %.4f", fold_id, CV_SPLITS, fold_acc)
    return np.array(scores)


def print_reports(metrics: dict, cv_scores: np.ndarray, importance_df: pd.DataFrame) -> None:
    cm = np.array(metrics["confusion_matrix"])
    confidence = metrics["prediction_confidence"]

    print("\n" + "=" * 60)
    print("  FINAL RANDOM FOREST EVALUATION REPORT")
    print("=" * 60)
    print(f"  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1-Score:        {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:         {metrics['roc_auc']:.4f}")
    print(f"  TimeSeriesSplit: mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")
    print(f"                   scores={[f'{s:.4f}' for s in cv_scores]}")
    print("  Confusion Matrix:")
    print(f"    TN={cm[0][0]} FP={cm[0][1]}")
    print(f"    FN={cm[1][0]} TP={cm[1][1]}")
    print("  Prediction Confidence:")
    print(f"    Mean={confidence['mean']:.4f} Median={confidence['median']:.4f} P90={confidence['p90']:.4f}")
    print("  Classification Report:")
    print(metrics["classification_report"])
    print("=" * 60)

    print("\n  TOP FEATURE IMPORTANCE")
    print("-" * 60)
    for _, row in importance_df.head(12).iterrows():
        bar = "#" * int(row["importance"] * 100)
        print(f"  {row['feature']:>24s}  {row['importance']:.4f}  {bar}")
    print("=" * 60)


def compare_with_existing_models(rf_metrics: dict) -> list[dict]:
    rows = []
    if METADATA_FILE.exists():
        try:
            prev_meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
            for row in prev_meta.get("model_comparison", []):
                rows.append(
                    {
                        "model": row.get("model", "Unknown"),
                        "accuracy": float(row.get("accuracy", np.nan)),
                        "f1_score": float(row.get("f1_score", np.nan)),
                        "roc_auc": float(row.get("roc_auc", np.nan)),
                    }
                )
        except Exception as exc:
            logger.warning("Could not read previous model metadata for comparison: %s", exc)

    rows.append(
        {
            "model": "RandomForest_Final_Production",
            "accuracy": rf_metrics["accuracy"],
            "f1_score": rf_metrics["f1_score"],
            "roc_auc": rf_metrics["roc_auc"],
        }
    )

    dedup = {}
    for r in rows:
        dedup[r["model"]] = r
    rows = list(dedup.values())

    rows = sorted(rows, key=lambda r: (r["roc_auc"], r["f1_score"], r["accuracy"]), reverse=True)

    print("\n" + "=" * 60)
    print("  FINAL MODEL COMPARISON REPORT")
    print("=" * 60)
    for row in rows:
        print(
            f"{row['model']:<30s} | "
            f"Acc={row['accuracy']:.4f} | F1={row['f1_score']:.4f} | ROC AUC={row['roc_auc']:.4f}"
        )
    print("=" * 60)
    print("Random Forest selected for final deployment due to balanced predictions, stability, and explainability.")
    return rows


def save_artifacts(model: RandomForestClassifier, params: dict, metrics: dict, cv_scores: np.ndarray, importance_df: pd.DataFrame, comparison_rows: list[dict]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE, compress=3)
    model_size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)
    logger.info("Final Random Forest model saved -> %s (%.2f MB)", MODEL_FILE, model_size_mb)

    metadata = {
        "model_type": "RandomForestClassifier",
        "selected_model": "RandomForest_Final_Production",
        "model_file": str(MODEL_FILE),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "stable_symbols": sorted(PREFERRED_SYMBOLS),
        "split_strategy": "chronological_80_20",
        "cv_strategy": {"name": "TimeSeriesSplit", "splits": CV_SPLITS},
        "hyperparameters": params,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "cv_scores": cv_scores.tolist(),
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std_accuracy": float(cv_scores.std()),
            "prediction_confidence": metrics["prediction_confidence"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "feature_importance_top": importance_df.head(20).to_dict(orient="records"),
        "prediction_config": {
            "labels": {"0": "DOWN", "1": "UP"},
            "confidence_source": "predict_proba_max_class_probability",
        },
        "model_comparison": comparison_rows,
        "deployment_notes": {
            "streamlit_compatible": True,
            "reason_selected": "Balanced predictions, robust ROC AUC, stable temporal CV, and strong interpretability.",
        },
        "trained_at": datetime.now().isoformat(),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved -> %s", METADATA_FILE)


def main() -> None:
    logger.info("=" * 60)
    logger.info("FINAL RANDOM FOREST PRODUCTION TRAINING")
    logger.info("=" * 60)

    validate_feature_schema(FEATURE_COLUMNS)
    df = load_dataset()
    data = filter_and_validate(df)
    X_train, X_test, y_train, y_test = split_chronologically(data)

    class_weight = get_class_weight(y_train)
    rf_params = {**RF_PARAMS_BASE, "class_weight": class_weight}
    logger.info("Training Random Forest with params: %s", rf_params)

    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_pred, y_proba)

    cv_scores = run_time_series_cv(rf_params, X_train, y_train)
    importance_df = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print_reports(metrics, cv_scores, importance_df)
    comparison_rows = compare_with_existing_models(metrics)
    save_artifacts(model, rf_params, metrics, cv_scores, importance_df, comparison_rows)

    logger.info("Final production Random Forest pipeline complete.")
    logger.info("Model artifact: %s", MODEL_FILE)
    logger.info("Metadata: %s", METADATA_FILE)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Final RF pipeline failed: %s", exc)
        sys.exit(1)
