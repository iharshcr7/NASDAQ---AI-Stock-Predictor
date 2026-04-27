"""
train_model.py
===============
Train a Random Forest Classifier for stock movement prediction
using a strict time-series workflow.

Usage:
    python scripts/train_model.py
"""

import sys
import json
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - runtime dependency guard
    XGBClassifier = None

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "final_featured_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "best_model.pkl"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Features used for training — must stay in sync with live inference
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA10", "MA20", "Daily_Returns", "Volatility",
    "Price_Change_Pct", "Lag_1", "Lag_3", "RSI",
    "Volume_Change_Pct", "EMA12", "BB_Position",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "Weekly_Momentum", "Avg_Volume_Trend",
    "Avg_5D_Volume_Trend", "Trend_Strength", "Rolling_Std_Returns",
]

TARGET_COLUMN = "Target"
DATE_COLUMN = "Date"

# Tuned Random Forest hyperparameters for better bias-variance balance
RF_PARAMS = {
    "n_estimators": 700,
    "max_depth": 18,
    "min_samples_split": 12,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

TEST_SIZE = 0.2
CV_SPLITS = 5
CLASS_IMBALANCE_THRESHOLD = 1.2

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
# Core Functions
# ---------------------------------------------------------------------------


def load_training_data(filepath: Path) -> pd.DataFrame:
    """Load the feature-engineered dataset."""
    if not filepath.exists():
        logger.error("Training data not found: %s", filepath)
        logger.error("Run feature_engineering.py first.")
        sys.exit(1)

    df = pd.read_csv(filepath)
    logger.info("Loaded training data: %d rows × %d columns", len(df), len(df.columns))
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract feature matrix X and target vector y after strict chronological ordering.
    """
    if DATE_COLUMN not in df.columns:
        logger.error("Date column '%s' not found for chronological split", DATE_COLUMN)
        sys.exit(1)

    missing_features = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_features:
        logger.error("Missing feature columns: %s", missing_features)
        sys.exit(1)

    if TARGET_COLUMN not in df.columns:
        logger.error("Target column '%s' not found", TARGET_COLUMN)
        sys.exit(1)

    ordered_df = df.copy()
    ordered_df[DATE_COLUMN] = pd.to_datetime(ordered_df[DATE_COLUMN])
    ordered_df = ordered_df.sort_values([DATE_COLUMN, "Symbol"]).reset_index(drop=True)

    X = ordered_df[FEATURE_COLUMNS].copy()
    y = ordered_df[TARGET_COLUMN].copy()
    dates = ordered_df[DATE_COLUMN].copy()

    # Replace infinity values with NaN and safely remove affected rows.
    inf_count = np.isinf(X.values).sum()
    if inf_count > 0:
        logger.warning("Replacing %d infinity values with NaN", inf_count)
        X = X.replace([np.inf, -np.inf], np.nan)

    nan_rows = X.isna().any(axis=1).sum()
    if nan_rows > 0:
        logger.warning("Dropping %d rows with NaN/Inf in features", nan_rows)
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]
        dates = dates[mask]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    dates = dates.reset_index(drop=True)

    logger.info("Feature matrix: %d samples × %d features", X.shape[0], X.shape[1])
    logger.info("Date range used: %s → %s", dates.iloc[0].date(), dates.iloc[-1].date())
    logger.info(
        "Target distribution:\n  UP (1): %d (%.1f%%)\n  DOWN (0): %d (%.1f%%)",
        int(y.sum()), y.mean() * 100,
        len(y) - int(y.sum()), (1 - y.mean()) * 100,
    )

    return X, y, dates


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute complete metric suite for a classification model."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=["DOWN", "UP"])
    confidence = np.maximum(y_proba, 1.0 - y_proba)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
        "confidence": {
            "mean": float(np.mean(confidence)),
            "median": float(np.median(confidence)),
            "p90": float(np.percentile(confidence, 90)),
        },
    }


def run_time_series_cv(
    model_name: str,
    model_factory,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> np.ndarray:
    """Run expanding-window TimeSeriesSplit CV for a model factory."""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    scores = []

    for fold_id, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        cv_model = model_factory()
        cv_model.fit(X_cv_train, y_cv_train)
        y_cv_pred = cv_model.predict(X_cv_val)
        fold_acc = accuracy_score(y_cv_val, y_cv_pred)
        scores.append(fold_acc)
        logger.info("%s TS-CV fold %d/%d accuracy: %.4f", model_name, fold_id, CV_SPLITS, fold_acc)

    return np.array(scores)


def get_balancing_params(y_train: pd.Series) -> tuple[dict, float]:
    """
    Determine class-balance strategy based on train-window class ratio.
    Returns RandomForest class_weight config and positive-class weight.
    """
    class_counts = y_train.value_counts().sort_index()
    neg = int(class_counts.get(0, 0))
    pos = int(class_counts.get(1, 0))
    ratio = max(neg, pos) / max(1, min(neg, pos))
    logger.info("Train class counts -> DOWN(0): %d | UP(1): %d | imbalance ratio: %.3f", neg, pos, ratio)

    if ratio >= CLASS_IMBALANCE_THRESHOLD:
        logger.info("Imbalance detected. Enabling balanced class-weight strategy.")
        rf_balance = {"class_weight": "balanced_subsample"}
    else:
        logger.info("Class distribution is near-balanced. Using neutral class weights.")
        rf_balance = {"class_weight": None}

    scale_pos_weight = (neg / max(1, pos))
    return rf_balance, scale_pos_weight


def train_and_select_model(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
) -> tuple[object, dict]:
    """
    Train and compare Random Forest + XGBoost with strict time-series evaluation.
    """
    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    train_dates, test_dates = dates.iloc[:split_index], dates.iloc[split_index:]

    logger.info(
        "Chronological split: %d train / %d test (%.0f/%.0f)",
        len(X_train), len(X_test), (1 - TEST_SIZE) * 100, TEST_SIZE * 100,
    )
    logger.info("Train window: %s → %s", train_dates.iloc[0].date(), train_dates.iloc[-1].date())
    logger.info("Test window:  %s → %s", test_dates.iloc[0].date(), test_dates.iloc[-1].date())

    rf_balance_params, scale_pos_weight = get_balancing_params(y_train)

    rf_params = {**RF_PARAMS, **rf_balance_params}
    logger.info("Training Random Forest Classifier with params: %s", rf_params)
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_predictions(y_test, rf_pred, rf_proba)
    rf_cv_scores = run_time_series_cv(
        "RandomForest",
        lambda: RandomForestClassifier(**rf_params),
        X_train,
        y_train,
    )
    rf_metrics["cv_scores"] = rf_cv_scores.tolist()
    rf_metrics["cv_mean"] = float(rf_cv_scores.mean())
    rf_metrics["cv_std"] = float(rf_cv_scores.std())

    candidate_models = [
        {
            "name": "RandomForestClassifier",
            "estimator": rf_model,
            "params": rf_params,
            "metrics": rf_metrics,
            "feature_importance": getattr(rf_model, "feature_importances_", None),
        }
    ]

    if XGBClassifier is not None:
        xgb_params = {
            "n_estimators": 900,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "gamma": 0.2,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
        }
        logger.info("Training XGBoost Classifier with params: %s", xgb_params)
        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate_predictions(y_test, xgb_pred, xgb_proba)
        xgb_cv_scores = run_time_series_cv(
            "XGBoost",
            lambda: XGBClassifier(**xgb_params),
            X_train,
            y_train,
        )
        xgb_metrics["cv_scores"] = xgb_cv_scores.tolist()
        xgb_metrics["cv_mean"] = float(xgb_cv_scores.mean())
        xgb_metrics["cv_std"] = float(xgb_cv_scores.std())
        candidate_models.append({
            "name": "XGBClassifier",
            "estimator": xgb_model,
            "params": xgb_params,
            "metrics": xgb_metrics,
            "feature_importance": getattr(xgb_model, "feature_importances_", None),
        })
    else:
        logger.warning(
            "xgboost is not installed. Skipping XGBoost comparison. Install via: pip install xgboost"
        )

    # Weighted model selection prioritizing robust discrimination + useful directional precision.
    def selection_score(item: dict) -> float:
        m = item["metrics"]
        return (0.4 * m["accuracy"]) + (0.3 * m["f1_score"]) + (0.3 * m["roc_auc"])

    candidate_models = sorted(candidate_models, key=selection_score, reverse=True)
    best = candidate_models[0]
    best_model = best["estimator"]
    best_metrics = best["metrics"]

    logger.info("Model comparison results:")
    for item in candidate_models:
        m = item["metrics"]
        logger.info(
            "  %s -> accuracy=%.4f f1=%.4f roc_auc=%.4f score=%.4f",
            item["name"], m["accuracy"], m["f1_score"], m["roc_auc"], selection_score(item)
        )
    logger.info("Selected best model: %s", best["name"])

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON REPORT")
    print("=" * 60)
    for item in candidate_models:
        m = item["metrics"]
        print(
            f"{item['name']:<24s} | Acc={m['accuracy']:.4f} | "
            f"F1={m['f1_score']:.4f} | ROC AUC={m['roc_auc']:.4f} | CV={m['cv_mean']:.4f}"
        )
    print("=" * 60)

    conf_matrix = np.array(best_metrics["confusion_matrix"])
    class_report = best_metrics["classification_report"]
    confidence_stats = best_metrics["confidence"]

    print("\n" + "=" * 60)
    print(f"  BEST MODEL EVALUATION REPORT ({best['name']})")
    print("=" * 60)
    print(f"\n  Accuracy:        {best_metrics['accuracy']:.4f}  ({best_metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision:       {best_metrics['precision']:.4f}")
    print(f"  Recall:          {best_metrics['recall']:.4f}")
    print(f"  F1-Score:        {best_metrics['f1_score']:.4f}")
    print(f"  ROC AUC:         {best_metrics['roc_auc']:.4f}")
    print(f"\n  TimeSeriesSplit CV ({CV_SPLITS} folds):")
    print(f"    Scores: {[f'{s:.4f}' for s in best_metrics['cv_scores']]}")
    print(f"    Mean:   {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}")
    print(f"\n  Prediction Confidence:")
    print(f"    Mean:   {confidence_stats['mean']:.4f}")
    print(f"    Median: {confidence_stats['median']:.4f}")
    print(f"    P90:    {confidence_stats['p90']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={conf_matrix[0][0]}  FP={conf_matrix[0][1]}")
    print(f"    FN={conf_matrix[1][0]}  TP={conf_matrix[1][1]}")
    print(f"\n  Classification Report:")
    print(class_report)
    print("=" * 60)

    if best["feature_importance"] is not None:
        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": best["feature_importance"],
        }).sort_values("Importance", ascending=False)

        print("\n  FEATURE IMPORTANCE (Top 10)")
        print("-" * 40)
        for _, row in importance_df.head(10).iterrows():
            bar = "#" * int(row["Importance"] * 100)
            print(f"  {row['Feature']:>24s}  {row['Importance']:.4f}  {bar}")
        print("=" * 60)

    return best_model, {
        **best_metrics,
        "selected_model": best["name"],
        "selected_hyperparameters": best["params"],
        "model_comparison": [
            {
                "model": item["name"],
                "accuracy": item["metrics"]["accuracy"],
                "f1_score": item["metrics"]["f1_score"],
                "roc_auc": item["metrics"]["roc_auc"],
                "cv_mean": item["metrics"]["cv_mean"],
                "selection_score": selection_score(item),
            }
            for item in candidate_models
        ],
    }


def save_model_artifacts(model: object, metrics: dict) -> None:
    """Save the trained model and metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE, compress=3)
    model_size = MODEL_FILE.stat().st_size / (1024 * 1024)
    logger.info("Model saved → %s (%.2f MB)", MODEL_FILE, model_size)

    metadata = {
        "model_type": metrics["selected_model"],
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "selected_model": metrics["selected_model"],
        "selected_hyperparameters": metrics["selected_hyperparameters"],
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "model_comparison": metrics["model_comparison"],
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "cv_mean_accuracy": metrics["cv_mean"],
            "prediction_confidence": metrics["confidence"],
        },
        "trained_at": datetime.now().isoformat(),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved → %s", METADATA_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=" * 60)
    logger.info("MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    df = load_training_data(INPUT_FILE)
    X, y, dates = prepare_features(df)
    model, metrics = train_and_select_model(X, y, dates)
    save_model_artifacts(model, metrics)

    logger.info("✅ Training pipeline complete!")
    logger.info("Model: %s", MODEL_FILE)
    logger.info("Metadata: %s", METADATA_FILE)


if __name__ == "__main__":
    main()
