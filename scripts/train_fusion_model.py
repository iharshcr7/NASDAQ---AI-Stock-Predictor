"""
train_fusion_model.py
======================
Train Random Forest model on fused structured + unstructured features.

This script trains the final production model that combines:
- 21 structured features (technical indicators from Spark)
- 128 CNN features (extracted from candlestick images)
- Total: 149 features

The fusion model should outperform the baseline structured-only model
by capturing visual patterns in addition to numerical indicators.

Output:
    - models/fusion_random_forest.pkl
    - models/fusion_metadata.json

Usage:
    python scripts/train_fusion_model.py
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
FUSION_FEATURES_CSV = PROJECT_ROOT / "data" / "fusion_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Output files
FUSION_MODEL_FILE = MODELS_DIR / "fusion_random_forest.pkl"
FUSION_METADATA_FILE = MODELS_DIR / "fusion_metadata.json"

# Baseline model for comparison
BASELINE_MODEL_FILE = MODELS_DIR / "final_random_forest.pkl"
BASELINE_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Training configuration
SYMBOL_COLUMN = "Symbol"
DATE_COLUMN = "Date"
TARGET_COLUMN = "Target"
TEST_SIZE = 0.2
CV_SPLITS = 5
CLASS_IMBALANCE_THRESHOLD = 1.2

# Structured feature columns (21 features)
STRUCTURED_FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA10", "MA20", "EMA12",
    "RSI", "MACD", "MACD_Signal",
    "Daily_Returns", "Volatility", "Price_Change_Pct",
    "Weekly_Momentum", "Trend_Strength", "BB_Width",
    "Avg_5D_Volume_Trend", "Lag_1", "Lag_3",
]

# CNN feature columns (128 features)
CNN_FEATURE_COLUMNS = [f"cnn_feature_{i}" for i in range(128)]

# All feature columns (149 total)
ALL_FEATURE_COLUMNS = STRUCTURED_FEATURE_COLUMNS + CNN_FEATURE_COLUMNS

# Random Forest hyperparameters (optimized for fusion)
RF_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 25,
    "min_samples_split": 8,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_fusion_dataset() -> pd.DataFrame:
    """Load fusion features dataset."""
    if not FUSION_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Fusion features not found: {FUSION_FEATURES_CSV}\n"
            f"Run: python scripts/fusion_feature_engineering.py"
        )
    
    logger.info(f"Loading fusion dataset: {FUSION_FEATURES_CSV}")
    df = pd.read_csv(FUSION_FEATURES_CSV)
    
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    return df


def validate_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare fusion dataset for training."""
    logger.info("Validating dataset...")
    
    # Check required columns
    required = {SYMBOL_COLUMN, DATE_COLUMN, TARGET_COLUMN, *ALL_FEATURE_COLUMNS}
    missing = required - set(df.columns)
    
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    
    # Convert Date to datetime
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    
    # Validate target values
    target_values = set(df[TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Invalid target values: {sorted(target_values)}. Expected {{0, 1}}")
    
    # Sort by date and symbol
    df = df.sort_values([DATE_COLUMN, SYMBOL_COLUMN]).reset_index(drop=True)
    
    # Remove invalid rows
    invalid_mask = df[ALL_FEATURE_COLUMNS + [TARGET_COLUMN]].replace(
        [np.inf, -np.inf], np.nan
    ).isna().any(axis=1)
    
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        logger.warning(f"Removing {invalid_count} rows with NaN/Inf values")
        df = df[~invalid_mask].reset_index(drop=True)
    
    if df.empty:
        raise RuntimeError("No valid rows remain after cleaning")
    
    logger.info(f"Validated dataset: {len(df):,} rows")
    logger.info(f"  Symbols: {df[SYMBOL_COLUMN].nunique()}")
    logger.info(f"  Date range: {df[DATE_COLUMN].min().date()} to {df[DATE_COLUMN].max().date()}")
    logger.info(f"  Features: {len(ALL_FEATURE_COLUMNS)} (structured={len(STRUCTURED_FEATURE_COLUMNS)}, CNN={len(CNN_FEATURE_COLUMNS)})")
    
    return df


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------


def split_chronologically(data: pd.DataFrame):
    """Split data chronologically for time-series-safe evaluation."""
    X = data[ALL_FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].astype(int).copy()
    dates = data[DATE_COLUMN].copy()
    
    split_idx = int(len(data) * (1 - TEST_SIZE))
    split_idx = max(1, min(split_idx, len(data) - 1))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    train_dates, test_dates = dates.iloc[:split_idx], dates.iloc[split_idx:]
    
    logger.info(f"Chronological split: {len(X_train):,} train / {len(X_test):,} test")
    logger.info(f"Train window: {train_dates.iloc[0].date()} → {train_dates.iloc[-1].date()}")
    logger.info(f"Test window:  {test_dates.iloc[0].date()} → {test_dates.iloc[-1].date()}")
    
    return X_train, X_test, y_train, y_test


def get_class_weight(y_train: pd.Series) -> str | None:
    """Determine if class weighting is needed."""
    class_counts = y_train.value_counts().sort_index()
    down_count = int(class_counts.get(0, 0))
    up_count = int(class_counts.get(1, 0))
    ratio = max(down_count, up_count) / max(1, min(down_count, up_count))
    
    logger.info(f"Target distribution (train): DOWN={down_count:,} UP={up_count:,} | ratio={ratio:.3f}")
    
    if ratio >= CLASS_IMBALANCE_THRESHOLD:
        logger.info("Imbalance detected. Using class_weight='balanced'")
        return "balanced"
    
    logger.info("Class distribution acceptable. class_weight=None")
    return None


def train_fusion_model(X_train, y_train, class_weight):
    """Train Random Forest on fusion features."""
    logger.info("Training Fusion Random Forest...")
    
    params = {**RF_PARAMS, "class_weight": class_weight}
    logger.info(f"Hyperparameters: {params}")
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info("Training complete")
    
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics."""
    logger.info("Evaluating model on test set...")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["DOWN", "UP"]
        ),
    }
    
    # Prediction confidence
    confidence = np.maximum(y_proba, 1.0 - y_proba)
    metrics["prediction_confidence"] = {
        "mean": float(np.mean(confidence)),
        "median": float(np.median(confidence)),
        "p90": float(np.percentile(confidence, 90)),
    }
    
    logger.info(f"Test Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  Precision:      {metrics['precision']:.4f}")
    logger.info(f"  Recall:         {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:       {metrics['f1_score']:.4f}")
    logger.info(f"  ROC AUC:        {metrics['roc_auc']:.4f}")
    
    return metrics


def run_time_series_cv(model_params, X_train, y_train) -> np.ndarray:
    """Run time-series cross-validation."""
    logger.info(f"Running {CV_SPLITS}-fold TimeSeriesSplit cross-validation...")
    
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
        
        logger.info(f"  Fold {fold_id}/{CV_SPLITS}: accuracy={fold_acc:.4f}")
    
    logger.info(f"CV Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return np.array(scores)


def analyze_feature_importance(model, feature_names) -> pd.DataFrame:
    """Analyze and return feature importance."""
    logger.info("Analyzing feature importance...")
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
        "type": ["structured" if f in STRUCTURED_FEATURE_COLUMNS else "cnn" for f in feature_names]
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    # Calculate importance by type
    structured_importance = importance_df[importance_df["type"] == "structured"]["importance"].sum()
    cnn_importance = importance_df[importance_df["type"] == "cnn"]["importance"].sum()
    
    logger.info(f"Feature importance analysis:")
    logger.info(f"  Structured features: {structured_importance:.4f} ({structured_importance*100:.1f}%)")
    logger.info(f"  CNN features:        {cnn_importance:.4f} ({cnn_importance*100:.1f}%)")
    
    logger.info("\nTop 15 most important features:")
    for idx, row in importance_df.head(15).iterrows():
        logger.info(f"  {idx+1:2d}. {row['feature']:30s} {row['importance']:.6f} [{row['type']}]")
    
    return importance_df


def compare_with_baseline(fusion_metrics: dict) -> dict:
    """Compare fusion model with baseline structured-only model."""
    logger.info("\nComparing with baseline model...")
    
    comparison = {
        "fusion_model": {
            "accuracy": fusion_metrics["accuracy"],
            "f1_score": fusion_metrics["f1_score"],
            "roc_auc": fusion_metrics["roc_auc"],
            "features": len(ALL_FEATURE_COLUMNS),
        },
        "baseline_model": None,
        "improvement": None,
    }
    
    if BASELINE_METADATA_FILE.exists():
        try:
            with open(BASELINE_METADATA_FILE, "r") as f:
                baseline_meta = json.load(f)
            
            baseline_metrics = baseline_meta.get("metrics", {})
            comparison["baseline_model"] = {
                "accuracy": baseline_metrics.get("accuracy", 0),
                "f1_score": baseline_metrics.get("f1_score", 0),
                "roc_auc": baseline_metrics.get("roc_auc", 0),
                "features": len(STRUCTURED_FEATURE_COLUMNS),
            }
            
            # Calculate improvement
            comparison["improvement"] = {
                "accuracy": fusion_metrics["accuracy"] - baseline_metrics.get("accuracy", 0),
                "f1_score": fusion_metrics["f1_score"] - baseline_metrics.get("f1_score", 0),
                "roc_auc": fusion_metrics["roc_auc"] - baseline_metrics.get("roc_auc", 0),
            }
            
            logger.info("Baseline vs Fusion comparison:")
            logger.info(f"  Baseline Accuracy: {comparison['baseline_model']['accuracy']:.4f}")
            logger.info(f"  Fusion Accuracy:   {comparison['fusion_model']['accuracy']:.4f}")
            logger.info(f"  Improvement:       {comparison['improvement']['accuracy']:+.4f} ({comparison['improvement']['accuracy']*100:+.2f}%)")
            logger.info(f"  Baseline ROC AUC:  {comparison['baseline_model']['roc_auc']:.4f}")
            logger.info(f"  Fusion ROC AUC:    {comparison['fusion_model']['roc_auc']:.4f}")
            logger.info(f"  Improvement:       {comparison['improvement']['roc_auc']:+.4f}")
            
        except Exception as e:
            logger.warning(f"Could not load baseline metadata: {e}")
    else:
        logger.warning("Baseline model metadata not found")
    
    return comparison


def save_fusion_model(model, metrics, cv_scores, importance_df, comparison, params):
    """Save fusion model and metadata."""
    logger.info("\nSaving fusion model and metadata...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, FUSION_MODEL_FILE, compress=3)
    model_size_mb = FUSION_MODEL_FILE.stat().st_size / (1024 ** 2)
    logger.info(f"Model saved: {FUSION_MODEL_FILE} ({model_size_mb:.2f} MB)")
    
    # Build metadata
    metadata = {
        "model_type": "RandomForestClassifier_Fusion",
        "description": "Fusion model combining structured (21) + CNN (128) features",
        "model_file": str(FUSION_MODEL_FILE),
        "feature_columns": ALL_FEATURE_COLUMNS,
        "feature_breakdown": {
            "structured_features": STRUCTURED_FEATURE_COLUMNS,
            "cnn_features": CNN_FEATURE_COLUMNS,
            "total_features": len(ALL_FEATURE_COLUMNS),
        },
        "target_column": TARGET_COLUMN,
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
        "feature_importance": {
            "top_20": importance_df.head(20).to_dict(orient="records"),
            "structured_total": float(importance_df[importance_df["type"] == "structured"]["importance"].sum()),
            "cnn_total": float(importance_df[importance_df["type"] == "cnn"]["importance"].sum()),
        },
        "baseline_comparison": comparison,
        "prediction_config": {
            "labels": {"0": "DOWN", "1": "UP"},
            "confidence_source": "predict_proba_max_class_probability",
        },
        "trained_at": datetime.now().isoformat(),
    }
    
    # Save metadata
    with open(FUSION_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved: {FUSION_METADATA_FILE}")


def print_final_report(metrics, cv_scores, comparison):
    """Print final training report."""
    cm = np.array(metrics["confusion_matrix"])
    
    print("\n" + "=" * 70)
    print("  FUSION MODEL TRAINING REPORT")
    print("=" * 70)
    print(f"  Model Type:      Random Forest (Fusion)")
    print(f"  Total Features:  {len(ALL_FEATURE_COLUMNS)} (structured={len(STRUCTURED_FEATURE_COLUMNS)}, CNN={len(CNN_FEATURE_COLUMNS)})")
    print(f"  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1-Score:        {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:         {metrics['roc_auc']:.4f}")
    print(f"  CV Mean:         {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    
    if comparison.get("baseline_model"):
        print(f"\n  Baseline Comparison:")
        print(f"    Baseline Accuracy: {comparison['baseline_model']['accuracy']:.4f}")
        print(f"    Fusion Accuracy:   {comparison['fusion_model']['accuracy']:.4f}")
        print(f"    Improvement:       {comparison['improvement']['accuracy']:+.4f} ({comparison['improvement']['accuracy']*100:+.2f}%)")
    
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        logger.info("=" * 70)
        logger.info("FUSION MODEL TRAINING PIPELINE")
        logger.info("=" * 70)
        
        # Load and prepare data
        df = load_fusion_dataset()
        data = validate_and_prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_chronologically(data)
        
        # Determine class weight
        class_weight = get_class_weight(y_train)
        params = {**RF_PARAMS, "class_weight": class_weight}
        
        # Train model
        model = train_fusion_model(X_train, y_train, class_weight)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Cross-validation
        cv_scores = run_time_series_cv(params, X_train, y_train)
        
        # Feature importance
        importance_df = analyze_feature_importance(model, ALL_FEATURE_COLUMNS)
        
        # Compare with baseline
        comparison = compare_with_baseline(metrics)
        
        # Save model and metadata
        save_fusion_model(model, metrics, cv_scores, importance_df, comparison, params)
        
        # Print final report
        print_final_report(metrics, cv_scores, comparison)
        
        logger.info("\nFUSION MODEL TRAINING COMPLETE")
        logger.info(f"Model: {FUSION_MODEL_FILE}")
        logger.info(f"Metadata: {FUSION_METADATA_FILE}")
        logger.info("\nNext step: Use fusion model for live predictions")
        logger.info("  python scripts/predict_live_fusion.py --symbol AAPL")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
