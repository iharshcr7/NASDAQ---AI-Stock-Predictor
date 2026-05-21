"""
model_config.py
===============
Shared production configuration and validation for final Random Forest system.

UPDATED: Removed stable stock restrictions - now supports all stocks dynamically.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_FILE = MODELS_DIR / "final_random_forest.pkl"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

SYMBOL_COLUMN = "Symbol"
DATE_COLUMN = "Date"
TARGET_COLUMN = "Target"

# REMOVED: STABLE_SYMBOLS restriction
# The system now supports all stocks dynamically from the dataset
# No hardcoded symbol filtering needed

FINAL_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA5",
    "MA10",
    "MA20",
    "EMA12",
    "RSI",
    "MACD",
    "MACD_Signal",
    "Daily_Returns",
    "Volatility",
    "Price_Change_Pct",
    "Weekly_Momentum",
    "Trend_Strength",
    "BB_Width",
    "Avg_5D_Volume_Trend",
    "Lag_1",
    "Lag_3",
]


def read_model_metadata() -> dict:
    if not MODEL_METADATA_FILE.exists():
        return {}
    try:
        return json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_expected_features() -> list[str]:
    metadata = read_model_metadata()
    features = metadata.get("feature_columns")
    if isinstance(features, list) and features:
        return features
    return FINAL_FEATURE_COLUMNS


def validate_feature_schema(feature_columns: list[str]) -> None:
    expected = get_expected_features()
    if feature_columns != expected:
        raise ValueError(
            "Feature schema mismatch. "
            f"Expected {expected} but got {feature_columns}."
        )

