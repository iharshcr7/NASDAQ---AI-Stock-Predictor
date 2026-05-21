"""
fusion_feature_engineering.py
==============================
Combine structured features (from Spark) with unstructured CNN features.

This script:
1. Loads structured features from data/final_featured_data.csv
2. Loads CNN features from data/cnn_features.csv
3. Merges on (symbol, date) with proper alignment
4. Validates feature consistency
5. Saves combined features to data/fusion_features.csv

Output Schema:
    - Symbol, Date, Target (metadata + label)
    - 21 structured features (MA, RSI, MACD, etc.)
    - 128 CNN features (cnn_feature_0 to cnn_feature_127)
    - Total: 149 features for model training

Usage:
    python scripts/fusion_feature_engineering.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

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

# Input files
STRUCTURED_FEATURES_CSV = PROJECT_ROOT / "data" / "final_featured_data.csv"
CNN_FEATURES_CSV = PROJECT_ROOT / "data" / "cnn_features.csv"

# Output file
FUSION_FEATURES_CSV = PROJECT_ROOT / "data" / "fusion_features.csv"

# Expected structured feature columns (from model_config.py)
STRUCTURED_FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA10", "MA20", "EMA12",
    "RSI", "MACD", "MACD_Signal",
    "Daily_Returns", "Volatility", "Price_Change_Pct",
    "Weekly_Momentum", "Trend_Strength", "BB_Width",
    "Avg_5D_Volume_Trend", "Lag_1", "Lag_3",
]

# Metadata columns
METADATA_COLUMNS = ["Symbol", "Date", "Target"]

# CNN feature columns (128 features)
CNN_FEATURE_COLUMNS = [f"cnn_feature_{i}" for i in range(128)]

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_structured_features() -> pd.DataFrame:
    """Load structured features from Spark output."""
    if not STRUCTURED_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Structured features not found: {STRUCTURED_FEATURES_CSV}\n"
            f"Run: python scripts/feature_engineering.py"
        )
    
    logger.info(f"Loading structured features: {STRUCTURED_FEATURES_CSV}")
    df = pd.read_csv(STRUCTURED_FEATURES_CSV)
    
    # Validate required columns
    required = set(METADATA_COLUMNS + STRUCTURED_FEATURE_COLUMNS)
    missing = required - set(df.columns)
    
    if missing:
        raise ValueError(f"Structured features missing columns: {sorted(missing)}")
    
    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    logger.info(f"  Symbols: {df['Symbol'].nunique()}")
    logger.info(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df


def load_cnn_features() -> pd.DataFrame:
    """Load CNN features from feature extraction output."""
    if not CNN_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"CNN features not found: {CNN_FEATURES_CSV}\n"
            f"Run: python scripts/extract_cnn_features.py"
        )
    
    logger.info(f"Loading CNN features: {CNN_FEATURES_CSV}")
    df = pd.read_csv(CNN_FEATURES_CSV)
    
    # Validate required columns
    required = {"symbol", "date"} | set(CNN_FEATURE_COLUMNS)
    missing = required - set(df.columns)
    
    if missing:
        raise ValueError(f"CNN features missing columns: {sorted(missing)}")
    
    # Normalize column names for merging
    df = df.rename(columns={"symbol": "Symbol", "date": "Date"})
    
    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    logger.info(f"  Symbols: {df['Symbol'].nunique()}")
    logger.info(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df


# ---------------------------------------------------------------------------
# Feature Fusion
# ---------------------------------------------------------------------------


def merge_features(
    structured_df: pd.DataFrame,
    cnn_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge structured and CNN features on (Symbol, Date).
    
    Uses inner join to ensure only rows with both feature types are kept.
    """
    logger.info("Merging structured and CNN features...")
    logger.info(f"  Structured rows: {len(structured_df):,}")
    logger.info(f"  CNN rows:        {len(cnn_df):,}")
    
    # Perform inner join on Symbol and Date
    merged_df = pd.merge(
        structured_df,
        cnn_df,
        on=["Symbol", "Date"],
        how="inner",
        suffixes=("", "_cnn")
    )
    
    logger.info(f"Merged result:   {len(merged_df):,} rows")
    
    if len(merged_df) == 0:
        raise RuntimeError(
            "Merge resulted in 0 rows! "
            "Check that Symbol and Date values match between datasets."
        )
    
    # Calculate merge statistics
    structured_coverage = 100 * len(merged_df) / len(structured_df)
    cnn_coverage = 100 * len(merged_df) / len(cnn_df)
    
    logger.info(f"  Coverage: {structured_coverage:.1f}% of structured, {cnn_coverage:.1f}% of CNN")
    
    return merged_df


def validate_fusion_features(df: pd.DataFrame) -> None:
    """Validate the fused feature dataset."""
    logger.info("Validating fusion features...")
    
    # Check for required columns
    required_cols = METADATA_COLUMNS + STRUCTURED_FEATURE_COLUMNS + CNN_FEATURE_COLUMNS
    missing = set(required_cols) - set(df.columns)
    
    if missing:
        raise ValueError(f"Fusion dataset missing columns: {sorted(missing)}")
    
    # Check for NaN/Inf values
    feature_cols = STRUCTURED_FEATURE_COLUMNS + CNN_FEATURE_COLUMNS
    
    nan_counts = df[feature_cols].isna().sum()
    total_nans = nan_counts.sum()
    
    if total_nans > 0:
        logger.warning(f"Found {total_nans} NaN values in features")
        logger.warning("Columns with NaNs:")
        for col, count in nan_counts[nan_counts > 0].items():
            logger.warning(f"  {col}: {count}")
    
    # Check for Inf values
    inf_mask = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).any(axis=1)
    inf_count = inf_mask.sum()
    
    if inf_count > 0:
        logger.warning(f"Found {inf_count} rows with Inf values")
    
    # Check target distribution
    if "Target" in df.columns:
        target_dist = df["Target"].value_counts().sort_index()
        logger.info("Target distribution:")
        for label, count in target_dist.items():
            logger.info(f"  {label}: {count:,} ({100*count/len(df):.1f}%)")
    
    logger.info("Validation complete")


def clean_fusion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean fusion features by removing invalid rows."""
    logger.info("Cleaning fusion features...")
    
    initial_rows = len(df)
    
    # Remove rows with NaN or Inf in features
    feature_cols = STRUCTURED_FEATURE_COLUMNS + CNN_FEATURE_COLUMNS
    
    # Replace Inf with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with any NaN in features or target
    df = df.dropna(subset=feature_cols + ["Target"])
    
    final_rows = len(df)
    removed = initial_rows - final_rows
    
    if removed > 0:
        logger.info(f"  Removed {removed:,} rows with invalid values ({100*removed/initial_rows:.1f}%)")
    
    logger.info(f"Clean dataset: {final_rows:,} rows")
    
    return df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        logger.info("=" * 70)
        logger.info("FUSION FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 70)
        
        # Step 1: Load structured features
        logger.info("\nStep 1: Loading structured features...")
        structured_df = load_structured_features()
        
        # Step 2: Load CNN features
        logger.info("\nStep 2: Loading CNN features...")
        cnn_df = load_cnn_features()
        
        # Step 3: Merge features
        logger.info("\nStep 3: Merging features...")
        fusion_df = merge_features(structured_df, cnn_df)
        
        # Step 4: Validate features
        logger.info("\nStep 4: Validating features...")
        validate_fusion_features(fusion_df)
        
        # Step 5: Clean features
        logger.info("\nStep 5: Cleaning features...")
        fusion_df = clean_fusion_features(fusion_df)
        
        # Step 6: Sort and organize columns
        logger.info("\nStep 6: Organizing columns...")
        
        # Column order: Metadata, Structured Features, CNN Features
        column_order = METADATA_COLUMNS + STRUCTURED_FEATURE_COLUMNS + CNN_FEATURE_COLUMNS
        fusion_df = fusion_df[column_order]
        
        # Sort by Symbol and Date
        fusion_df = fusion_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
        
        # Step 7: Save fusion features
        logger.info("\nStep 7: Saving fusion features...")
        FUSION_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
        fusion_df.to_csv(FUSION_FEATURES_CSV, index=False)
        
        file_size_mb = FUSION_FEATURES_CSV.stat().st_size / (1024 ** 2)
        
        logger.info(f"Fusion features saved: {FUSION_FEATURES_CSV}")
        logger.info(f"  Rows: {len(fusion_df):,}")
        logger.info(f"  Columns: {len(fusion_df.columns)}")
        logger.info(f"  File size: {file_size_mb:.2f} MB")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("FUSION FEATURE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total samples:          {len(fusion_df):,}")
        logger.info(f"Structured features:    {len(STRUCTURED_FEATURE_COLUMNS)}")
        logger.info(f"CNN features:           {len(CNN_FEATURE_COLUMNS)}")
        logger.info(f"Total features:         {len(STRUCTURED_FEATURE_COLUMNS) + len(CNN_FEATURE_COLUMNS)}")
        logger.info(f"Unique symbols:         {fusion_df['Symbol'].nunique()}")
        logger.info(f"Date range:             {fusion_df['Date'].min().date()} to {fusion_df['Date'].max().date()}")
        
        # Target distribution
        if "Target" in fusion_df.columns:
            target_dist = fusion_df["Target"].value_counts().sort_index()
            logger.info(f"Target distribution:")
            for label, count in target_dist.items():
                label_name = "DOWN" if label == 0 else "UP"
                logger.info(f"  {label_name} ({label}): {count:,} ({100*count/len(fusion_df):.1f}%)")
        
        logger.info(f"\nOutput file:            {FUSION_FEATURES_CSV}")
        logger.info("=" * 70)
        
        # Show sample
        logger.info("\nSample fusion features (first 5 rows, first 15 columns):")
        print(fusion_df.iloc[:5, :15].to_string(index=False))
        
        logger.info("\nFUSION FEATURE ENGINEERING COMPLETE")
        logger.info("\nNext step: Train fusion model")
        logger.info("  python scripts/train_fusion_model.py")
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fusion feature engineering failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
