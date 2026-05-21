"""
preprocessing.py
=================
Clean and validate the merged stock dataset for downstream feature
engineering and model training.

Usage:
    python scripts/preprocessing.py
"""

import sys
import logging
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "merged_stock_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "cleaned_stock_data.csv"

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]
NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

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


def load_merged_data(filepath: Path) -> pd.DataFrame:
    """Load the merged CSV and perform initial validation."""
    if not filepath.exists():
        logger.error("Input file not found: %s", filepath)
        logger.error("Run merge_stock_data.py first.")
        sys.exit(1)

    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), filepath.name)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure all required columns are present."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", missing)
        sys.exit(1)
    logger.info("Column validation passed")


def convert_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the Date column to datetime, dropping any unparseable rows."""
    initial_len = len(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = df["Date"].isna().sum()

    if invalid_dates > 0:
        logger.warning("Dropping %d rows with invalid dates", invalid_dates)
        df = df.dropna(subset=["Date"])

    logger.info(
        "Date conversion: %d → %d rows (dropped %d)",
        initial_len, len(df), initial_len - len(df),
    )
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns and report any conversion issues."""
    for col in NUMERIC_COLUMNS:
        before_nulls = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_nulls = df[col].isna().sum()
        new_nulls = after_nulls - before_nulls
        if new_nulls > 0:
            logger.warning("Column '%s': %d values coerced to NaN", col, new_nulls)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (Symbol, Date) pairs, keeping the last occurrence."""
    initial_len = len(df)
    df = df.drop_duplicates(subset=["Symbol", "Date"], keep="last")
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info("Removed %d duplicate rows", dropped)
    else:
        logger.info("No duplicate rows found")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with a two-step strategy:
    1. Forward-fill within each stock group (carries last known value)
    2. Drop any remaining rows with NaNs (e.g., leading NaNs)
    """
    initial_len = len(df)
    null_before = df[NUMERIC_COLUMNS].isna().sum().sum()
    logger.info("Null values before handling: %d", null_before)

    # Forward-fill within each stock group
    df[NUMERIC_COLUMNS] = df.groupby("Symbol")[NUMERIC_COLUMNS].transform(
        lambda x: x.ffill()
    )

    null_after_ffill = df[NUMERIC_COLUMNS].isna().sum().sum()
    logger.info("Null values after forward-fill: %d", null_after_ffill)

    # Drop remaining NaN rows
    df = df.dropna(subset=NUMERIC_COLUMNS)
    dropped = initial_len - len(df)

    if dropped > 0:
        logger.info("Dropped %d rows with remaining NaN values", dropped)

    return df


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by Symbol, then by Date ascending."""
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    logger.info("Data sorted by [Symbol, Date]")
    return df


def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns not needed for feature engineering."""
    drop_cols = [c for c in ["Adj Close"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info("Removed unnecessary columns: %s", drop_cols)
    return df


def filter_low_data_stocks(df: pd.DataFrame, min_rows: int = 100) -> pd.DataFrame:
    """Remove stocks with fewer than min_rows data points."""
    counts = df.groupby("Symbol").size()
    low_data = counts[counts < min_rows].index.tolist()
    if low_data:
        logger.warning(
            "Removing %d stocks with < %d rows: %s",
            len(low_data), min_rows, low_data,
        )
        df = df[~df["Symbol"].isin(low_data)]
    return df


def validate_price_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where prices are non-positive or High < Low."""
    initial_len = len(df)

    # Non-positive prices
    price_cols = ["Open", "High", "Low", "Close"]
    mask_positive = (df[price_cols] > 0).all(axis=1)

    # High >= Low
    mask_hl = df["High"] >= df["Low"]

    # Volume >= 0
    mask_vol = df["Volume"] >= 0

    df = df[mask_positive & mask_hl & mask_vol].copy()
    dropped = initial_len - len(df)

    if dropped > 0:
        logger.warning("Removed %d rows failing price sanity checks", dropped)
    else:
        logger.info("Price sanity validation passed")

    return df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    validate_columns(df)
    df = convert_date_column(df)
    df = convert_numeric_columns(df)
    df = remove_duplicates(df)
    df = sort_data(df)
    df = handle_missing_values(df)
    df = remove_unnecessary_columns(df)
    df = validate_price_sanity(df)
    df = filter_low_data_stocks(df)
    df = df.reset_index(drop=True)
    return df


def main() -> None:
    logger.info("=" * 60)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    df = load_merged_data(INPUT_FILE)
    df = preprocess(df)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    # Summary
    logger.info("-" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("-" * 60)
    logger.info("Output rows:      %d", len(df))
    logger.info("Output columns:   %s", list(df.columns))
    logger.info("Symbols:          %d unique", df["Symbol"].nunique())
    logger.info("Date range:       %s → %s", df["Date"].min(), df["Date"].max())
    logger.info("Null values:      %d", df.isna().sum().sum())
    logger.info("File size:        %.2f MB", size_mb)
    logger.info("Output:           %s", OUTPUT_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
