"""
merge_stock_data.py
====================
Merge multiple NASDAQ stock CSV files from the dataset into a single
master DataFrame suitable for downstream preprocessing and ML training.

Usage:
    python scripts/merge_stock_data.py
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Top NASDAQ stocks — expand this list freely
DEFAULT_SYMBOLS = [
    "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN",
    "META", "NVDA", "NFLX", "INTC", "AMD",
    "PYPL", "ADBE", "CRM", "CSCO", "QCOM",
    "AVGO", "TXN", "COST", "PEP", "SBUX",
]

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCKS_DIR = PROJECT_ROOT / "data" / "stock_market_dataset" / "stocks"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "merged_stock_data.csv"

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


def discover_available_symbols(stocks_dir: Path) -> list[str]:
    """Return sorted list of all stock symbols available in the directory."""
    return sorted(
        p.stem for p in stocks_dir.glob("*.csv")
    )


def load_stock_csv(filepath: Path, symbol: str) -> pd.DataFrame | None:
    """
    Load a single stock CSV, validate columns, and attach the symbol.

    Returns None if the file is missing required columns or is empty.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", filepath.name, exc)
        return None

    if df.empty:
        logger.warning("Empty file skipped: %s", filepath.name)
        return None

    # Column validation
    present = set(df.columns)
    missing = REQUIRED_COLUMNS - present
    if missing:
        logger.warning(
            "%s missing columns %s — skipped", symbol, missing
        )
        return None

    # Keep only the columns we need + Adj Close if present
    keep = list(REQUIRED_COLUMNS)
    if "Adj Close" in present:
        keep.append("Adj Close")
    df = df[keep].copy()

    # Tag with symbol
    df["Symbol"] = symbol

    logger.info(
        "Loaded %-6s | rows: %7d | date range: %s → %s",
        symbol, len(df), df["Date"].iloc[0], df["Date"].iloc[-1],
    )
    return df


def merge_stocks(
    stocks_dir: Path,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read, validate, and concatenate stock CSVs into one DataFrame.

    Parameters
    ----------
    stocks_dir : Path
        Directory containing individual stock CSV files.
    symbols : list[str] or None
        Specific symbols to load. If None, uses DEFAULT_SYMBOLS.
        Any symbol whose CSV is not found is skipped with a warning.

    Returns
    -------
    pd.DataFrame
        Concatenated master DataFrame with a 'Symbol' column.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    available = set(discover_available_symbols(stocks_dir))
    logger.info(
        "Total CSVs in directory: %d | Requested symbols: %d",
        len(available), len(symbols),
    )

    frames: list[pd.DataFrame] = []

    for sym in symbols:
        if sym not in available:
            logger.warning("Symbol %s not found in dataset — skipped", sym)
            continue

        filepath = stocks_dir / f"{sym}.csv"
        df = load_stock_csv(filepath, sym)
        if df is not None:
            frames.append(df)

    if not frames:
        logger.error("No valid stock data loaded. Aborting.")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    logger.info(
        "Merged dataset: %d rows × %d columns from %d stocks",
        len(merged), len(merged.columns), len(frames),
    )
    return merged


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to CSV, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Saved → %s (%.2f MB)", output_path, size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=" * 60)
    logger.info("STOCK DATA MERGE PIPELINE")
    logger.info("=" * 60)

    if not STOCKS_DIR.exists():
        logger.error("Stocks directory not found: %s", STOCKS_DIR)
        sys.exit(1)

    merged_df = merge_stocks(STOCKS_DIR)
    save_dataframe(merged_df, OUTPUT_FILE)

    # Summary
    logger.info("-" * 60)
    logger.info("MERGE SUMMARY")
    logger.info("-" * 60)
    logger.info("Symbols included: %s", sorted(merged_df["Symbol"].unique()))
    logger.info("Total rows:       %d", len(merged_df))
    logger.info("Columns:          %s", list(merged_df.columns))
    logger.info("Output file:      %s", OUTPUT_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
