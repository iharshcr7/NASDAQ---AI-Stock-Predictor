"""
merge_stock_data.py
====================
Merge multiple NASDAQ stock CSV files from the dataset into a single
master DataFrame suitable for downstream preprocessing and ML training.

NOW SUPPORTS: Dynamic discovery of all stocks in the dataset directory.
No hardcoded symbol lists needed!

Usage:
    # Merge all available stocks
    python scripts/merge_stock_data.py
    
    # Merge specific stocks
    python scripts/merge_stock_data.py --symbols AAPL MSFT GOOGL
    
    # Merge first N stocks (for testing)
    python scripts/merge_stock_data.py --limit 50
"""

import os
import sys
import logging
import argparse
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Read, validate, and concatenate stock CSVs into one DataFrame.
    
    NOW SUPPORTS: Dynamic discovery of all stocks if symbols=None

    Parameters
    ----------
    stocks_dir : Path
        Directory containing individual stock CSV files.
    symbols : list[str] or None
        Specific symbols to load. If None, loads ALL available stocks.
        Any symbol whose CSV is not found is skipped with a warning.
    limit : int or None
        Maximum number of stocks to load (useful for testing).
        If None, loads all requested stocks.

    Returns
    -------
    pd.DataFrame
        Concatenated master DataFrame with a 'Symbol' column.
    """
    available = set(discover_available_symbols(stocks_dir))
    
    # If no symbols specified, use ALL available stocks
    if symbols is None:
        symbols = sorted(available)
        logger.info(
            "No symbols specified - loading ALL %d available stocks from dataset",
            len(symbols)
        )
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        original_count = len(symbols)
        symbols = symbols[:limit]
        logger.info(
            "Limit applied: loading %d of %d stocks",
            len(symbols), original_count
        )
    
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge stock CSV files into a single dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all available stocks (dynamic discovery)
  python scripts/merge_stock_data.py
  
  # Merge specific stocks
  python scripts/merge_stock_data.py --symbols AAPL MSFT GOOGL TSLA
  
  # Merge first 50 stocks (for testing)
  python scripts/merge_stock_data.py --limit 50
  
  # Merge first 100 stocks with custom output
  python scripts/merge_stock_data.py --limit 100 --output data/test_merged.csv
        """
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Specific stock symbols to merge (e.g., AAPL MSFT GOOGL). If not specified, merges ALL available stocks."
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of stocks to merge (useful for testing with large datasets)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help=f"Output CSV file path (default: {OUTPUT_FILE})"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("STOCK DATA MERGE PIPELINE (DYNAMIC DISCOVERY)")
    logger.info("=" * 60)

    if not STOCKS_DIR.exists():
        logger.error("Stocks directory not found: %s", STOCKS_DIR)
        sys.exit(1)

    # Determine output file
    output_file = Path(args.output) if args.output else OUTPUT_FILE
    
    # Merge stocks with dynamic discovery
    merged_df = merge_stocks(
        STOCKS_DIR,
        symbols=args.symbols,
        limit=args.limit
    )
    
    save_dataframe(merged_df, output_file)

    # Summary
    logger.info("-" * 60)
    logger.info("MERGE SUMMARY")
    logger.info("-" * 60)
    logger.info("Symbols included: %d unique stocks", merged_df["Symbol"].nunique())
    logger.info("Sample symbols:   %s", ", ".join(sorted(merged_df["Symbol"].unique())[:10]))
    if merged_df["Symbol"].nunique() > 10:
        logger.info("                  ... and %d more", merged_df["Symbol"].nunique() - 10)
    logger.info("Total rows:       %d", len(merged_df))
    logger.info("Columns:          %s", list(merged_df.columns))
    logger.info("Output file:      %s", output_file)
    logger.info("File size:        %.2f MB", output_file.stat().st_size / (1024 * 1024))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
