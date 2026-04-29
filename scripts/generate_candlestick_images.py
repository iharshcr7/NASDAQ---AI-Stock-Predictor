"""
generate_candlestick_images.py
================================
Generate labeled candlestick chart images from stock data for
unstructured data analysis. Images are classified as bullish (UP)
or bearish (DOWN) based on next-day price movement.

Usage:
    python scripts/generate_candlestick_images.py
"""

import sys
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "cleaned_stock_data.csv"
IMAGES_DIR = PROJECT_ROOT / "images"
BULLISH_DIR = IMAGES_DIR / "bullish"
BEARISH_DIR = IMAGES_DIR / "bearish"

# Chart parameters
WINDOW_SIZE = 30          # Trading days per candlestick chart
MAX_IMAGES_PER_STOCK = 500  # Increased for >1GB dataset requirement
IMAGE_DPI = 150           # Increased DPI for higher quality (larger file size)
IMAGE_SIZE = (8, 6)       # inches (width, height) - larger for better quality

# Custom style for candlestick charts
CHART_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#00c853",       # Green for up
        down="#ff1744",     # Red for down
        edge="inherit",
        wick="inherit",
        volume="in",
    ),
    figcolor="#1a1a2e",
    facecolor="#1a1a2e",
    gridcolor="#2d2d44",
    gridstyle="--",
    gridaxis="both",
    y_on_right=True,
)

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


def prepare_stock_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Filter and prepare data for a single stock.
    Sets Date as index with proper datetime format required by mplfinance.
    """
    stock_df = df[df["Symbol"] == symbol].copy()
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df = stock_df.sort_values("Date").reset_index(drop=True)

    # mplfinance requires DatetimeIndex
    stock_df = stock_df.set_index("Date")
    stock_df.index.name = "Date"

    # Ensure proper column types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")

    stock_df = stock_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    return stock_df


def determine_label(window_df: pd.DataFrame, full_df: pd.DataFrame) -> str | None:
    """
    Determine if the next trading day after the window is bullish or bearish.

    Returns
    -------
    str or None
        'bullish' if next day close > last day close,
        'bearish' if next day close <= last day close,
        None if no next day data available.
    """
    last_date = window_df.index[-1]
    last_close = window_df["Close"].iloc[-1]

    # Find the next trading day in the full dataset
    future_dates = full_df.index[full_df.index > last_date]
    if len(future_dates) == 0:
        return None

    next_close = full_df.loc[future_dates[0], "Close"]

    return "bullish" if next_close > last_close else "bearish"


def generate_chart_image(
    window_df: pd.DataFrame,
    output_path: Path,
    symbol: str,
) -> bool:
    """
    Generate and save a single candlestick chart image.

    Returns True on success, False on failure.
    """
    try:
        fig, axes = mpf.plot(
            window_df,
            type="candle",
            style=CHART_STYLE,
            volume=True,
            figsize=IMAGE_SIZE,
            returnfig=True,
            tight_layout=True,
        )

        # Remove axis labels and title for clean CNN input
        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=6)

        fig.savefig(
            output_path,
            dpi=IMAGE_DPI,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            pad_inches=0.1,
        )
        plt.close(fig)
        return True

    except Exception as exc:
        logger.warning("Failed to generate chart: %s", exc)
        plt.close("all")
        return False


def generate_images_for_stock(
    df: pd.DataFrame,
    symbol: str,
    max_images: int = MAX_IMAGES_PER_STOCK,
) -> dict:
    """
    Generate candlestick chart images for a single stock.

    Creates sliding windows of WINDOW_SIZE days, labels each based
    on next-day price movement, and saves to bullish/bearish directories.

    Returns
    -------
    dict
        Counts of generated images: {'bullish': int, 'bearish': int, 'skipped': int}
    """
    stock_df = prepare_stock_data(df, symbol)

    if len(stock_df) < WINDOW_SIZE + 1:
        logger.warning(
            "%s: Insufficient data (%d rows, need %d). Skipping.",
            symbol, len(stock_df), WINDOW_SIZE + 1,
        )
        return {"bullish": 0, "bearish": 0, "skipped": 0}

    # Calculate total possible windows
    total_windows = len(stock_df) - WINDOW_SIZE
    if total_windows <= 0:
        return {"bullish": 0, "bearish": 0, "skipped": 0}

    # Sample evenly if too many windows
    if total_windows > max_images:
        indices = np.linspace(0, total_windows - 1, max_images, dtype=int)
    else:
        indices = np.arange(total_windows)

    counts = {"bullish": 0, "bearish": 0, "skipped": 0}

    for i, start_idx in enumerate(indices):
        window = stock_df.iloc[start_idx : start_idx + WINDOW_SIZE]

        # Determine label
        label = determine_label(window, stock_df)
        if label is None:
            counts["skipped"] += 1
            continue

        # Build filename
        end_date = window.index[-1].strftime("%Y%m%d")
        filename = f"{symbol}_{end_date}.png"

        if label == "bullish":
            output_path = BULLISH_DIR / filename
        else:
            output_path = BEARISH_DIR / filename

        # Generate and save
        success = generate_chart_image(window, output_path, symbol)
        if success:
            counts[label] += 1
        else:
            counts["skipped"] += 1

        # Progress logging
        if (i + 1) % 50 == 0:
            logger.info(
                "  %s: %d/%d images generated...",
                symbol, i + 1, len(indices),
            )

    return counts


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=" * 60)
    logger.info("CANDLESTICK IMAGE GENERATION PIPELINE")
    logger.info("=" * 60)

    if not INPUT_FILE.exists():
        logger.error("Input file not found: %s", INPUT_FILE)
        logger.error("Run preprocessing.py first.")
        sys.exit(1)

    # Create output directories
    BULLISH_DIR.mkdir(parents=True, exist_ok=True)
    BEARISH_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directories created ✓")

    # Load data
    df = pd.read_csv(INPUT_FILE)
    symbols = sorted(df["Symbol"].unique())
    logger.info("Loaded %d rows for %d stocks", len(df), len(symbols))
    logger.info("Window size: %d days | Max images/stock: %d", WINDOW_SIZE, MAX_IMAGES_PER_STOCK)

    # Generate images per stock
    total_counts = {"bullish": 0, "bearish": 0, "skipped": 0}

    for symbol in symbols:
        logger.info("Generating charts for %s...", symbol)
        counts = generate_images_for_stock(df, symbol)
        for key in total_counts:
            total_counts[key] += counts[key]
        logger.info(
            "  %s complete: bullish=%d, bearish=%d, skipped=%d",
            symbol, counts["bullish"], counts["bearish"], counts["skipped"],
        )

    # Summary
    logger.info("-" * 60)
    logger.info("IMAGE GENERATION SUMMARY")
    logger.info("-" * 60)
    logger.info("Total bullish images: %d → %s", total_counts["bullish"], BULLISH_DIR)
    logger.info("Total bearish images: %d → %s", total_counts["bearish"], BEARISH_DIR)
    logger.info("Total skipped:        %d", total_counts["skipped"])
    logger.info("Grand total:          %d images", total_counts["bullish"] + total_counts["bearish"])
    
    # Calculate dataset size
    total_size_bytes = 0
    for img_dir in [BULLISH_DIR, BEARISH_DIR]:
        for img_file in img_dir.glob("*.png"):
            total_size_bytes += img_file.stat().st_size
    
    total_size_gb = total_size_bytes / (1024**3)
    total_size_mb = total_size_bytes / (1024**2)
    avg_size_kb = (total_size_bytes / (total_counts["bullish"] + total_counts["bearish"])) / 1024 if (total_counts["bullish"] + total_counts["bearish"]) > 0 else 0
    
    logger.info("-" * 60)
    logger.info("DATASET SIZE STATISTICS")
    logger.info("-" * 60)
    logger.info("Total dataset size:   %.2f GB (%.2f MB)", total_size_gb, total_size_mb)
    logger.info("Average image size:   %.2f KB", avg_size_kb)
    logger.info("Class balance ratio:  %.2f", max(total_counts["bullish"], total_counts["bearish"]) / max(1, min(total_counts["bullish"], total_counts["bearish"])))
    
    if total_size_gb >= 1.0:
        logger.info("✓ Dataset size requirement MET: >1GB")
    else:
        logger.warning("⚠ Dataset size: %.2f GB (target: >1GB)", total_size_gb)
        logger.info("  Recommendation: Increase MAX_IMAGES_PER_STOCK or IMAGE_DPI")
    
    logger.info("=" * 60)
    logger.info("UNSTRUCTURED DATA PROOF")
    logger.info("=" * 60)
    logger.info("✓ Candlestick images: %d total", total_counts["bullish"] + total_counts["bearish"])
    logger.info("✓ Dataset size: %.2f GB", total_size_gb)
    logger.info("✓ Class distribution: Bullish=%d (%.1f%%), Bearish=%d (%.1f%%)",
                total_counts["bullish"],
                100 * total_counts["bullish"] / max(1, total_counts["bullish"] + total_counts["bearish"]),
                total_counts["bearish"],
                100 * total_counts["bearish"] / max(1, total_counts["bullish"] + total_counts["bearish"]))
    logger.info("✓ Image format: PNG (lossless)")
    logger.info("✓ Resolution: %dx%d @ %d DPI", int(IMAGE_SIZE[0] * IMAGE_DPI), int(IMAGE_SIZE[1] * IMAGE_DPI), IMAGE_DPI)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
