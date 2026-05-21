"""
generate_image_labels.py
=========================
Generate image_labels.csv from existing candlestick images.

Creates a CSV file mapping each image to its label (bullish/bearish),
symbol, and date for CNN training and feature extraction.

Output:
    data/image_labels.csv with columns:
    - image_path: relative path to image
    - symbol: stock ticker symbol
    - date: trading date (YYYYMMDD format)
    - label: 'bullish' or 'bearish'

Usage:
    python scripts/generate_image_labels.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
BULLISH_DIR = IMAGES_DIR / "bullish"
BEARISH_DIR = IMAGES_DIR / "bearish"
OUTPUT_CSV = PROJECT_ROOT / "data" / "image_labels.csv"


def parse_image_filename(filename: str) -> tuple[str, str]:
    """
    Parse image filename to extract symbol and date.
    
    Expected format: SYMBOL_YYYYMMDD.png
    Example: AAPL_20230115.png -> ('AAPL', '20230115')
    
    Args:
        filename: Image filename
        
    Returns:
        Tuple of (symbol, date)
    """
    try:
        # Remove .png extension
        name = filename.replace('.png', '')
        
        # Split by underscore
        parts = name.split('_')
        
        if len(parts) >= 2:
            symbol = parts[0]
            date = parts[1]
            return symbol, date
        else:
            logger.warning(f"Invalid filename format: {filename}")
            return None, None
            
    except Exception as e:
        logger.warning(f"Failed to parse filename {filename}: {e}")
        return None, None


def generate_labels() -> pd.DataFrame:
    """
    Generate image labels DataFrame from existing images.
    
    Returns:
        DataFrame with columns: image_path, symbol, date, label
    """
    records = []
    
    # Process bullish images
    logger.info("Processing bullish images...")
    if BULLISH_DIR.exists():
        bullish_images = list(BULLISH_DIR.glob("*.png"))
        logger.info(f"Found {len(bullish_images)} bullish images")
        
        for img_path in bullish_images:
            symbol, date = parse_image_filename(img_path.name)
            
            if symbol and date:
                # Store relative path from project root
                rel_path = img_path.relative_to(PROJECT_ROOT)
                
                records.append({
                    'image_path': str(rel_path).replace('\\', '/'),  # Use forward slashes
                    'symbol': symbol,
                    'date': date,
                    'label': 'bullish'
                })
    else:
        logger.error(f"Bullish directory not found: {BULLISH_DIR}")
        sys.exit(1)
    
    # Process bearish images
    logger.info("Processing bearish images...")
    if BEARISH_DIR.exists():
        bearish_images = list(BEARISH_DIR.glob("*.png"))
        logger.info(f"Found {len(bearish_images)} bearish images")
        
        for img_path in bearish_images:
            symbol, date = parse_image_filename(img_path.name)
            
            if symbol and date:
                # Store relative path from project root
                rel_path = img_path.relative_to(PROJECT_ROOT)
                
                records.append({
                    'image_path': str(rel_path).replace('\\', '/'),  # Use forward slashes
                    'symbol': symbol,
                    'date': date,
                    'label': 'bearish'
                })
    else:
        logger.error(f"Bearish directory not found: {BEARISH_DIR}")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    return df


def main() -> None:
    logger.info("=" * 70)
    logger.info("IMAGE LABELS GENERATION")
    logger.info("=" * 70)
    
    # Check directories exist
    if not IMAGES_DIR.exists():
        logger.error(f"Images directory not found: {IMAGES_DIR}")
        logger.error("Run generate_candlestick_images.py first")
        sys.exit(1)
    
    # Generate labels
    df = generate_labels()
    
    if df.empty:
        logger.error("No images found. Cannot generate labels.")
        sys.exit(1)
    
    # Summary statistics
    logger.info("-" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("-" * 70)
    logger.info(f"Total images: {len(df)}")
    logger.info(f"Bullish images: {len(df[df['label'] == 'bullish'])}")
    logger.info(f"Bearish images: {len(df[df['label'] == 'bearish'])}")
    logger.info(f"Unique symbols: {df['symbol'].nunique()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Class balance
    bullish_count = len(df[df['label'] == 'bullish'])
    bearish_count = len(df[df['label'] == 'bearish'])
    balance_ratio = max(bullish_count, bearish_count) / max(1, min(bullish_count, bearish_count))
    logger.info(f"Class balance ratio: {balance_ratio:.2f}")
    
    if balance_ratio > 1.5:
        logger.warning("Classes are imbalanced (ratio > 1.5)")
    else:
        logger.info("Classes are well balanced")
    
    # Save to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    logger.info("-" * 70)
    logger.info(f"Image labels saved: {OUTPUT_CSV}")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info("=" * 70)
    
    # Show sample
    logger.info("\nSample rows:")
    print(df.head(10).to_string(index=False))
    
    logger.info("\nImage labels generation complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Failed to generate image labels: {e}")
        sys.exit(1)
