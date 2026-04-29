"""
predict_live.py
===============
Production-ready multi-stock live prediction engine with HDFS integration.

Flow:
    User selects stock → Live API fetch → Save CSV locally → 
    Upload to HDFS → Feature engineering → Schema validation → 
    Random Forest prediction → Confidence score → MongoDB save → 
    Dashboard display

Supports multiple stable stocks:
    AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, NFLX

Features:
    - Dynamic stock selection
    - Alpha Vantage + Yahoo Finance fallback
    - Automatic CSV dump and HDFS upload
    - Exact feature engineering matching training pipeline
    - Strict feature validation
    - Confidence scores with probabilities
    - MongoDB integration
    - Production-grade error handling
    - Professional logging
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import joblib
import pandas as pd

from fetch_live_data import fetch_live_stock_data
from mongo_store import save_prediction
from model_config import FINAL_MODEL_FILE, get_expected_features, validate_feature_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Supported stable stocks for live prediction
SUPPORTED_STOCKS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Google
    "AMZN",   # Amazon
    "NVDA",   # NVIDIA
    "TSLA",   # Tesla
    "META",   # Meta (Facebook)
    "NFLX",   # Netflix
]

# Directory paths for live data storage
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_API_DUMPS_DIR = PROJECT_ROOT / "live_api_dumps"
HDFS_LIVE_DUMPS_PATH = "/stock_data/live_api_dumps/"
HDFS_NAMENODE = "hdfs://localhost:9000"

# Global model cache to avoid reloading
_MODEL_CACHE = None


def load_model(model_file: Path = FINAL_MODEL_FILE, use_cache: bool = True):
    """
    Load the final production Random Forest model with caching.
    
    Args:
        model_file: Path to the final model pickle file
        use_cache: Whether to use cached model (improves performance)
        
    Returns:
        Trained Random Forest model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    global _MODEL_CACHE
    
    if use_cache and _MODEL_CACHE is not None:
        logger.debug("Using cached model")
        return _MODEL_CACHE
    
    try:
        if not model_file.exists():
            raise FileNotFoundError(
                f"Final production model not found: {model_file}\n"
                f"Please ensure the model is trained and saved at this location."
            )
        
        logger.info(f"Loading model from: {model_file}")
        model = joblib.load(model_file)
        
        if use_cache:
            _MODEL_CACHE = model
        
        logger.info("✓ Model loaded successfully")
        return model
        
    except Exception as e:
        logger.exception("Failed to load model")
        raise


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize stock symbol.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Normalized uppercase symbol
        
    Raises:
        ValueError: If symbol is not supported
    """
    symbol = symbol.strip().upper()
    
    if not symbol:
        raise ValueError("Stock symbol cannot be empty")
    
    if symbol not in SUPPORTED_STOCKS:
        raise ValueError(
            f"Stock '{symbol}' is not supported.\n"
            f"Supported stocks: {', '.join(SUPPORTED_STOCKS)}"
        )
    
    return symbol


def ensure_live_dumps_directory() -> Path:
    """
    Ensure the live_api_dumps directory exists.
    
    Returns:
        Path to live_api_dumps directory
    """
    LIVE_API_DUMPS_DIR.mkdir(parents=True, exist_ok=True)
    return LIVE_API_DUMPS_DIR


def save_live_data_to_csv(
    live_data: Dict,
    symbol: str,
    timestamp: Optional[datetime] = None
) -> Path:
    """
    Save raw live API data to CSV file with timestamp.
    
    Args:
        live_data: Live data dictionary from fetch_live_stock_data
        symbol: Stock ticker symbol
        timestamp: Optional timestamp (defaults to now)
        
    Returns:
        Path to saved CSV file
        
    Raises:
        Exception: If CSV save fails
    """
    try:
        # Ensure directory exists
        dumps_dir = ensure_live_dumps_directory()
        
        # Generate timestamp-based filename
        if timestamp is None:
            timestamp = datetime.now()
        
        date_str = timestamp.strftime("%Y_%m_%d_%H%M%S")
        filename = f"{symbol}_live_{date_str}.csv"
        csv_path = dumps_dir / filename
        
        logger.info(f"Saving live data to CSV: {filename}")
        
        # Prepare data for CSV
        # Include quote data and basic info
        csv_data = {
            "Symbol": [symbol],
            "Date": [live_data["latest_date"]],
            "Open": [live_data["quote"]["open"]],
            "High": [live_data["quote"]["high"]],
            "Low": [live_data["quote"]["low"]],
            "Close": [live_data["quote"]["close"]],
            "Volume": [live_data["quote"]["volume"]],
            "Source": [live_data["source"]],
            "Timestamp": [timestamp.isoformat()],
        }
        
        # Add all features
        for feature_name, feature_value in live_data["features"].items():
            csv_data[feature_name] = [feature_value]
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        
        file_size = csv_path.stat().st_size
        logger.info(f"✓ CSV saved successfully: {csv_path.name} ({file_size} bytes)")
        
        return csv_path
        
    except Exception as e:
        logger.exception(f"Failed to save live data to CSV for {symbol}")
        raise


def check_hdfs_available() -> bool:
    """
    Check if HDFS is available and accessible.
    
    Returns:
        True if HDFS is available, False otherwise
    """
    try:
        # Try to run hdfs command
        result = subprocess.run(
            ["hdfs", "dfs", "-test", "-d", "/"],
            capture_output=True,
            timeout=10,
            shell=True  # Use shell on Windows to find hdfs in PATH
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.warning(f"HDFS availability check failed: {e}")
        return False


def upload_to_hdfs(
    local_file: Path,
    hdfs_directory: str = HDFS_LIVE_DUMPS_PATH,
    overwrite: bool = False
) -> Tuple[bool, str]:
    """
    Upload a local file to HDFS with production-safe error handling.
    
    Args:
        local_file: Path to local file to upload
        hdfs_directory: HDFS directory path (without hdfs:// prefix)
        overwrite: Whether to overwrite if file exists
        
    Returns:
        Tuple of (success: bool, hdfs_path: str)
        
    Raises:
        FileNotFoundError: If local file doesn't exist
    """
    if not local_file.exists():
        raise FileNotFoundError(f"Local file not found: {local_file}")
    
    try:
        # Construct full HDFS path
        hdfs_full_path = f"{HDFS_NAMENODE}{hdfs_directory}{local_file.name}"
        
        logger.info(f"Uploading to HDFS: {hdfs_directory}{local_file.name}")
        
        # Check if HDFS is available
        if not check_hdfs_available():
            logger.error("HDFS is not available or not running")
            return False, ""
        
        # Create HDFS directory if it doesn't exist
        mkdir_cmd = ["hdfs", "dfs", "-mkdir", "-p", hdfs_directory]
        subprocess.run(mkdir_cmd, capture_output=True, timeout=30, check=False, shell=True)
        
        # Check if file already exists in HDFS
        hdfs_file_path = f"{hdfs_directory}{local_file.name}"
        test_cmd = ["hdfs", "dfs", "-test", "-e", hdfs_file_path]
        test_result = subprocess.run(test_cmd, capture_output=True, timeout=10, shell=True)
        
        file_exists = (test_result.returncode == 0)
        
        if file_exists and not overwrite:
            logger.warning(f"File already exists in HDFS: {hdfs_file_path}")
            logger.info("Using existing file (overwrite=False)")
            return True, hdfs_full_path
        
        # Upload file to HDFS
        put_cmd = ["hdfs", "dfs", "-put"]
        if overwrite:
            put_cmd.append("-f")  # Force overwrite
        put_cmd.extend([str(local_file), hdfs_directory])
        
        result = subprocess.run(
            put_cmd,
            capture_output=True,
            text=True,
            timeout=60,
            shell=True  # Use shell on Windows
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Successfully uploaded to HDFS: {hdfs_file_path}")
            
            # Verify upload
            verify_cmd = ["hdfs", "dfs", "-ls", hdfs_file_path]
            verify_result = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                timeout=10,
                shell=True  # Use shell on Windows
            )
            
            if verify_result.returncode == 0:
                logger.info("✓ Upload verified successfully")
            else:
                logger.warning("Upload verification failed, but file may exist")
            
            return True, hdfs_full_path
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error(f"HDFS upload failed: {error_msg}")
            return False, ""
            
    except subprocess.TimeoutExpired:
        logger.error("HDFS upload timed out")
        return False, ""
    except Exception as e:
        logger.exception(f"Unexpected error during HDFS upload: {e}")
        return False, ""


def save_and_upload_live_data(
    live_data: Dict,
    symbol: str,
    skip_hdfs: bool = False
) -> Dict[str, str]:
    """
    Save live data to CSV and upload to HDFS.
    
    Args:
        live_data: Live data dictionary from fetch_live_stock_data
        symbol: Stock ticker symbol
        skip_hdfs: Skip HDFS upload (for testing)
        
    Returns:
        Dictionary with local_path and hdfs_path
        
    Raises:
        Exception: If CSV save fails (HDFS failure is non-critical)
    """
    result = {
        "local_path": "",
        "hdfs_path": "",
        "hdfs_success": False,
    }
    
    try:
        # Step 1: Save to local CSV
        timestamp = datetime.now()
        csv_path = save_live_data_to_csv(live_data, symbol, timestamp)
        result["local_path"] = str(csv_path)
        
        # Step 2: Upload to HDFS (non-critical)
        if not skip_hdfs:
            try:
                logger.info("Uploading to HDFS...")
                success, hdfs_path = upload_to_hdfs(csv_path, HDFS_LIVE_DUMPS_PATH)
                
                if success:
                    result["hdfs_path"] = hdfs_path
                    result["hdfs_success"] = True
                else:
                    logger.warning("HDFS upload failed, but continuing with prediction")
                    logger.warning("Live data is saved locally and can be uploaded manually")
                    
            except Exception as e:
                logger.warning(f"HDFS upload error (non-critical): {e}")
                logger.info("Continuing with prediction using local data")
        else:
            logger.info("Skipping HDFS upload (skip_hdfs=True)")
        
        return result
        
    except Exception as e:
        logger.exception("Failed to save live data")
        raise


def predict_live(
    symbol: str, 
    source: str = "auto", 
    api_key: str = "",
    save_to_db: bool = True,
    skip_hdfs: bool = False,
) -> Dict:
    """
    Perform live prediction for a given stock symbol with HDFS integration.
    
    This is the main production prediction function that:
    1. Validates the stock symbol
    2. Fetches live data from API
    3. Saves raw data to CSV locally
    4. Uploads CSV to HDFS
    5. Generates features matching training pipeline
    6. Validates feature schema
    7. Makes prediction with confidence score
    8. Saves prediction to MongoDB
    9. Returns comprehensive result
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        source: Data source - 'alpha_vantage', 'yfinance', or 'auto'
        api_key: Alpha Vantage API key (optional if set in environment)
        save_to_db: Whether to save prediction to MongoDB
        skip_hdfs: Skip HDFS upload (for testing without HDFS)
        
    Returns:
        Dictionary containing:
            - symbol: Stock ticker
            - prediction: 'UP' or 'DOWN'
            - confidence: Confidence percentage (0-100)
            - probabilities: Dict with DOWN and UP probabilities
            - latest_date: Date of latest data used
            - source: Data source used
            - features: Feature values used for prediction
            - mongo_id: MongoDB document ID (if saved)
            - model_file: Path to model used
            - timestamp: Prediction timestamp
            - local_csv_path: Path to local CSV dump
            - hdfs_path: HDFS path (if uploaded)
            
    Raises:
        ValueError: If symbol is invalid or features mismatch
        Exception: If API fetch, model loading, or prediction fails
    """
    try:
        # Step 1: Validate symbol
        logger.info("=" * 70)
        logger.info(f"Starting live prediction for: {symbol}")
        logger.info("=" * 70)
        
        symbol = validate_symbol(symbol)
        logger.info(f"✓ Symbol validated: {symbol}")
        
        # Step 2: Load model
        model = load_model()
        
        # Step 3: Fetch live data with features
        logger.info(f"Fetching live data for {symbol}...")
        live = fetch_live_stock_data(symbol=symbol, api_key=api_key, source=source)
        logger.info(f"✓ Live data fetched from {live['source']} (date: {live['latest_date']})")
        
        # Step 4: Save to CSV and upload to HDFS
        logger.info("Saving live data to CSV and uploading to HDFS...")
        storage_result = save_and_upload_live_data(live, symbol, skip_hdfs=skip_hdfs)
        logger.info(f"✓ Live data saved locally: {Path(storage_result['local_path']).name}")
        if storage_result['hdfs_success']:
            logger.info(f"✓ Live data uploaded to HDFS: {HDFS_LIVE_DUMPS_PATH}")
        
        # Step 5: Get expected features and validate schema
        logger.info("Validating feature schema...")
        expected_features = get_expected_features()
        
        # Ensure all expected features are present in live data
        missing_features = set(expected_features) - set(live["features"].keys())
        if missing_features:
            raise ValueError(
                f"Missing features in live data: {sorted(missing_features)}\n"
                f"Expected: {expected_features}"
            )
        
        validate_feature_schema(expected_features)
        logger.info(f"✓ Feature schema validated ({len(expected_features)} features)")
        
        # Step 5: Prepare feature vector in exact order
        logger.info("Generating features...")
        row = {c: live["features"][c] for c in expected_features}
        X = pd.DataFrame([row], columns=expected_features)
        logger.info("✓ Feature vector prepared")
        
        # Step 6: Make prediction
        logger.info("Predicting...")
        pred = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        confidence = float(probs.max() * 100)
        label = "UP 📈" if pred == 1 else "DOWN 📉"
        
        logger.info(f"✓ Prediction: {label} | Confidence: {confidence:.2f}%")
        
        # Step 7: Save to MongoDB
        mongo_id = ""
        if save_to_db:
            try:
                logger.info("Saving to MongoDB...")
                mongo_id = save_prediction(
                    symbol=symbol,
                    prediction=label,
                    confidence=confidence,
                    source=live["source"],
                    model="Random Forest (Final)",
                    meta={
                        "latest_date": live["latest_date"],
                        "down_probability": float(probs[0]),
                        "up_probability": float(probs[1]),
                        "quote": live["quote"],
                        "feature_count": len(expected_features),
                    },
                )
                if mongo_id:
                    logger.info(f"✓ Saved to MongoDB (ID: {mongo_id})")
                else:
                    logger.warning("MongoDB save returned empty ID")
            except Exception as e:
                logger.warning(f"MongoDB save failed (non-critical): {e}")
        
        # Step 8: Build comprehensive result
        result = {
            "symbol": symbol,
            "prediction": label,
            "confidence": round(confidence, 2),
            "probabilities": {
                "DOWN": round(float(probs[0]) * 100, 2),
                "UP": round(float(probs[1]) * 100, 2),
            },
            "latest_date": live["latest_date"],
            "source": live["source"],
            "quote": live["quote"],
            "features": {k: round(v, 4) for k, v in row.items()},
            "mongo_id": mongo_id,
            "model_file": str(FINAL_MODEL_FILE),
            "timestamp": datetime.now().isoformat(),
            "feature_count": len(expected_features),
            "local_csv_path": storage_result["local_path"],
            "hdfs_path": storage_result["hdfs_path"],
            "hdfs_uploaded": storage_result["hdfs_success"],
        }
        
        logger.info("=" * 70)
        logger.info("✓ Prediction completed successfully")
        logger.info("=" * 70)
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.exception(f"Prediction failed for {symbol}")
        raise


def predict_multiple(
    symbols: List[str],
    source: str = "auto",
    api_key: str = "",
    save_to_db: bool = True,
    skip_hdfs: bool = False,
) -> Dict[str, Dict]:
    """
    Perform live predictions for multiple stocks with HDFS integration.
    
    Args:
        symbols: List of stock ticker symbols
        source: Data source - 'alpha_vantage', 'yfinance', or 'auto'
        api_key: Alpha Vantage API key
        save_to_db: Whether to save predictions to MongoDB
        skip_hdfs: Skip HDFS upload (for testing)
        
    Returns:
        Dictionary mapping symbol to prediction result
        Failed predictions are included with error information
    """
    results = {}
    
    logger.info(f"Starting batch prediction for {len(symbols)} stocks")
    logger.info(f"Symbols: {', '.join(symbols)}")
    
    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
            result = predict_live(
                symbol=symbol,
                source=source,
                api_key=api_key,
                save_to_db=save_to_db,
                skip_hdfs=skip_hdfs,
            )
            results[symbol] = result
            
        except Exception as e:
            logger.error(f"Failed to predict {symbol}: {e}")
            results[symbol] = {
                "symbol": symbol,
                "error": str(e),
                "success": False,
            }
    
    success_count = sum(1 for r in results.values() if "error" not in r)
    logger.info(f"\nBatch prediction complete: {success_count}/{len(symbols)} successful")
    
    return results


def get_supported_stocks() -> List[str]:
    """
    Get list of supported stock symbols.
    
    Returns:
        List of supported stock ticker symbols
    """
    return SUPPORTED_STOCKS.copy()


def main() -> None:
    """
    CLI entry point for live stock prediction.
    
    Examples:
        # Single stock prediction
        python predict_live.py --symbol AAPL
        
        # Multiple stocks
        python predict_live.py --symbols AAPL MSFT GOOGL
        
        # With specific data source
        python predict_live.py --symbol NVDA --source yfinance
        
        # List supported stocks
        python predict_live.py --list-stocks
    """
    parser = argparse.ArgumentParser(
        description="Production-ready multi-stock live prediction engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported Stocks:
  {', '.join(SUPPORTED_STOCKS)}

Examples:
  # Predict single stock
  python predict_live.py --symbol AAPL
  
  # Predict multiple stocks
  python predict_live.py --symbols AAPL MSFT GOOGL NVDA
  
  # Use specific data source
  python predict_live.py --symbol TSLA --source yfinance
  
  # Skip MongoDB save (testing)
  python predict_live.py --symbol META --no-save
  
  # List supported stocks
  python predict_live.py --list-stocks
        """
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        help="Single stock symbol to predict (e.g., AAPL)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Multiple stock symbols to predict (e.g., AAPL MSFT GOOGL)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["alpha_vantage", "yfinance", "auto"],
        help="Data source (default: auto - tries Alpha Vantage first, falls back to yfinance)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
        help="Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY env variable)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving prediction to MongoDB (for testing)"
    )
    parser.add_argument(
        "--skip-hdfs",
        action="store_true",
        help="Skip HDFS upload (for testing without HDFS)"
    )
    parser.add_argument(
        "--list-stocks",
        action="store_true",
        help="List all supported stock symbols and exit"
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["json", "summary"],
        default="json",
        help="Output format (default: json)"
    )
    
    args = parser.parse_args()
    
    # Handle --list-stocks
    if args.list_stocks:
        print("\nSupported Stocks for Live Prediction:")
        print("=" * 50)
        for symbol in SUPPORTED_STOCKS:
            print(f"  • {symbol}")
        print("=" * 50)
        print(f"Total: {len(SUPPORTED_STOCKS)} stocks")
        sys.exit(0)
    
    # Validate input
    if not args.symbol and not args.symbols:
        parser.error("Either --symbol or --symbols must be provided")
    
    if args.symbol and args.symbols:
        parser.error("Cannot use both --symbol and --symbols together")
    
    exit_code = 0
    
    try:
        # Single stock prediction
        if args.symbol:
            result = predict_live(
                symbol=args.symbol,
                source=args.source,
                api_key=args.api_key,
                save_to_db=not args.no_save,
                skip_hdfs=args.skip_hdfs,
            )
            
            if args.output == "json":
                print("\n" + json.dumps(result, indent=2))
            else:
                print("\n" + "=" * 70)
                print(f"  PREDICTION RESULT — {result['symbol']}")
                print("=" * 70)
                print(f"  Prediction:  {result['prediction']}")
                print(f"  Confidence:  {result['confidence']:.2f}%")
                print(f"  Date:        {result['latest_date']}")
                print(f"  Source:      {result['source']}")
                print(f"  Close Price: ${result['quote']['close']:.2f}")
                print("=" * 70)
        
        # Multiple stocks prediction
        else:
            results = predict_multiple(
                symbols=args.symbols,
                source=args.source,
                api_key=args.api_key,
                save_to_db=not args.no_save,
                skip_hdfs=args.skip_hdfs,
            )
            
            if args.output == "json":
                print("\n" + json.dumps(results, indent=2))
            else:
                print("\n" + "=" * 70)
                print("  BATCH PREDICTION RESULTS")
                print("=" * 70)
                for symbol, result in results.items():
                    if "error" in result:
                        print(f"  {symbol}: ❌ FAILED - {result['error']}")
                    else:
                        print(f"  {symbol}: {result['prediction']} ({result['confidence']:.1f}%)")
                print("=" * 70)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        exit_code = 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        exit_code = 2
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        exit_code = 3
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

