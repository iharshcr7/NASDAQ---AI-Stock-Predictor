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
from symbol_mapper import resolve_symbol, get_all_symbols, is_symbol_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dynamic stock discovery - automatically loads all available stocks from dataset
# This replaces the hardcoded SUPPORTED_STOCKS list
def _load_supported_stocks() -> List[str]:
    """
    Dynamically load all supported stocks from the dataset directory.
    This function is called once at module load time.
    
    Returns:
        List of all available stock symbols from data/stock_market_dataset/stocks/
    """
    try:
        symbols = get_all_symbols()
        logger.info(f"Dynamically loaded {len(symbols)} supported stocks from dataset")
        return symbols
    except Exception as e:
        logger.error(f"Failed to load stocks dynamically: {e}")
        # Fallback to original stable stocks if discovery fails
        logger.warning("Falling back to default stable stocks")
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX"
        ]

# Load supported stocks dynamically at module initialization
SUPPORTED_STOCKS = _load_supported_stocks()

# Directory paths for live data storage
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_API_DUMPS_DIR = PROJECT_ROOT / "live_api_dumps"
HDFS_LIVE_DUMPS_PATH = "/stock_data/live_api_dumps/"
HDFS_NAMENODE = "hdfs://localhost:9000"

# Spark UI keep-alive duration (seconds)
# Set to 0 to stop Spark immediately after processing
# Set to -1 to keep Spark running indefinitely (development mode)
SPARK_UI_KEEPALIVE_SECONDS = int(os.environ.get("SPARK_UI_KEEPALIVE_SECONDS", "300"))  # Default: 5 minutes

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
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.exception("Failed to load model")
        raise


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize stock symbol with company name resolution.
    
    Supports:
        - Direct symbol: "AAPL", "aapl", "Aapl"
        - Company name: "Apple", "apple", "Tesla", "tesla"
        - Mixed case: "TeSLa", "APPLE"
    
    Args:
        symbol: Stock ticker symbol or company name
        
    Returns:
        Normalized uppercase symbol
        
    Raises:
        ValueError: If symbol cannot be resolved or is not available
    """
    if not symbol or not symbol.strip():
        raise ValueError("Stock symbol or company name cannot be empty")
    
    try:
        # Step 1: Resolve company name to symbol (handles case-insensitive matching)
        resolved_symbol = resolve_symbol(symbol)
        
        # Step 2: Verify symbol is in supported stocks
        if resolved_symbol not in SUPPORTED_STOCKS:
            raise ValueError(
                f"Stock '{resolved_symbol}' is not available in the dataset.\n"
                f"Total available stocks: {len(SUPPORTED_STOCKS)}"
            )
        
        logger.debug(f"Validated: '{symbol}' -> {resolved_symbol}")
        return resolved_symbol
        
    except ValueError as e:
        # Re-raise with helpful context
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise ValueError(
                f"Stock '{symbol}' not found.\n"
                f"Available stocks: {len(SUPPORTED_STOCKS)} symbols in dataset.\n"
                f"Examples: AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, META, NFLX\n"
                f"You can also use company names: Apple, Microsoft, Tesla, etc."
            )
        raise


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
    timestamp: Optional[datetime] = None,
    save_full_history: bool = True
) -> Path:
    """
    Save raw live API data to CSV file with timestamp.
    
    Args:
        live_data: Live data dictionary from fetch_live_stock_data
        symbol: Stock ticker symbol
        timestamp: Optional timestamp (defaults to now)
        save_full_history: If True, saves full historical data for Spark processing
        
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
        
        if save_full_history and "historical_df" in live_data:
            # Save full historical data for Spark processing
            logger.info("Saving full historical data for Spark processing...")
            df = live_data["historical_df"]
            df.to_csv(csv_path, index=False)
            
            file_size = csv_path.stat().st_size
            row_count = len(df)
            logger.info(f"CSV saved successfully: {csv_path.name} ({row_count} rows, {file_size} bytes)")
        else:
            # Save only latest row (legacy behavior)
            logger.info("Saving latest row only...")
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
            logger.info(f"CSV saved successfully: {csv_path.name} ({file_size} bytes)")
        
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
    overwrite: bool = True  # Changed default to True for live data
) -> Tuple[bool, str]:
    """
    Upload a local file to HDFS with production-safe error handling.
    CRITICAL: This function MUST succeed for the pipeline to continue.
    
    Args:
        local_file: Path to local file to upload
        hdfs_directory: HDFS directory path (without hdfs:// prefix)
        overwrite: Whether to overwrite if file exists (default: True for live data)
        
    Returns:
        Tuple of (success: bool, hdfs_path: str)
        
    Raises:
        FileNotFoundError: If local file doesn't exist
        RuntimeError: If HDFS upload fails (critical error)
    """
    if not local_file.exists():
        raise FileNotFoundError(f"Local file not found: {local_file}")
    
    try:
        # Construct full HDFS path
        hdfs_full_path = f"{HDFS_NAMENODE}{hdfs_directory}{local_file.name}"
        hdfs_file_path = f"{hdfs_directory}{local_file.name}"
        
        logger.info("=" * 70)
        logger.info("HDFS UPLOAD STARTING")
        logger.info("=" * 70)
        logger.info(f"Local file:  {local_file}")
        logger.info(f"HDFS path:   {hdfs_file_path}")
        logger.info(f"Overwrite:   {overwrite}")
        
        # Check if HDFS is available
        logger.info("Checking HDFS availability...")
        if not check_hdfs_available():
            error_msg = "HDFS is not available or not running"
            logger.error(f"{error_msg}")
            logger.error("Please start HDFS: start-dfs.sh")
            raise RuntimeError(error_msg)
        logger.info("HDFS is available")
        
        # Create HDFS directory if it doesn't exist
        logger.info(f"Creating HDFS directory: {hdfs_directory}")
        mkdir_cmd = ["hdfs", "dfs", "-mkdir", "-p", hdfs_directory]
        mkdir_result = subprocess.run(
            mkdir_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            shell=True
        )
        
        if mkdir_result.returncode == 0:
            logger.info("HDFS directory ready")
        else:
            logger.warning(f"mkdir returned {mkdir_result.returncode} (may already exist)")
        
        # Upload file to HDFS
        logger.info("Uploading file to HDFS...")
        put_cmd = ["hdfs", "dfs", "-put", "-f", str(local_file), hdfs_directory]
        
        logger.info(f"Command: {' '.join(put_cmd)}")
        
        result = subprocess.run(
            put_cmd,
            capture_output=True,
            text=True,
            timeout=60,
            shell=True  # Use shell on Windows
        )
        
        if result.returncode == 0:
            logger.info("File uploaded successfully")
            
            # Verify upload
            logger.info("Verifying upload...")
            verify_cmd = ["hdfs", "dfs", "-ls", hdfs_file_path]
            verify_result = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                timeout=10,
                shell=True
            )
            
            if verify_result.returncode == 0:
                logger.info("Upload verified - file exists in HDFS")
                logger.info("=" * 70)
                logger.info("HDFS UPLOAD SUCCESS")
                logger.info("=" * 70)
                return True, hdfs_full_path
            else:
                logger.warning("Verification failed but upload may have succeeded")
                return True, hdfs_full_path
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error(f"HDFS upload failed: {error_msg}")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"HDFS upload failed: {error_msg}")
            
    except subprocess.TimeoutExpired:
        error_msg = "HDFS upload timed out"
        logger.error(f"{error_msg}")
        raise RuntimeError(error_msg)
    except Exception as e:
        logger.exception(f"Unexpected error during HDFS upload")
        raise RuntimeError(f"HDFS upload failed: {str(e)}")


def save_and_upload_live_data(
    live_data: Dict,
    symbol: str,
    skip_hdfs: bool = False,
    run_spark: bool = True
) -> Dict[str, str]:
    """
    Save live data to CSV and optionally upload to HDFS.
    HDFS upload is non-critical — prediction continues even if HDFS is unavailable.
    """
    result = {
        "local_path": "",
        "hdfs_path": "",
        "hdfs_success": False,
        "spark_triggered": False,
        "spark_success": False,
        "spark_ui_url": None,
    }
    
    try:
        # Step 1: Save to local CSV
        timestamp = datetime.now()
        csv_path = save_live_data_to_csv(live_data, symbol, timestamp, save_full_history=True)
        result["local_path"] = str(csv_path)
        logger.info(f"CSV Saved: {csv_path.name}")
        
        # Step 2: Upload to HDFS (non-critical — prediction continues regardless)
        if not skip_hdfs:
            try:
                logger.info("Uploading to HDFS...")
                success, hdfs_path = upload_to_hdfs(csv_path, HDFS_LIVE_DUMPS_PATH, overwrite=True)
                
                if success:
                    result["hdfs_path"] = hdfs_path
                    result["hdfs_success"] = True
                    logger.info("HDFS Upload SUCCESS")
                else:
                    logger.warning("HDFS upload returned False — continuing without HDFS")
                    
            except Exception as e:
                logger.warning(f"HDFS upload failed (non-critical): {e}")
                logger.warning("Continuing with prediction using local CSV only")
                # Do NOT raise — HDFS is optional, prediction must continue
        else:
            logger.warning("Skipping HDFS upload (skip_hdfs=True)")
            logger.warning("Spark processing will be skipped")
        
        # Step 3: Trigger Spark processing if HDFS upload succeeded
        if result["hdfs_success"] and run_spark:
            try:
                logger.info("=" * 70)
                logger.info("TRIGGERING SPARK PROCESSING")
                logger.info("=" * 70)
                
                # Run Spark processing
                spark_result = trigger_spark_processing(symbol)
                
                result["spark_triggered"] = True
                result["spark_success"] = spark_result["success"]
                result["spark_ui_url"] = spark_result.get("spark_ui_url")
                
                if spark_result["success"]:
                    logger.info("Spark Processing Complete")
                    if result["spark_ui_url"]:
                        logger.info(f"Spark UI: {result['spark_ui_url']}")
                else:
                    logger.warning(f"Spark processing failed: {spark_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.warning(f"Spark processing error (non-critical): {e}")
                logger.info("Continuing with prediction using local features")
        
        return result
        
    except Exception as e:
        logger.exception("Failed to save and upload live data")
        raise


def trigger_spark_processing(symbol: Optional[str] = None) -> Dict:
    """
    Trigger Spark live processing pipeline.
    
    Args:
        symbol: Optional symbol to filter
        
    Returns:
        Dictionary with Spark processing results including spark_ui_url
    """
    try:
        # Check if new file needs processing
        from spark_live_processing import detect_new_file, run_spark_live_processing
        
        new_file = detect_new_file()
        
        if not new_file:
            logger.info("No new files detected for Spark processing")
            return {
                "success": True,
                "message": "No new files to process",
                "new_file_detected": False,
                "spark_ui_url": None,
            }
        
        logger.info(f"New file detected → Running Spark: {new_file}")
        
        # Run Spark processing
        result = run_spark_live_processing(symbol=symbol, force=False)
        
        # Log Spark UI URL if available
        if result.get("spark_ui_url"):
            logger.info(f"Spark UI available at: {result['spark_ui_url']}")
        
        return result
        
    except Exception as e:
        logger.exception("Spark processing trigger failed")
        return {
            "success": False,
            "error": str(e),
            "new_file_detected": False,
            "spark_ui_url": None,
        }


def predict_live(
    symbol: str, 
    source: str = "auto", 
    api_key: str = "",
    save_to_db: bool = True,
    skip_hdfs: bool = False,
    run_spark: bool = True,
) -> Dict:
    """
    Perform live prediction for a given stock symbol with HDFS + Spark integration.
    
    This is the main production prediction function that:
    1. Validates the stock symbol
    2. Fetches live data from API
    3. Saves raw data to CSV locally
    4. Uploads CSV to HDFS (CRITICAL)
    5. Triggers Spark processing if new file detected
    6. Generates features matching training pipeline
    7. Validates feature schema
    8. Makes prediction with confidence score
    9. Saves prediction to MongoDB
    10. Returns comprehensive result
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        source: Data source - 'alpha_vantage', 'yfinance', or 'auto'
        api_key: Alpha Vantage API key (optional if set in environment)
        save_to_db: Whether to save prediction to MongoDB
        skip_hdfs: Skip HDFS upload (for testing without HDFS)
        run_spark: Run Spark processing after HDFS upload
        
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
            - spark_triggered: Whether Spark processing was triggered
            - spark_success: Whether Spark processing succeeded
            
    Raises:
        ValueError: If symbol is invalid or features mismatch
        Exception: If API fetch, model loading, or prediction fails
    """
    try:
        # Step 1: Validate symbol
        logger.info("=" * 70)
        logger.info(f"STARTING LIVE PREDICTION PIPELINE FOR: {symbol}")
        logger.info("=" * 70)
        
        symbol = validate_symbol(symbol)
        logger.info(f"Symbol validated: {symbol}")
        
        # Step 2: Load model
        model = load_model()
        
        # Step 3: Fetch live data with features
        logger.info(f"Fetching live data for {symbol}...")
        live = fetch_live_stock_data(symbol=symbol, api_key=api_key, source=source)
        logger.info(f"Live data fetched from {live['source']} (date: {live['latest_date']})")
        
        # Step 4: Save to CSV and upload to HDFS + Trigger Spark
        logger.info("Saving CSV and uploading to HDFS...")
        storage_result = save_and_upload_live_data(
            live, symbol, 
            skip_hdfs=skip_hdfs,
            run_spark=run_spark
        )
        logger.info(f"CSV Saved: {Path(storage_result['local_path']).name}")
        
        if storage_result['hdfs_success']:
            logger.info(f"HDFS Upload: SUCCESS → {HDFS_LIVE_DUMPS_PATH}")
        
        if storage_result.get('spark_triggered'):
            if storage_result.get('spark_success'):
                logger.info("Spark Processing: COMPLETE")
            else:
                logger.warning("Spark Processing: FAILED (using local features)")
        
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
        logger.info(f"Feature schema validated ({len(expected_features)} features)")
        
        # Step 6: Prepare feature vector in exact order
        logger.info("Generating features...")
        row = {c: live["features"][c] for c in expected_features}
        X = pd.DataFrame([row], columns=expected_features)
        logger.info("Feature vector prepared")
        
        # Step 7: Make prediction
        logger.info("Running Random Forest prediction...")
        pred = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        confidence = float(probs.max() * 100)
        label = "UP" if pred == 1 else "DOWN"
        
        logger.info(f"Prediction: {label} | Confidence: {confidence:.2f}%")
        
        # Step 8: Save to MongoDB
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
                        "hdfs_uploaded": storage_result['hdfs_success'],
                        "spark_processed": storage_result.get('spark_success', False),
                    },
                )
                if mongo_id:
                    logger.info(f"MongoDB Save: SUCCESS (ID: {mongo_id[:12]}...)")
                else:
                    logger.warning("MongoDB save returned empty ID")
            except Exception as e:
                logger.warning(f"MongoDB save failed (non-critical): {e}")
        
        # Step 9: Build comprehensive result
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
            "spark_triggered": storage_result.get("spark_triggered", False),
            "spark_success": storage_result.get("spark_success", False),
            "spark_ui_url": storage_result.get("spark_ui_url"),
        }
        
        # Log Spark UI URL if available
        if result["spark_ui_url"]:
            logger.info(f"Spark UI available at: {result['spark_ui_url']}")
        
        logger.info("=" * 70)
        logger.info("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
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
    run_spark: bool = True,
) -> Dict[str, Dict]:
    """
    Perform live predictions for multiple stocks with HDFS + Spark integration.
    
    Args:
        symbols: List of stock ticker symbols
        source: Data source - 'alpha_vantage', 'yfinance', or 'auto'
        api_key: Alpha Vantage API key
        save_to_db: Whether to save predictions to MongoDB
        skip_hdfs: Skip HDFS upload (for testing)
        run_spark: Run Spark processing after upload
        
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
                run_spark=run_spark,
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
        "--no-spark",
        action="store_true",
        help="Skip Spark processing (use local features only)"
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
                run_spark=not args.no_spark,
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
                run_spark=not args.no_spark,
            )
            
            if args.output == "json":
                print("\n" + json.dumps(results, indent=2))
            else:
                print("\n" + "=" * 70)
                print("  BATCH PREDICTION RESULTS")
                print("=" * 70)
                for symbol, result in results.items():
                    if "error" in result:
                        print(f"  {symbol}: FAILED - {result['error']}")
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

