"""
predict_lstm.py
===============
LSTM live inference engine for stock movement prediction.

Flow:
    Fetch 60+ days of historical data
    → Compute same 21 features as training pipeline
    → Scale with saved MinMaxScaler
    → Build sequence of last SEQUENCE_LENGTH days
    → Run LSTM model
    → Return UP/DOWN prediction + confidence

NOTE: LSTM is LIVE MODE ONLY.
      It requires a sequence of past days — a single manual snapshot
      is not sufficient input for an LSTM model.

Usage (CLI):
    python scripts/predict_lstm.py --symbol AAPL
    python scripts/predict_lstm.py --symbol TSLA --source yfinance
"""

from __future__ import annotations

import logging
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

LSTM_MODEL_FILE = MODELS_DIR / "lstm_model.keras"
LSTM_SCALER_FILE = MODELS_DIR / "lstm_scaler.pkl"
LSTM_METADATA_FILE = MODELS_DIR / "lstm_metadata.json"

SEQUENCE_LENGTH = 60  # Must match training

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA10", "MA20", "EMA12",
    "RSI", "MACD", "MACD_Signal",
    "Daily_Returns", "Volatility", "Price_Change_Pct",
    "Weekly_Momentum", "Trend_Strength", "BB_Width",
    "Avg_5D_Volume_Trend", "Lag_1", "Lag_3",
]

# Global cache — avoids reloading model on every prediction
_LSTM_MODEL_CACHE = None
_LSTM_SCALER_CACHE = None


# ---------------------------------------------------------------------------
# Model & Scaler Loading
# ---------------------------------------------------------------------------

def is_lstm_available() -> bool:
    """Check if LSTM model and scaler files exist."""
    return LSTM_MODEL_FILE.exists() and LSTM_SCALER_FILE.exists()


def load_lstm_model(use_cache: bool = True):
    """Load LSTM model with optional caching."""
    global _LSTM_MODEL_CACHE

    if use_cache and _LSTM_MODEL_CACHE is not None:
        return _LSTM_MODEL_CACHE

    if not LSTM_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"LSTM model not found: {LSTM_MODEL_FILE}\n"
            f"Run: python scripts/train_lstm_model.py"
        )

    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError(
            "TensorFlow not installed.\n"
            "Run: pip install tensorflow==2.15.0"
        )

    logger.info("Loading LSTM model from: %s", LSTM_MODEL_FILE)
    model = keras.models.load_model(str(LSTM_MODEL_FILE))

    if use_cache:
        _LSTM_MODEL_CACHE = model

    logger.info("LSTM model loaded successfully")
    return model


def load_lstm_scaler(use_cache: bool = True):
    """Load MinMaxScaler with optional caching."""
    global _LSTM_SCALER_CACHE

    if use_cache and _LSTM_SCALER_CACHE is not None:
        return _LSTM_SCALER_CACHE

    if not LSTM_SCALER_FILE.exists():
        raise FileNotFoundError(
            f"LSTM scaler not found: {LSTM_SCALER_FILE}\n"
            f"Run: python scripts/train_lstm_model.py"
        )

    scaler = joblib.load(LSTM_SCALER_FILE)

    if use_cache:
        _LSTM_SCALER_CACHE = scaler

    logger.info("LSTM scaler loaded from: %s", LSTM_SCALER_FILE)
    return scaler


# ---------------------------------------------------------------------------
# Feature Engineering (mirrors fetch_live_data.py)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the same 21 technical indicator features used during LSTM training.
    Mirrors compute_live_features() in fetch_live_data.py exactly.
    """
    df = df.copy()

    df["MA5"]  = df["Close"].rolling(window=5,  min_periods=5).mean()
    df["MA10"] = df["Close"].rolling(window=10, min_periods=10).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=20).mean()

    df["Daily_Returns"]   = df["Close"].pct_change() * 100
    df["Volatility"]      = df["High"] - df["Low"]
    df["Price_Change_Pct"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100

    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_3"] = df["Close"].shift(3)

    # RSI (14-period, Wilder's smoothing)
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    df["EMA12"] = df["Close"].ewm(span=12, min_periods=12, adjust=False).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()

    sma20 = df["Close"].rolling(window=20, min_periods=20).mean()
    std20 = df["Close"].rolling(window=20, min_periods=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BB_Width"] = ((upper - lower) / sma20.replace(0, np.nan)) * 100

    df["Weekly_Momentum"]    = df["Close"].pct_change(periods=5) * 100
    vol_ma5                  = df["Volume"].rolling(window=5, min_periods=5).mean()
    df["Avg_5D_Volume_Trend"] = df["Volume"] / vol_ma5.replace(0, np.nan)

    ma5  = df["Close"].rolling(window=5,  min_periods=5).mean()
    ma20 = df["Close"].rolling(window=20, min_periods=20).mean()
    df["Trend_Strength"] = ((ma5 - ma20).abs() / ma20.replace(0, np.nan)) * 100

    return df


# ---------------------------------------------------------------------------
# Sequence Builder
# ---------------------------------------------------------------------------

def build_prediction_sequence(df: pd.DataFrame, scaler) -> np.ndarray:
    """
    Build the input sequence for LSTM prediction from a historical DataFrame.

    Steps:
        1. Compute features
        2. Drop rows with NaN (warmup period)
        3. Take the last SEQUENCE_LENGTH rows
        4. Scale with the fitted scaler
        5. Reshape to (1, SEQUENCE_LENGTH, n_features)

    Returns
    -------
    np.ndarray  shape (1, SEQUENCE_LENGTH, n_features)
    """
    df = compute_features(df)

    # Keep only feature columns and drop NaN rows
    df_feat = df[FEATURE_COLUMNS].dropna()

    if len(df_feat) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough data to build LSTM sequence. "
            f"Need at least {SEQUENCE_LENGTH} rows after feature computation, "
            f"got {len(df_feat)}. "
            f"Try fetching more historical data (use outputsize='full' for Alpha Vantage "
            f"or period='1y' for yfinance)."
        )

    # Take the most recent SEQUENCE_LENGTH rows
    sequence = df_feat.tail(SEQUENCE_LENGTH).values.astype(np.float32)

    # Scale
    sequence_scaled = scaler.transform(sequence)

    # Reshape to (1, seq_len, n_features)
    return sequence_scaled.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))


# ---------------------------------------------------------------------------
# Main Prediction Function
# ---------------------------------------------------------------------------

def predict_lstm_live(
    symbol: str,
    source: str = "auto",
    api_key: str = "",
) -> Dict:
    """
    Run LSTM prediction for a given stock symbol using live historical data.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL').
    source : str
        Data source: 'alpha_vantage', 'yfinance', or 'auto'.
    api_key : str
        Alpha Vantage API key (optional if set in .env).

    Returns
    -------
    dict with keys:
        symbol, prediction, confidence, probabilities,
        latest_date, source, sequence_length, model_file
    """
    logger.info("=" * 60)
    logger.info("LSTM PREDICTION PIPELINE — %s", symbol)
    logger.info("=" * 60)

    # 1. Load model and scaler
    model  = load_lstm_model()
    scaler = load_lstm_scaler()

    # 2. Fetch historical data (need enough rows for sequence + warmup)
    #    We request 'full' from Alpha Vantage or '1y' from yfinance
    #    to ensure we have at least SEQUENCE_LENGTH + 30 (warmup) rows
    logger.info("Fetching historical data for %s...", symbol)

    try:
        from fetch_live_data import fetch_alpha_vantage, fetch_yfinance
    except ImportError:
        raise ImportError("fetch_live_data.py not found in scripts/")

    df = None
    used_source = None

    if source in ("alpha_vantage", "auto"):
        try:
            import os
            _api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            # Use 'full' outputsize to get enough history for the sequence
            df = fetch_alpha_vantage(symbol, _api_key, outputsize="full")
            used_source = "alpha_vantage"
        except Exception as e:
            logger.warning("Alpha Vantage failed: %s", e)
            if source == "alpha_vantage":
                raise
            logger.info("Falling back to yfinance...")

    if df is None:
        # Use 1 year of data — enough for 60-day sequence + warmup
        df = fetch_yfinance(symbol, period="1y")
        used_source = "yfinance"

    logger.info("Fetched %d rows from %s", len(df), used_source)

    # 3. Build prediction sequence
    logger.info("Building LSTM input sequence (last %d days)...", SEQUENCE_LENGTH)
    X = build_prediction_sequence(df, scaler)
    logger.info("Sequence shape: %s", X.shape)

    # 4. Run prediction
    logger.info("Running LSTM inference...")
    # sigmoid output: y_proba = P(UP) since Target=1 means UP in training
    y_proba = float(model.predict(X, verbose=0).flatten()[0])
    y_pred  = 1 if y_proba >= 0.5 else 0

    p_up   = y_proba          # probability of UP
    p_down = 1.0 - y_proba    # probability of DOWN

    direction  = "UP" if y_pred == 1 else "DOWN"
    confidence = p_up * 100 if y_pred == 1 else p_down * 100

    logger.info("Prediction: %s | P(UP)=%.4f | P(DOWN)=%.4f | Confidence: %.2f%%",
                direction, p_up, p_down, confidence)

    # 5. Get latest date from data
    latest_date = str(df["Date"].iloc[-1].date()) if hasattr(df["Date"].iloc[-1], "date") else str(df["Date"].iloc[-1])

    result = {
        "symbol":          symbol,
        "prediction":      direction,
        "confidence":      round(confidence, 2),
        "probabilities": {
            "DOWN": round(p_down * 100, 2),
            "UP":   round(p_up   * 100, 2),
        },
        "latest_date":     latest_date,
        "source":          used_source,
        "sequence_length": SEQUENCE_LENGTH,
        "model_file":      str(LSTM_MODEL_FILE),
        "model_type":      "LSTM",
    }

    logger.info("=" * 60)
    logger.info("LSTM PREDICTION COMPLETE")
    logger.info("=" * 60)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM live stock prediction")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--api-key", type=str, default="", help="Alpha Vantage API key")
    parser.add_argument(
        "--source", type=str, default="auto",
        choices=["alpha_vantage", "yfinance", "auto"],
        help="Data source",
    )
    args = parser.parse_args()

    result = predict_lstm_live(
        symbol=args.symbol,
        source=args.source,
        api_key=args.api_key,
    )

    print("\n" + "=" * 60)
    print(f"  LSTM PREDICTION — {result['symbol']}")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
