"""
fetch_live_data.py
===================
Fetch real-time stock data from Alpha Vantage API with yfinance fallback.
Computes the same engineered features used in training so that live data
is model-ready.

Usage:
    python scripts/fetch_live_data.py --symbol AAPL
    python scripts/fetch_live_data.py --symbol TSLA --source yfinance
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from model_config import get_expected_features, validate_feature_schema

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs when python-dotenv is unavailable."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    load_env_file(PROJECT_ROOT / ".env")

# API key from environment variable (preferred) or fallback
DEFAULT_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Features the trained model expects (must match train_model.py)
MODEL_FEATURES = get_expected_features()

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
# Data Fetching — Alpha Vantage
# ---------------------------------------------------------------------------


def fetch_alpha_vantage(symbol: str, api_key: str, outputsize: str = "compact") -> pd.DataFrame:
    """
    Fetch daily stock data from Alpha Vantage TIME_SERIES_DAILY endpoint.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL').
    api_key : str
        Alpha Vantage API key.
    outputsize : str
        'compact' (last 100 days) or 'full' (20+ years).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    if not api_key:
        raise ValueError(
            "Alpha Vantage API key not provided. "
            "Set ALPHA_VANTAGE_API_KEY environment variable or pass --api-key."
        )

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }

    logger.info("Fetching %s from Alpha Vantage (outputsize=%s)...", symbol, outputsize)

    response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Check for API errors
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit: {data['Note']}")
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Unexpected API response format. Keys: {list(data.keys())}")

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Rename columns
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.astype(float)
    df.insert(0, "Date", df.index)
    df = df.reset_index(drop=True)

    logger.info(
        "Alpha Vantage: %d rows fetched | %s → %s",
        len(df), df["Date"].iloc[0].date(), df["Date"].iloc[-1].date(),
    )
    return df


# ---------------------------------------------------------------------------
# Data Fetching — yfinance (backup)
# ---------------------------------------------------------------------------


def fetch_yfinance(symbol: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetch stock data using yfinance as a backup source.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL').
    period : str
        Data period: '1mo', '3mo', '6mo', '1y', etc.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    logger.info("Fetching %s from yfinance (period=%s)...", symbol, period)

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)

    if hist.empty:
        raise ValueError(f"No data returned from yfinance for {symbol}")

    df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.insert(0, "Date", df.index)
    df = df.reset_index(drop=True)

    logger.info(
        "yfinance: %d rows fetched | %s → %s",
        len(df), df["Date"].iloc[0].date(), df["Date"].iloc[-1].date(),
    )
    return df


# ---------------------------------------------------------------------------
# Feature Computation (mirrors feature_engineering.py)
# ---------------------------------------------------------------------------


def compute_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the same engineered features used during training.
    Expects a DataFrame with at least 30 rows of historical data.
    """
    df = df.copy()

    # Moving Averages
    df["MA5"] = df["Close"].rolling(window=5, min_periods=5).mean()
    df["MA10"] = df["Close"].rolling(window=10, min_periods=10).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=20).mean()

    # Daily Returns
    df["Daily_Returns"] = df["Close"].pct_change() * 100

    # Volatility
    df["Volatility"] = df["High"] - df["Low"]

    # Price Change %
    df["Price_Change_Pct"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100

    # Lag Features
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_3"] = df["Close"].shift(3)

    # RSI (14-period, Wilder's smoothing)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # Volume Change %
    df["Volume_Change_Pct"] = df["Volume"].pct_change() * 100

    # EMA 12
    df["EMA12"] = df["Close"].ewm(span=12, min_periods=12, adjust=False).mean()

    # Bollinger Band Position
    sma20 = df["Close"].rolling(window=20, min_periods=20).mean()
    std20 = df["Close"].rolling(window=20, min_periods=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_width = upper - lower
    df["BB_Position"] = (df["Close"] - lower) / band_width.replace(0, np.nan)

    # MACD family
    ema12_fast = df["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26_slow = df["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD"] = ema12_fast - ema26_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Width (normalized)
    df["BB_Width"] = ((upper - lower) / sma20.replace(0, np.nan)) * 100

    # Weekly momentum and volume trend
    df["Weekly_Momentum"] = df["Close"].pct_change(periods=5) * 100
    vol_ma10 = df["Volume"].rolling(window=10, min_periods=10).mean()
    df["Avg_Volume_Trend"] = df["Volume"] / vol_ma10.replace(0, np.nan)
    vol_ma5 = df["Volume"].rolling(window=5, min_periods=5).mean()
    df["Avg_5D_Volume_Trend"] = df["Volume"] / vol_ma5.replace(0, np.nan)

    # Trend strength and rolling return risk
    df["Trend_Strength"] = ((df["MA5"] - df["MA20"]).abs() / df["MA20"].replace(0, np.nan)) * 100
    df["Rolling_Std_Returns"] = df["Daily_Returns"].rolling(window=10, min_periods=10).std()

    return df


def validate_live_feature_schema(feature_columns: list[str]) -> None:
    validate_feature_schema(feature_columns)


# ---------------------------------------------------------------------------
# Unified Fetch + Feature Pipeline
# ---------------------------------------------------------------------------


def fetch_live_stock_data(
    symbol: str,
    api_key: str = "",
    source: str = "auto",
) -> dict:
    """
    Fetch live stock data and compute model-ready features.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol.
    api_key : str
        Alpha Vantage API key. Ignored if source='yfinance'.
    source : str
        'alpha_vantage', 'yfinance', or 'auto' (try AV first, fallback to yf).

    Returns
    -------
    dict
        Contains: symbol, latest quote data, computed features, source used,
        and the feature vector ready for model prediction.
    """
    df = None
    used_source = None

    # --- Fetch raw data ---
    if source in ("alpha_vantage", "auto"):
        try:
            df = fetch_alpha_vantage(symbol, api_key or DEFAULT_API_KEY)
            used_source = "alpha_vantage"
        except Exception as exc:
            logger.warning("Alpha Vantage failed: %s", exc)
            if source == "alpha_vantage":
                raise

    if df is None:
        df = fetch_yfinance(symbol)
        used_source = "yfinance"

    # --- Compute features ---
    df = compute_live_features(df)

    validate_live_feature_schema(MODEL_FEATURES)

    # Get the latest complete row (with all features computed)
    latest = df.dropna(subset=MODEL_FEATURES).iloc[-1]

    result = {
        "symbol": symbol,
        "source": used_source,
        "latest_date": str(latest["Date"].date()) if hasattr(latest["Date"], "date") else str(latest["Date"]),
        "quote": {
            "open": round(float(latest["Open"]), 4),
            "high": round(float(latest["High"]), 4),
            "low": round(float(latest["Low"]), 4),
            "close": round(float(latest["Close"]), 4),
            "volume": int(latest["Volume"]),
        },
        "features": {col: round(float(latest[col]), 6) for col in MODEL_FEATURES},
        "feature_vector": [float(latest[col]) for col in MODEL_FEATURES],
    }

    logger.info(
        "Live data ready for %s (source=%s, date=%s)",
        symbol, used_source, result["latest_date"],
    )
    return result


def get_historical_for_chart(
    symbol: str,
    api_key: str = "",
    source: str = "auto",
) -> pd.DataFrame:
    """
    Fetch recent historical data suitable for charting (candlestick / line).
    Returns a clean DataFrame with Date, Open, High, Low, Close, Volume.
    """
    df = None

    if source in ("alpha_vantage", "auto"):
        try:
            df = fetch_alpha_vantage(symbol, api_key or DEFAULT_API_KEY)
        except Exception:
            pass

    if df is None:
        df = fetch_yfinance(symbol, period="3mo")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live stock data")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--api-key", type=str, default="", help="Alpha Vantage API key")
    parser.add_argument(
        "--source", type=str, default="auto",
        choices=["alpha_vantage", "yfinance", "auto"],
        help="Data source",
    )
    args = parser.parse_args()

    result = fetch_live_stock_data(
        symbol=args.symbol,
        api_key=args.api_key,
        source=args.source,
    )

    print("\n" + "=" * 60)
    print(f"  LIVE DATA — {result['symbol']}")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
