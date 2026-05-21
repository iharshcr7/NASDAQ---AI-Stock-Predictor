"""
feature_engineering.py
=======================
Create ML-ready features from cleaned stock data, including technical
indicators, lag features, and the binary classification target.

Usage:
    python scripts/feature_engineering.py
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "cleaned_stock_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "final_featured_data.csv"
USE_STABLE_STOCKS_ONLY = True
PREFERRED_SYMBOLS = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"}

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
# Feature Engineering Functions
# ---------------------------------------------------------------------------


def compute_moving_averages(group: pd.DataFrame) -> pd.DataFrame:
    """Compute Simple Moving Averages: MA5, MA10, MA20."""
    group = group.copy()
    group["MA5"] = group["Close"].rolling(window=5, min_periods=5).mean()
    group["MA10"] = group["Close"].rolling(window=10, min_periods=10).mean()
    group["MA20"] = group["Close"].rolling(window=20, min_periods=20).mean()
    return group


def compute_daily_returns(group: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns as percentage change of Close price."""
    group = group.copy()
    group["Daily_Returns"] = group["Close"].pct_change() * 100
    return group


def compute_volatility(group: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday volatility as High - Low."""
    group = group.copy()
    group["Volatility"] = group["High"] - group["Low"]
    return group


def compute_price_change_pct(group: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday price change percentage: (Close - Open) / Open * 100."""
    group = group.copy()
    group["Price_Change_Pct"] = ((group["Close"] - group["Open"]) / group["Open"]) * 100
    return group


def compute_lag_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute lag features: previous day close (Lag_1) and 3-day-ago close (Lag_3)."""
    group = group.copy()
    group["Lag_1"] = group["Close"].shift(1)
    group["Lag_3"] = group["Close"].shift(3)
    return group


def compute_rsi(group: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI) using Wilder's smoothing method.

    RSI = 100 - (100 / (1 + RS))
    RS  = Average Gain / Average Loss  (exponentially weighted)
    """
    group = group.copy()
    delta = group["Close"].diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing (equivalent to EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    group["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    return group


def compute_volume_change(group: pd.DataFrame) -> pd.DataFrame:
    """Compute volume change percentage from previous day."""
    group = group.copy()
    group["Volume_Change_Pct"] = group["Volume"].pct_change() * 100
    return group


def compute_ema(group: pd.DataFrame) -> pd.DataFrame:
    """Compute Exponential Moving Average (12-period)."""
    group = group.copy()
    group["EMA12"] = group["Close"].ewm(span=12, min_periods=12, adjust=False).mean()
    return group


def compute_bollinger_position(group: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute the position of Close price within Bollinger Bands (0 to 1 scale).
    Values near 1 = near upper band, near 0 = near lower band.
    """
    group = group.copy()
    sma = group["Close"].rolling(window=window, min_periods=window).mean()
    std = group["Close"].rolling(window=window, min_periods=window).std()

    upper = sma + 2 * std
    lower = sma - 2 * std
    band_width = upper - lower

    group["BB_Position"] = ((group["Close"] - lower) / band_width.replace(0, np.nan))
    return group


def compute_macd(group: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram."""
    group = group.copy()
    ema_12 = group["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = group["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    group["MACD"] = ema_12 - ema_26
    group["MACD_Signal"] = group["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    group["MACD_Hist"] = group["MACD"] - group["MACD_Signal"]
    return group


def compute_bollinger_width(group: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute normalized Bollinger Band width as a volatility regime feature."""
    group = group.copy()
    sma = group["Close"].rolling(window=window, min_periods=window).mean()
    std = group["Close"].rolling(window=window, min_periods=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    group["BB_Width"] = ((upper - lower) / sma.replace(0, np.nan)) * 100
    return group


def compute_weekly_momentum(group: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """Compute 5-trading-day momentum in percentage terms."""
    group = group.copy()
    group["Weekly_Momentum"] = group["Close"].pct_change(periods=period) * 100
    return group


def compute_volume_trend(group: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute current volume relative to short-term average volume."""
    group = group.copy()
    volume_ma = group["Volume"].rolling(window=window, min_periods=window).mean()
    group["Avg_Volume_Trend"] = group["Volume"] / volume_ma.replace(0, np.nan)
    return group


def compute_avg_5day_volume_trend(group: pd.DataFrame) -> pd.DataFrame:
    """Compute current volume relative to a 5-day average volume baseline."""
    group = group.copy()
    volume_ma_5 = group["Volume"].rolling(window=5, min_periods=5).mean()
    group["Avg_5D_Volume_Trend"] = group["Volume"] / volume_ma_5.replace(0, np.nan)
    return group


def compute_trend_strength(group: pd.DataFrame) -> pd.DataFrame:
    """Compute trend strength as MA5-MA20 distance normalized by MA20."""
    group = group.copy()
    ma5 = group["Close"].rolling(window=5, min_periods=5).mean()
    ma20 = group["Close"].rolling(window=20, min_periods=20).mean()
    group["Trend_Strength"] = ((ma5 - ma20).abs() / ma20.replace(0, np.nan)) * 100
    return group


def compute_rolling_std_returns(group: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute rolling standard deviation of daily returns (risk regime feature)."""
    group = group.copy()
    group["Rolling_Std_Returns"] = group["Daily_Returns"].rolling(window=window, min_periods=window).std()
    return group


def compute_target(group: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification target from next-day return.

    FutureReturn = ((Close[t+1] - Close[t]) / Close[t]) * 100
    Target = 1 if FutureReturn > +1%
    Target = 0 if FutureReturn < -1%
    Rows in [-1%, +1%] are marked as noisy and removed later.
    """
    group = group.copy()
    future_return = ((group["Close"].shift(-1) - group["Close"]) / group["Close"]) * 100
    group["Future_Return_Pct"] = future_return
    group["Target"] = np.select(
        [future_return > 1.0, future_return < -1.0],
        [1, 0],
        default=np.nan,
    )
    return group


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations per stock group.

    Process:
    1. Sort by Symbol + Date
    2. Compute all technical indicators per stock
    3. Create target variable
    4. Drop noisy target rows (-1% to +1%) and NaN rows from warmup windows
    5. Keep leak-free final training dataset
    """
    logger.info("Starting feature engineering on %d rows...", len(df))

    if USE_STABLE_STOCKS_ONLY and "Symbol" in df.columns:
        before_filter = len(df)
        df = df[df["Symbol"].isin(PREFERRED_SYMBOLS)].copy()
        logger.info(
            "Stable-stock filter enabled: kept %d/%d rows for symbols %s",
            len(df), before_filter, sorted(PREFERRED_SYMBOLS),
        )

    # Ensure proper sorting
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # Apply all feature functions per stock group
    feature_functions = [
        ("Moving Averages (MA5, MA10, MA20)", compute_moving_averages),
        ("Daily Returns", compute_daily_returns),
        ("Volatility", compute_volatility),
        ("Price Change %", compute_price_change_pct),
        ("Lag Features (Lag_1, Lag_3)", compute_lag_features),
        ("RSI (14-period)", compute_rsi),
        ("Volume Change %", compute_volume_change),
        ("EMA (12-period)", compute_ema),
        ("Bollinger Band Position", compute_bollinger_position),
        ("MACD (line, signal, histogram)", compute_macd),
        ("Bollinger Width", compute_bollinger_width),
        ("Weekly Momentum (5-day)", compute_weekly_momentum),
        ("Average Volume Trend", compute_volume_trend),
        ("Average 5-Day Volume Trend", compute_avg_5day_volume_trend),
        ("Trend Strength", compute_trend_strength),
        ("Rolling Std Dev of Returns", compute_rolling_std_returns),
        ("Target Variable", compute_target),
    ]

    processed_groups = []

    for symbol, group in df.groupby("Symbol"):
        for name, func in feature_functions:
            group = func(group)
        processed_groups.append(group)
        logger.info("Processed features for %-6s (%d rows)", symbol, len(group))

    df = pd.concat(processed_groups, ignore_index=True)

    # Log feature creation
    for name, _ in feature_functions:
        logger.info("  %s", name)

    # Remove noisy targets (small next-day move) first, then remove NaNs from warmup windows
    initial_len = len(df)
    noisy_rows = df["Target"].isna().sum()
    df = df[df["Target"].notna()].copy()
    logger.info("Removed %d noisy target rows in [-1%%, +1%%] future return band", noisy_rows)

    before_dropna = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(
        "Dropped %d rows with NaN values (rolling warmup, lags, last-row lookahead)",
        before_dropna - len(df),
    )
    logger.info("Total rows removed: %d", initial_len - len(df))

    # Convert target to int (may have become float after concat)
    df["Target"] = df["Target"].astype(int)

    return df


def main() -> None:
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    if not INPUT_FILE.exists():
        logger.error("Input file not found: %s", INPUT_FILE)
        logger.error("Run preprocessing.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    df = engineer_features(df)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    # Summary
    logger.info("-" * 60)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("-" * 60)
    logger.info("Output rows:      %d", len(df))
    logger.info("Features:         %s", [
        c for c in df.columns if c not in ("Date", "Symbol", "Target")
    ])
    logger.info("Target column:    Target")
    logger.info("Target dist:      \n%s", df["Target"].value_counts().to_string())
    logger.info("Symbols:          %d", df["Symbol"].nunique())
    logger.info("File size:        %.2f MB", size_mb)
    logger.info("Output:           %s", OUTPUT_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
