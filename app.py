"""
NASDAQ AI Stock Predictor — Streamlit Dashboard
=================================================
Premium dark-themed dashboard with live/manual prediction modes,
candlestick charts, historical graphs, and confidence scores.

Usage:
    streamlit run app.py
"""

import sys
import os
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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


# Load .env file for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    load_env_file(PROJECT_ROOT / ".env")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime
from model_config import FINAL_MODEL_FILE, MODEL_METADATA_FILE, get_expected_features
from predict_live import predict_live, get_supported_stocks
from mongo_store import fetch_recent_predictions, save_prediction
from fetch_live_data import get_historical_for_chart, compute_live_features
from symbol_mapper import search_stocks, get_company_name, resolve_symbol
from predict_lstm import (
    predict_lstm_live,
    is_lstm_available,
    LSTM_METADATA_FILE,
    load_lstm_model,
    load_lstm_scaler,
    FEATURE_COLUMNS as LSTM_FEATURE_COLUMNS,
    SEQUENCE_LENGTH as LSTM_SEQUENCE_LENGTH,
    compute_features as compute_lstm_features,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NASDAQ AI Predictor",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium Dark Theme CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main, .stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #111128 50%, #0d0d24 100%);
}
.main-header {
    text-align: center; padding: 1.5rem 0 1rem 0;
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08));
    border-radius: 16px; border: 1px solid rgba(99,102,241,0.15);
    margin-bottom: 1.5rem;
}
.main-header h1 {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.main-header p { color: #94a3b8; font-size: 0.95rem; margin: 0.3rem 0 0 0; }
.metric-card {
    background: linear-gradient(135deg, rgba(30,30,60,0.9), rgba(20,20,50,0.9));
    border: 1px solid rgba(99,102,241,0.2); border-radius: 12px;
    padding: 1.2rem; text-align: center; backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(99,102,241,0.5); }
.metric-label {
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: #94a3b8; margin-bottom: 0.3rem;
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }
.pred-up {
    background: linear-gradient(135deg, rgba(0,200,83,0.15), rgba(0,230,118,0.08));
    border: 2px solid rgba(0,200,83,0.4); border-radius: 16px;
    padding: 1.5rem; text-align: center;
}
.pred-down {
    background: linear-gradient(135deg, rgba(255,23,68,0.15), rgba(255,82,82,0.08));
    border: 2px solid rgba(255,23,68,0.4); border-radius: 16px;
    padding: 1.5rem; text-align: center;
}
.pred-label { font-size: 2rem; font-weight: 800; }
.pred-conf { font-size: 1rem; color: #94a3b8; margin-top: 0.3rem; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f2a 0%, #141432 100%);
    border-right: 1px solid rgba(99,102,241,0.15);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = FINAL_MODEL_FILE
FEATURE_COLUMNS = get_expected_features()

# Get supported stocks from predict_live.py (production stocks with full pipeline support)
SUPPORTED_STOCKS_LIST = get_supported_stocks()

# Build dynamic stock dropdown with company names
STOCKS = {}
for symbol in sorted(SUPPORTED_STOCKS_LIST[:100]):  # Show first 100 for UI performance
    company_name = get_company_name(symbol)
    if company_name:
        STOCKS[f"{company_name} ({symbol})"] = symbol
    else:
        STOCKS[symbol] = symbol

# Add popular stocks at the top if not already present
POPULAR_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
}

# Merge popular stocks at the beginning
for key, value in POPULAR_STOCKS.items():
    if key not in STOCKS:
        STOCKS = {key: value, **STOCKS}

# ---------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>NASDAQ AI Stock Predictor</h1>
    <p>Hybrid Machine Learning — Structured + Unstructured Data Analysis</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Model not found. Run `python scripts/train_model.py` first to train the model.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Configuration")
    
    # Stock selection with search
    st.markdown("#### Stock Selection")
    
    # Search box for finding stocks
    search_query = st.text_input(
        "Search stocks",
        placeholder="Type symbol or company name (e.g., TSLA, Tesla, Apple)",
        help="Search by stock symbol or company name"
    )
    
    if search_query:
        # Perform search
        search_results = search_stocks(search_query, limit=20)
        
        if search_results:
            st.caption(f"Found {len(search_results)} matches:")
            
            # Build search results dropdown
            search_stocks_dict = {}
            for sym, name in search_results:
                if name:
                    search_stocks_dict[f"{name} ({sym})"] = sym
                else:
                    search_stocks_dict[sym] = sym
            
            selected_stock = st.selectbox(
                "Select from results",
                list(search_stocks_dict.keys()),
                index=0,
                key="search_results"
            )
            symbol = search_stocks_dict[selected_stock]
        else:
            st.warning(f"No stocks found matching '{search_query}'")
            selected_stock = st.selectbox("Select Stock", list(STOCKS.keys()), index=0)
            symbol = STOCKS[selected_stock]
    else:
        # Default dropdown with popular stocks
        selected_stock = st.selectbox("Select Stock", list(STOCKS.keys()), index=0)
        symbol = STOCKS[selected_stock]
    
    st.caption(f"Total available stocks: {len(SUPPORTED_STOCKS_LIST):,}")

    st.markdown("---")
    mode = st.radio("Data Source", ["Live Data", "Manual Input"], index=0)

    st.markdown("---")
    st.markdown("### RF Model Info")
    meta_path = MODEL_METADATA_FILE
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        st.metric("Accuracy", f"{meta['metrics']['accuracy']*100:.1f}%")
        st.metric("ROC AUC", f"{meta['metrics']['roc_auc']:.3f}")
        st.metric("F1 Score", f"{meta['metrics']['f1_score']:.3f}")
        st.caption(f"Model: {meta.get('selected_model', meta.get('model_type', 'RandomForest'))}")
        st.caption(f"Trained: {meta.get('trained_at', 'N/A')[:10]}")
    else:
        st.info("Train RF model to see metrics")

    # -----------------------------------------------------------------------
    # Model availability status
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Models")

    lstm_ready = is_lstm_available()

    if lstm_ready:
        st.success("LSTM — ready")
        st.info("Random Forest — ready")
    else:
        st.info("Random Forest — ready")
        st.warning("LSTM — not trained")
        st.caption("Run: `python scripts/train_lstm_model.py`")

    # Show LSTM metadata if available
    lstm_meta_path = Path(str(LSTM_METADATA_FILE))
    if lstm_ready and lstm_meta_path.exists():
        with st.expander("LSTM Model Info"):
            lstm_meta = json.loads(lstm_meta_path.read_text())
            st.metric("LSTM Accuracy", f"{lstm_meta['metrics']['accuracy']*100:.1f}%")
            st.metric("LSTM ROC AUC", f"{lstm_meta['metrics']['roc_auc']:.3f}")
            st.metric("LSTM F1 Score", f"{lstm_meta['metrics']['f1_score']:.3f}")
            st.caption(f"Seq Length: {lstm_meta.get('sequence_length', 60)} days")
            st.caption(f"Trained: {lstm_meta.get('trained_at', 'N/A')[:10]}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def predict_direction(features_dict: dict) -> tuple:
    """Run prediction using the trained model. Returns (direction, confidence, probabilities)."""
    feature_df = pd.DataFrame([{c: features_dict[c] for c in FEATURE_COLUMNS}])
    pred = model.predict(feature_df)[0]
    probs = model.predict_proba(feature_df)[0]
    return ("UP" if pred == 1 else "DOWN"), float(probs.max()) * 100, probs


def render_metric(label: str, value: str, col):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>""", unsafe_allow_html=True)


def render_prediction(direction: str, confidence: float):
    is_up = "UP" in direction
    cls   = "pred-up" if is_up else "pred-down"
    color = "#00c853" if is_up else "#ff1744"
    arrow = "STOCK WILL GO UP" if is_up else "STOCK WILL GO DOWN"
    st.markdown(f"""
    <div class="{cls}">
        <div class="pred-label" style="color:{color};">{arrow}</div>
        <div class="pred-conf">Confidence: {confidence:.1f}%</div>
    </div>""", unsafe_allow_html=True)


def make_candlestick(df, title):
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00c853", decreasing_fillcolor="#ff1744",
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        height=450, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,40,0.8)",
        xaxis_rangeslider_visible=False,
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=30, t=50, b=40),
    )
    return fig


def make_close_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"], mode="lines", name="Close",
        line=dict(color="#818cf8", width=2),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.1)",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        height=350, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=30, t=50, b=40), yaxis_title="Price ($)",
    )
    return fig


# ---------------------------------------------------------------------------
# NEW: Live Volume Bar Chart
# ---------------------------------------------------------------------------

def create_volume_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Create live volume bar chart with bullish/bearish coloring.
    
    Args:
        df: DataFrame with Date, Open, Close, Volume columns
        symbol: Stock symbol for title
        
    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Volume Chart - No Data Available",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
    
    # Determine bar colors based on price movement
    # Green for bullish (Close > Open), Red for bearish (Close <= Open)
    colors = [
        "#00c853" if close > open_price else "#ff1744"
        for open_price, close in zip(df["Open"], df["Close"])
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["Volume"],
        name="Volume",
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        hovertemplate=(
            "<b>Date:</b> %{x}<br>"
            "<b>Volume:</b> %{y:,.0f}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f"{symbol} — Live Volume Analysis",
            font=dict(size=16, color="#e2e8f0")
        ),
        height=300,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Volume", tickformat=","),
        showlegend=False,
        hovermode="x unified"
    )
    
    return fig


# ---------------------------------------------------------------------------
# NEW: Live Correlation Heatmap
# ---------------------------------------------------------------------------

def create_correlation_heatmap(df: pd.DataFrame, features: dict = None, window: int = 50) -> go.Figure:
    """
    Create live correlation heatmap from recent rolling window of data.
    
    Args:
        df: Historical DataFrame with technical indicators
        features: Dictionary of live features (optional, for single-point analysis)
        window: Rolling window size (default: 50 recent records)
        
    Returns:
        Plotly Figure object
    """
    # Define features to analyze
    feature_cols = [
        "Close", "Volume", "RSI", "MACD", "Volatility",
        "MA5", "MA20", "Daily_Returns", "Trend_Strength"
    ]
    
    # If we have a DataFrame with historical data, use rolling window
    if df is not None and not df.empty and len(df) > 1:
        # Take last N records for live rolling correlation
        df_window = df.tail(min(window, len(df))).copy()
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df_window.columns]
        
        if len(available_cols) < 2:
            # Not enough features, return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Correlation Heatmap - Insufficient Data",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,40,0.8)",
            )
            return fig
        
        # Compute correlation matrix
        corr_matrix = df_window[available_cols].corr()
        
    elif features is not None:
        # Single point - create synthetic correlation from feature dict
        # This is a fallback for when we only have current features
        available_cols = [col for col in feature_cols if col in features]
        
        if len(available_cols) < 2:
            fig = go.Figure()
            fig.update_layout(
                title="Correlation Heatmap - Insufficient Features",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,15,40,0.8)",
            )
            return fig
        
        # Create identity matrix as placeholder
        corr_matrix = pd.DataFrame(
            np.eye(len(available_cols)),
            index=available_cols,
            columns=available_cols
        )
    else:
        # No data available
        fig = go.Figure()
        fig.update_layout(
            title="Correlation Heatmap - No Data Available",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[
            [0.0, "#ff1744"],      # Strong negative - Red
            [0.25, "#ff5252"],     # Negative - Light Red
            [0.5, "#424242"],      # Neutral - Dark Gray
            [0.75, "#69f0ae"],     # Positive - Light Green
            [1.0, "#00c853"]       # Strong positive - Green
        ],
        zmid=0,
        text=corr_matrix.values,
        texttemplate="%{text:.2f}",
        textfont=dict(size=10, color="#e2e8f0"),
        hovertemplate=(
            "<b>%{y} vs %{x}</b><br>"
            "Correlation: %{z:.3f}<br>"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Correlation", side="right"),
            tickmode="linear",
            tick0=-1,
            dtick=0.5,
            thickness=15,
            len=0.7,
            tickfont=dict(color="#94a3b8")
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Live Correlation Matrix (Last {min(window, len(df) if df is not None else 0)} Records)",
            font=dict(size=16, color="#e2e8f0")
        ),
        height=500,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=100, r=50, t=80, b=80),
        xaxis=dict(
            side="bottom",
            tickangle=-45,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            tickfont=dict(size=11)
        )
    )
    
    return fig


# ---------------------------------------------------------------------------
# NEW: Live Daily Returns Histogram
# ---------------------------------------------------------------------------

def create_returns_histogram(df: pd.DataFrame, symbol: str, window: int = 100) -> go.Figure:
    """
    Create live daily returns histogram from recent rolling window.
    
    Args:
        df: DataFrame with Date and Close columns
        symbol: Stock symbol for title
        window: Rolling window size (default: 100 recent records)
        
    Returns:
        Plotly Figure object
    """
    if df is None or df.empty or len(df) < 2:
        # Return empty figure if insufficient data
        fig = go.Figure()
        fig.update_layout(
            title="Daily Returns Histogram - Insufficient Data",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
    
    # Take last N records for live rolling analysis
    df_window = df.tail(min(window, len(df))).copy()
    
    # Compute daily returns if not already present
    if "Daily_Returns" not in df_window.columns:
        df_window["Daily_Returns"] = (
            (df_window["Close"] - df_window["Close"].shift(1)) / 
            df_window["Close"].shift(1) * 100
        )
    
    # Remove NaN values
    returns = df_window["Daily_Returns"].dropna()
    
    if len(returns) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Daily Returns Histogram - No Valid Returns",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
    
    # Calculate statistics
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        name="Daily Returns",
        marker=dict(
            color="#818cf8",
            line=dict(color="#a78bfa", width=1)
        ),
        hovertemplate=(
            "<b>Return Range:</b> %{x:.2f}%<br>"
            "<b>Frequency:</b> %{y}<br>"
            "<extra></extra>"
        )
    ))
    
    # Add mean line
    fig.add_vline(
        x=mean_return,
        line=dict(color="#00c853", width=2, dash="dash"),
        annotation=dict(
            text=f"Mean: {mean_return:.2f}%",
            font=dict(size=11, color="#00c853")
        )
    )
    
    # Add +/- 1 std lines
    fig.add_vline(
        x=mean_return + std_return,
        line=dict(color="#ff9800", width=1, dash="dot"),
        annotation=dict(
            text=f"+1σ: {mean_return + std_return:.2f}%",
            font=dict(size=10, color="#ff9800")
        )
    )
    
    fig.add_vline(
        x=mean_return - std_return,
        line=dict(color="#ff9800", width=1, dash="dot"),
        annotation=dict(
            text=f"-1σ: {mean_return - std_return:.2f}%",
            font=dict(size=10, color="#ff9800")
        )
    )
    
    fig.update_layout(
        title=dict(
            text=f"{symbol} — Live Daily Returns Distribution (Last {len(returns)} Days)",
            font=dict(size=16, color="#e2e8f0")
        ),
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=30, t=80, b=40),
        xaxis=dict(
            title="Daily Return (%)",
            tickformat=".2f"
        ),
        yaxis=dict(
            title="Frequency"
        ),
        showlegend=False,
        hovermode="x"
    )
    
    # Add statistics annotation
    fig.add_annotation(
        text=(
            f"<b>Statistics:</b><br>"
            f"Mean: {mean_return:.3f}%<br>"
            f"Std Dev: {std_return:.3f}%<br>"
            f"Min: {returns.min():.2f}%<br>"
            f"Max: {returns.max():.2f}%"
        ),
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        xanchor="right", yanchor="top",
        showarrow=False,
        bgcolor="rgba(30,30,60,0.8)",
        bordercolor="rgba(99,102,241,0.3)",
        borderwidth=1,
        borderpad=10,
        font=dict(size=11, color="#e2e8f0")
    )
    
    return fig


# ---------------------------------------------------------------------------
# LSTM True Stock-Specific Helper Functions
# ---------------------------------------------------------------------------

def run_lstm_backtest(hist_df: pd.DataFrame, symbol: str):
    """
    Sequential backtest of the LSTM model over the last 30 trading days of the stock.
    Runs in a single efficient batch inference pass.
    """
    if hist_df is None or hist_df.empty or len(hist_df) < LSTM_SEQUENCE_LENGTH + 5:
        return pd.DataFrame(), 0.0
        
    try:
        # 1. Compute features using predict_lstm feature pipeline
        df_feat = compute_lstm_features(hist_df)
        df_feat = df_feat.dropna(subset=LSTM_FEATURE_COLUMNS).copy()
        
        if len(df_feat) < LSTM_SEQUENCE_LENGTH + 1:
            return pd.DataFrame(), 0.0
            
        # Load model and scaler (fast cached load)
        model = load_lstm_model()
        scaler = load_lstm_scaler()
        
        # Determine number of test days
        num_test_days = min(30, len(df_feat) - LSTM_SEQUENCE_LENGTH)
        
        # Build batch of sequences
        X_batch = []
        features_data = df_feat[LSTM_FEATURE_COLUMNS].values.astype(np.float32)
        features_scaled = scaler.transform(features_data)
        
        for offset in range(num_test_days, 0, -1):
            k = len(df_feat) - offset
            seq = features_scaled[k - LSTM_SEQUENCE_LENGTH : k]
            X_batch.append(seq)
            
        X_batch = np.array(X_batch, dtype=np.float32)
        
        # Predict batch (returns Sigmoid probability of UP direction)
        y_probas = model.predict(X_batch, verbose=0).flatten()
        
        # Extract dates, prices, predictions and actual outcomes
        dates = []
        closes = []
        probs = []
        predictions = []
        actuals = []
        correctness = []
        
        for idx, offset in enumerate(range(num_test_days, 0, -1)):
            k = len(df_feat) - offset
            y_proba = float(y_probas[idx])
            pred_signal = 1 if y_proba >= 0.5 else 0
            
            # Day T is k - 1 (the final day of sequence)
            # Day T+1 is k (the day we are predicting movement FOR)
            close_today = df_feat.iloc[k - 1]["Close"]
            close_next = df_feat.iloc[k]["Close"]
            
            actual_signal = 1 if close_next > close_today else 0
            
            dates.append(df_feat.iloc[k]["Date"])
            closes.append(close_next)
            probs.append(y_proba)
            predictions.append("UP" if pred_signal == 1 else "DOWN")
            actuals.append("UP" if actual_signal == 1 else "DOWN")
            correctness.append(pred_signal == actual_signal)
            
        backtest_df = pd.DataFrame({
            "Date": dates,
            "Close": closes,
            "Prob": probs,
            "Prediction": predictions,
            "Actual": actuals,
            "Correct": correctness
        })
        
        accuracy = (backtest_df["Correct"].sum() / len(backtest_df)) * 100 if len(backtest_df) > 0 else 0.0
        return backtest_df, accuracy
        
    except Exception as e:
        logger.error(f"Error in run_lstm_backtest: {e}")
        return pd.DataFrame(), 0.0


def create_lstm_forecast_chart(hist_df: pd.DataFrame, lstm_res: dict) -> go.Figure:
    """
    Plots the last 60 trading days of Close prices (the sequence input)
    and draws tomorrow's forecasted next-day trajectory.
    """
    symbol = lstm_res["symbol"]
    pred = lstm_res["prediction"] # "UP" or "DOWN"
    confidence = lstm_res["confidence"]
    
    # Slice the last 60 days
    df_seq = hist_df.tail(60).copy()
    
    fig = go.Figure()
    
    # 1. Historical Close line
    fig.add_trace(go.Scatter(
        x=df_seq["Date"], y=df_seq["Close"],
        mode="lines", name="Historical Close Price",
        line=dict(color="#6366f1", width=3),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.05)"
    ))
    
    # Calculate projection T+1 point
    latest_date = df_seq["Date"].iloc[-1]
    latest_close = df_seq["Close"].iloc[-1]
    
    if isinstance(latest_date, str):
        try:
            latest_dt = pd.to_datetime(latest_date)
            next_dt = latest_dt + pd.tseries.offsets.BDay(1)
        except Exception:
            next_dt = latest_date + " (T+1)"
    else:
        next_dt = latest_date + pd.tseries.offsets.BDay(1)
        
    # Projected price point (2% directional trend lines for aesthetics)
    proj_change = (latest_close * 0.02) if "UP" in pred else (-latest_close * 0.02)
    proj_price = latest_close + proj_change
    
    proj_color = "#00c853" if "UP" in pred else "#ff1744"
    
    # 2. Projection line (dashed)
    fig.add_trace(go.Scatter(
        x=[latest_date, next_dt],
        y=[latest_close, proj_price],
        mode="lines+markers",
        name=f"Projected {pred.split()[0]}",
        line=dict(color=proj_color, width=3, dash="dash"),
        marker=dict(size=10, symbol="triangle-up" if "UP" in pred else "triangle-down", color=proj_color)
    ))
    
    # 3. Highlight final close point
    fig.add_trace(go.Scatter(
        x=[latest_date], y=[latest_close],
        mode="markers", name="Latest Close",
        marker=dict(size=12, color="#a78bfa", line=dict(color="#ffffff", width=2))
    ))
    
    # Annotation for forecast
    fig.add_annotation(
        x=next_dt, y=proj_price,
        text=f"<b>LSTM FORECAST: {pred}</b><br>Confidence: {confidence:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=proj_color,
        ax=40,
        ay=-30 if "UP" in pred else 30,
        bgcolor="rgba(30,30,60,0.9)",
        bordercolor=proj_color,
        borderwidth=2,
        borderpad=6,
        font=dict(size=12, color="#e2e8f0")
    )
    
    fig.update_layout(
        title=dict(
            text=f"LSTM 60-Day Input Sequence & Forecast Timeline ({symbol})",
            font=dict(size=16, color="#e2e8f0")
        ),
        height=450,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=80, t=80, b=40),
        yaxis=dict(
            title="Close Price ($)",
            gridcolor="rgba(255,255,255,0.05)",
            tickprefix="$"
        ),
        xaxis=dict(
            title="Trading Date",
            gridcolor="rgba(255,255,255,0.05)",
        ),
        showlegend=False
    )
    
    return fig


def create_lstm_backtest_chart(backtest_df: pd.DataFrame, symbol: str, accuracy: float) -> go.Figure:
    """
    Plots the close price over the last 30 days and overlays color-coded prediction signals
    with solid green/red for correct predictions and empty circles/red crosses for incorrect predictions.
    """
    fig = go.Figure()
    
    if backtest_df.empty:
        fig.update_layout(
            title="LSTM Backtest - Insufficient Data",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
        
    # 1. Plot Close Price line of the backtest period
    fig.add_trace(go.Scatter(
        x=backtest_df["Date"], y=backtest_df["Close"],
        mode="lines", name="Actual Close",
        line=dict(color="#94a3b8", width=2, dash="solid"),
    ))
    
    # 2. Add scatter points for Correct predictions
    correct_df = backtest_df[backtest_df["Correct"] == True]
    if not correct_df.empty:
        fig.add_trace(go.Scatter(
            x=correct_df["Date"], y=correct_df["Close"],
            mode="markers",
            name="Correct Signal",
            marker=dict(
                size=12,
                color="#00c853",
                line=dict(color="#ffffff", width=1.5),
                symbol="circle",
                opacity=0.9
            ),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Price:</b> $%{y:.2f}<br>"
                "<b>LSTM Prediction:</b> %{customdata[0]} (Correct)<br>"
                "<extra></extra>"
            ),
            customdata=np.stack([correct_df["Prediction"]], axis=-1)
        ))
        
    # 3. Add scatter points for Incorrect predictions
    incorrect_df = backtest_df[backtest_df["Correct"] == False]
    if not incorrect_df.empty:
        fig.add_trace(go.Scatter(
            x=incorrect_df["Date"], y=incorrect_df["Close"],
            mode="markers",
            name="Incorrect Signal",
            marker=dict(
                size=12,
                color="#ff1744",
                line=dict(color="#ffffff", width=1.5),
                symbol="x",
                opacity=0.9
            ),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Price:</b> $%{y:.2f}<br>"
                "<b>LSTM Prediction:</b> %{customdata[0]} (Incorrect)<br>"
                "<extra></extra>"
            ),
            customdata=np.stack([incorrect_df["Prediction"]], axis=-1)
        ))
        
    fig.update_layout(
        title=dict(
            text=f"LSTM 30-Day Signal Backtest ({symbol}) — Historical Accuracy: {accuracy:.1f}%",
            font=dict(size=16, color="#e2e8f0")
        ),
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,40,0.8)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=50, r=30, t=80, b=40),
        yaxis=dict(
            title="Close Price ($)",
            gridcolor="rgba(255,255,255,0.05)",
            tickprefix="$"
        ),
        xaxis=dict(
            title="Trading Date",
            gridcolor="rgba(255,255,255,0.05)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    
    return fig


def create_lstm_metric_chart(metric_name: str) -> go.Figure:
    """
    Plots the global LSTM model training history (Epoch vs Loss/Accuracy).
    metric_name can be "loss" or "accuracy".
    """
    history_file = PROJECT_ROOT / "models" / "lstm_history.json"
    fig = go.Figure()
    
    if not history_file.exists():
        fig.update_layout(
            title=f"LSTM {metric_name.capitalize()} History - No Data Found",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
        )
        return fig
        
    try:
        history = json.loads(history_file.read_text())
        train_data = history.get(metric_name, [])
        val_name = f"val_{metric_name}"
        val_data = history.get(val_name, [])
        
        # Build x-axes dynamically based on list length
        train_epochs = list(range(1, len(train_data) + 1))
        val_epochs = list(range(1, len(val_data) + 1))
        
        color_train = "#6366f1" if metric_name == "loss" else "#00c853"
        color_val = "#a78bfa" if metric_name == "loss" else "#ff1744"
        
        if train_data:
            fig.add_trace(go.Scatter(
                x=train_epochs, y=train_data,
                mode="lines", name=f"Train {metric_name.capitalize()}",
                line=dict(color=color_train, width=2.5)
            ))
            
        if val_data:
            fig.add_trace(go.Scatter(
                x=val_epochs, y=val_data,
                mode="lines+markers", name=f"Validation {metric_name.capitalize()}",
                line=dict(color=color_val, width=2, dash="dash"),
                marker=dict(size=4, color=color_val)
            ))
            
        fig.update_layout(
            title=dict(
                text=f"Global LSTM Training {metric_name.capitalize()} vs Epochs",
                font=dict(size=14, color="#e2e8f0")
            ),
            height=350,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,15,40,0.8)",
            font=dict(family="Inter", color="#94a3b8"),
            margin=dict(l=50, r=30, t=60, b=40),
            yaxis=dict(
                title=metric_name.capitalize(),
                gridcolor="rgba(255,255,255,0.05)"
            ),
            xaxis=dict(
                title="Epoch",
                gridcolor="rgba(255,255,255,0.05)"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)"
            )
        )
    except Exception as e:
        logger.error(f"Error creating metric chart for {metric_name}: {e}")
        
    return fig


# ---------------------------------------------------------------------------
# LIVE MODE - PRODUCTION PIPELINE
# ---------------------------------------------------------------------------

if mode == "Live Data":
    st.markdown(f"### Live Prediction Pipeline — {selected_stock}")

    if symbol not in SUPPORTED_STOCKS_LIST:
        st.warning(f"{symbol} is not in the production pipeline.")
        st.stop()

    st.markdown("---")

    # ── Options row ──────────────────────────────────────────────────────────
    col_src, col_hdfs, col_mongo = st.columns(3)
    with col_src:
        data_source = st.selectbox(
            "Data Source",
            ["auto", "alpha_vantage", "yfinance"],
            index=0,
            help="auto: tries Alpha Vantage first, falls back to Yahoo Finance"
        )
    with col_hdfs:
        skip_hdfs = st.checkbox("Skip HDFS", value=True,
                                help="Uncheck only if Hadoop is running")
    with col_mongo:
        skip_mongo = st.checkbox("Skip MongoDB", value=False)

    st.markdown("")

    # ── Two predict buttons side by side ─────────────────────────────────────
    col_rf_btn, col_lstm_btn = st.columns(2)
    with col_rf_btn:
        rf_button = st.button(
            "Predict with Random Forest",
            type="primary",
            use_container_width=True,
            help="Predict using Random Forest — uses 21 technical indicator features"
        )
    with col_lstm_btn:
        lstm_button = st.button(
            "Predict with LSTM",
            type="primary" if lstm_ready else "secondary",
            use_container_width=True,
            disabled=not lstm_ready,
            help="Predict using LSTM — uses 60-day sequence of historical data"
            if lstm_ready else "Train LSTM first: python scripts/train_lstm_model.py"
        )

    st.markdown("---")

    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

    # ── RF Button clicked ─────────────────────────────────────────────────────
    if rf_button:
        # Clear any previous results so stale data never shows
        for k in ['rf_result', 'rf_hist_df', 'lstm_result', 'active_model']:
            st.session_state.pop(k, None)

        with st.spinner(f"Running Random Forest prediction for {symbol}..."):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            try:
                status_text.text("Fetching live data...")
                progress_bar.progress(25)
                status_text.text("Saving CSV & uploading to HDFS...")
                progress_bar.progress(50)
                status_text.text("Engineering features...")
                progress_bar.progress(75)
                status_text.text("Running Random Forest...")
                progress_bar.progress(90)

                rf_res = predict_live(
                    symbol=symbol, source=data_source,
                    api_key=api_key, save_to_db=not skip_mongo,
                    skip_hdfs=skip_hdfs
                )
                progress_bar.progress(100)
                status_text.text("Done!")

                hist_df = get_historical_for_chart(symbol=symbol, api_key=api_key, source=data_source)
                if hist_df is not None and not hist_df.empty:
                    try:
                        hist_df = compute_live_features(hist_df)
                    except Exception as he:
                        logger.warning(f"Could not compute features on hist_df: {he}")
                st.session_state['rf_result']   = rf_res
                st.session_state['rf_hist_df']  = hist_df
                st.session_state['active_model'] = 'rf'
                if rf_res.get('spark_ui_url'):
                    st.session_state['spark_ui_url'] = rf_res.get('spark_ui_url')

            except Exception as e:
                st.error(f"RF prediction failed: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    # ── LSTM Button clicked ───────────────────────────────────────────────────
    if lstm_button and lstm_ready:
        for k in ['rf_result', 'rf_hist_df', 'lstm_result', 'lstm_backtest', 'lstm_accuracy', 'active_model']:
            st.session_state.pop(k, None)

        with st.spinner(f"Running LSTM prediction for {symbol} (60-day sequence)..."):
            try:
                lstm_res = predict_lstm_live(symbol=symbol, source=data_source, api_key=api_key)
                hist_df  = get_historical_for_chart(symbol=symbol, api_key=api_key, source=data_source, period="1y")
                backtest_df = pd.DataFrame()
                backtest_acc = 0.0
                if hist_df is not None and not hist_df.empty:
                    try:
                        hist_df = compute_live_features(hist_df)
                        # Run sequential backtesting
                        backtest_df, backtest_acc = run_lstm_backtest(hist_df, symbol)
                    except Exception as he:
                        logger.warning(f"Could not compute features or backtest: {he}")
                st.session_state['lstm_result']  = lstm_res
                st.session_state['rf_hist_df']   = hist_df
                st.session_state['lstm_backtest'] = backtest_df
                st.session_state['lstm_accuracy'] = backtest_acc
                st.session_state['active_model'] = 'lstm'
            except Exception as e:
                st.error(f"LSTM prediction failed: {e}")

    # ── Display results ───────────────────────────────────────────────────────
    active_model = st.session_state.get('active_model')

    # ── RF Results ────────────────────────────────────────────────────────────
    if active_model == 'rf' and 'rf_result' in st.session_state:
        result  = st.session_state['rf_result']
        hist_df = st.session_state.get('rf_hist_df')

        # Pipeline status
        st.markdown("### Pipeline Status")
        cs1, cs2, cs3, cs4 = st.columns(4)
        with cs1:
            st.metric("Data Source", result['source'].upper(),
                      delta="Live API" if result['source'] in ['alpha_vantage','yfinance'] else None)
        with cs2:
            st.metric("CSV Dump", "Saved" if result.get('local_csv_path') else "Failed")
            if result.get('local_csv_path'):
                st.caption(f"File: {Path(result['local_csv_path']).name}")
        with cs3:
            if result.get('hdfs_uploaded'):
                st.metric("HDFS Upload", "Uploaded")
                st.caption("Path: /stock_data/live_api_dumps/")
            elif skip_hdfs:
                st.metric("HDFS Upload", "Skipped")
            else:
                st.metric("HDFS Upload", "Failed")
        with cs4:
            mongo_status = "Saved" if result.get('mongo_id') else ("Skipped" if skip_mongo else "Failed")
            st.metric("MongoDB", mongo_status)
            if result.get('mongo_id'):
                st.caption(f"ID: {result['mongo_id'][:12]}...")

        # Spark UI Availability Panel
        spark_ui_url = st.session_state.get('spark_ui_url') or result.get('spark_ui_url')
        if spark_ui_url:
            st.markdown("""
            <div style="background-color: #0d2a1d; border-left: 5px solid #10b981; padding: 12px 16px; border-radius: 4px; color: #34d399; font-weight: 600; font-size: 0.95rem; margin-top: 15px; margin-bottom: 20px;">
            Spark UI Available
            </div>
            """, unsafe_allow_html=True)
            
            col_ui1, col_ui2, col_ui3, col_ui4 = st.columns(4)
            with col_ui1:
                st.markdown(f"**[Overview]({spark_ui_url})**")
                st.caption("Main dashboard")
                st.caption("Spark UI will remain available for 5 minutes after processing completes")
                
                from urllib.parse import urlparse
                try:
                    parsed_url = urlparse(spark_ui_url)
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                except Exception:
                    base_url = spark_ui_url
                st.caption(f"Base URL: [{base_url}]({spark_ui_url})")
                
            with col_ui2:
                st.markdown(f"**[Jobs]({spark_ui_url}/jobs/)**")
                st.caption("All Spark jobs")
                
            with col_ui3:
                st.markdown(f"**[Stages]({spark_ui_url}/stages/)**")
                st.caption("Stage execution")
                
            with col_ui4:
                st.markdown(f"**[Storage]({spark_ui_url}/storage/)**")
                st.caption("RDD storage")
                
            st.markdown("")
            with st.expander("More Spark UI Pages"):
                col_more1, col_more2 = st.columns(2)
                with col_more1:
                    st.markdown("Monitoring:")
                    st.markdown(f"- [Environment]({spark_ui_url}/environment/) - Spark configuration")
                    st.markdown(f"- [Executors]({spark_ui_url}/executors/) - Executor metrics")
                    st.markdown(f"- [SQL]({spark_ui_url}/SQL/) - SQL query execution")
                with col_more2:
                    st.markdown("Analysis:")
                    st.markdown(f"- [Stages]({spark_ui_url}/stages/) - View completed stages")
                    st.markdown(f"- [Jobs]({spark_ui_url}/jobs/) - Click any job to see DAG")
                    st.markdown(f"- [Storage]({spark_ui_url}/storage/) - Cached data")

        st.markdown("---")

        # Quote
        st.markdown("### Latest Quote Data")
        q = result["quote"]
        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        render_metric("Open",   f"${q['open']:.2f}",   qc1)
        render_metric("High",   f"${q['high']:.2f}",   qc2)
        render_metric("Low",    f"${q['low']:.2f}",    qc3)
        render_metric("Close",  f"${q['close']:.2f}",  qc4)
        render_metric("Volume", f"{q['volume']:,}",     qc5)

        st.markdown("")

        # Prediction
        st.markdown("### Prediction Result  <span style='font-size:0.75rem;color:#94a3b8;'>— Random Forest</span>", unsafe_allow_html=True)
        render_prediction(result["prediction"], result["confidence"])
        st.markdown("")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("DOWN Probability", f"{result['probabilities']['DOWN']:.2f}%")
        pc2.metric("UP Probability",   f"{result['probabilities']['UP']:.2f}%")
        pc3.metric("Confidence", f"{result['confidence']:.2f}%",
                   delta="High" if result['confidence'] > 80 else ("Medium" if result['confidence'] > 60 else "Low"))
        st.markdown("")
        st.caption(
            f"Latest Trading Day: **{result['latest_date']}** | "
            f"Model: **{Path(result['model_file']).name}** | "
            f"Features: **{result['feature_count']}** | "
            f"Timestamp: **{result['timestamp'][:19]}**"
        )

        st.markdown("---")

        # Charts
        if hist_df is not None and not hist_df.empty:
            st.markdown("### Live Market Visualizations")
            st.markdown("#### Price Action & Volume Analysis")
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                st.plotly_chart(make_candlestick(hist_df, f"{symbol} — Candlestick Chart"),
                                use_container_width=True, key="rf_candle")
            with cc2:
                st.plotly_chart(create_volume_chart(hist_df, symbol),
                                use_container_width=True, key="rf_volume")
            st.markdown("")
            st.markdown("#### Price Trend Analysis")
            st.plotly_chart(make_close_chart(hist_df, f"{symbol} — Close Price Trend"),
                            use_container_width=True, key="rf_close")
            st.markdown("")
            st.markdown("#### Technical Indicator Correlations")
            st.plotly_chart(create_correlation_heatmap(hist_df, result.get("features"), window=50),
                            use_container_width=True, key="rf_heatmap")
            st.markdown("")
            st.markdown("#### Daily Returns Distribution")
            st.plotly_chart(create_returns_histogram(hist_df, symbol, window=100),
                            use_container_width=True, key="rf_returns")

        st.markdown("---")

        # RF Model info expander
        with st.expander("Feature Values Used for Prediction"):
            feat_df = pd.DataFrame([result["features"]])
            st.dataframe(feat_df.T.rename(columns={0: "Value"}).style.format("{:.6f}"),
                         use_container_width=True)

        with st.expander("Model Information"):
            mi1, mi2 = st.columns(2)
            with mi1:
                st.markdown("**Model Details:**")
                st.write(f"- Model File: `{Path(result['model_file']).name}`")
                st.write(f"- Model Type: Random Forest")
                st.write(f"- Features Used: {result['feature_count']}")
                if MODEL_METADATA_FILE.exists():
                    _m = json.loads(MODEL_METADATA_FILE.read_text())
                    st.write(f"- Training Accuracy: {_m['metrics']['accuracy']*100:.2f}%")
                    st.write(f"- ROC AUC: {_m['metrics']['roc_auc']:.4f}")
                    st.write(f"- F1 Score: {_m['metrics']['f1_score']:.4f}")
            with mi2:
                st.markdown("**Prediction Details:**")
                st.write(f"- Symbol: {result['symbol']}")
                st.write(f"- Prediction: {result['prediction']}")
                st.write(f"- Confidence: {result['confidence']:.2f}%")
                st.write(f"- Data Source: {result['source']}")
                st.write(f"- Latest Date: {result['latest_date']}")
                st.write(f"- Timestamp: {result['timestamp'][:19]}")

        with st.expander("Data Storage Information"):
            ds1, ds2 = st.columns(2)
            with ds1:
                st.markdown("**Local Storage:**")
                if result.get('local_csv_path'):
                    st.success(f"CSV saved: `{Path(result['local_csv_path']).name}`")
                else:
                    st.error("CSV save failed")
            with ds2:
                st.markdown("**HDFS Storage:**")
                if result.get('hdfs_uploaded'):
                    st.success("Uploaded to HDFS")
                    st.caption(f"Path: {result.get('hdfs_path','N/A')}")
                elif skip_hdfs:
                    st.info("Skipped")
                else:
                    st.warning("Failed (non-critical)")
            st.markdown("**MongoDB:**")
            if result.get('mongo_id'):
                st.success(f"Saved — ID: {result['mongo_id'][:16]}...")
            elif skip_mongo:
                st.info("Skipped")
            else:
                st.warning("Failed (non-critical)")

        with st.expander("Recent Prediction History (MongoDB)"):
            try:
                recent = fetch_recent_predictions(limit=15)
                if recent:
                    hdf = pd.DataFrame(recent)
                    dcols = ['symbol','prediction','confidence','timestamp','source','model']
                    if all(c in hdf.columns for c in dcols):
                        hdf = hdf[dcols]
                        hdf.columns = ['Symbol','Prediction','Confidence (%)','Timestamp','Source','Model']
                        hdf['Timestamp'] = pd.to_datetime(hdf['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        st.dataframe(hdf, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(pd.DataFrame(recent), use_container_width=True)
                else:
                    st.info("No records found or MongoDB unavailable.")
            except Exception as e:
                st.warning(f"Could not fetch history: {e}")

    # ── LSTM Results ──────────────────────────────────────────────────────────
    elif active_model == 'lstm' and 'lstm_result' in st.session_state:
        lstm_res = st.session_state['lstm_result']
        hist_df  = st.session_state.get('rf_hist_df')

        # Spark UI Availability Panel
        spark_ui_url = st.session_state.get('spark_ui_url') or lstm_res.get('spark_ui_url')
        if spark_ui_url:
            st.markdown("""
            <div style="background-color: #0d2a1d; border-left: 5px solid #10b981; padding: 12px 16px; border-radius: 4px; color: #34d399; font-weight: 600; font-size: 0.95rem; margin-top: 15px; margin-bottom: 20px;">
            Spark UI Available
            </div>
            """, unsafe_allow_html=True)
            
            col_ui1, col_ui2, col_ui3, col_ui4 = st.columns(4)
            with col_ui1:
                st.markdown(f"**[Overview]({spark_ui_url})**")
                st.caption("Main dashboard")
                st.caption("Spark UI will remain available for 5 minutes after processing completes")
                
                from urllib.parse import urlparse
                try:
                    parsed_url = urlparse(spark_ui_url)
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                except Exception:
                    base_url = spark_ui_url
                st.caption(f"Base URL: [{base_url}]({spark_ui_url})")
                
            with col_ui2:
                st.markdown(f"**[Jobs]({spark_ui_url}/jobs/)**")
                st.caption("All Spark jobs")
                
            with col_ui3:
                st.markdown(f"**[Stages]({spark_ui_url}/stages/)**")
                st.caption("Stage execution")
                
            with col_ui4:
                st.markdown(f"**[Storage]({spark_ui_url}/storage/)**")
                st.caption("RDD storage")
                
            st.markdown("")
            with st.expander("More Spark UI Pages"):
                col_more1, col_more2 = st.columns(2)
                with col_more1:
                    st.markdown("Monitoring:")
                    st.markdown(f"- [Environment]({spark_ui_url}/environment/) - Spark configuration")
                    st.markdown(f"- [Executors]({spark_ui_url}/executors/) - Executor metrics")
                    st.markdown(f"- [SQL]({spark_ui_url}/SQL/) - SQL query execution")
                with col_more2:
                    st.markdown("Analysis:")
                    st.markdown(f"- [Stages]({spark_ui_url}/stages/) - View completed stages")
                    st.markdown(f"- [Jobs]({spark_ui_url}/jobs/) - Click any job to see DAG")
                    st.markdown(f"- [Storage]({spark_ui_url}/storage/) - Cached data")
            
            st.markdown("---")

        # Quote (from hist_df if available)
        if hist_df is not None and not hist_df.empty:
            latest_row = hist_df.iloc[-1]
            st.markdown("### Latest Quote Data")
            qc1, qc2, qc3, qc4, qc5 = st.columns(5)
            render_metric("Open",   f"${float(latest_row['Open']):.2f}",   qc1)
            render_metric("High",   f"${float(latest_row['High']):.2f}",   qc2)
            render_metric("Low",    f"${float(latest_row['Low']):.2f}",    qc3)
            render_metric("Close",  f"${float(latest_row['Close']):.2f}",  qc4)
            render_metric("Volume", f"{int(latest_row['Volume']):,}",       qc5)
            st.markdown("")

        # Prediction
        st.markdown("### Prediction Result  <span style='font-size:0.75rem;color:#94a3b8;'>— LSTM</span>", unsafe_allow_html=True)
        render_prediction(lstm_res["prediction"], lstm_res["confidence"])
        st.markdown("")
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("DOWN Probability", f"{lstm_res['probabilities']['DOWN']:.2f}%")
        lc2.metric("UP Probability",   f"{lstm_res['probabilities']['UP']:.2f}%")
        lc3.metric("Confidence", f"{lstm_res['confidence']:.2f}%",
                   delta="High" if lstm_res['confidence'] > 80 else ("Medium" if lstm_res['confidence'] > 60 else "Low"))
        st.markdown("")
        st.caption(
            f"Latest Trading Day: **{lstm_res['latest_date']}** | "
            f"Model: **LSTM** | "
            f"Sequence: **{lstm_res['sequence_length']} days** | "
            f"Source: **{lstm_res['source']}**"
        )

        st.markdown("---")

        # True LSTM Stock-Specific Visualizations
        if hist_df is not None and not hist_df.empty:
            st.markdown("### LSTM Neural Network Visualizations")
            
            # Forecast chart
            fig_forecast = create_lstm_forecast_chart(hist_df, lstm_res)
            st.plotly_chart(fig_forecast, use_container_width=True, key="lstm_forecast_chart_plotly")
            
            # Backtest section
            backtest_df = st.session_state.get('lstm_backtest', pd.DataFrame())
            backtest_acc = st.session_state.get('lstm_accuracy', 0.0)
            
            if not backtest_df.empty:
                st.markdown("---")
                # Visual metric cards for backtest performance
                st.markdown("### LSTM Backtesting Report (Recent 30 Days)")
                
                btc1, btc2, btc3 = st.columns(3)
                btc1.metric("Sequential Backtest Accuracy", f"{backtest_acc:.1f}%")
                btc2.metric("Total Signals Tested", f"{len(backtest_df)} days")
                correct_signals = backtest_df["Correct"].sum()
                btc3.metric("Correct Directional Signals", f"{correct_signals} / {len(backtest_df)}")
                
                fig_backtest = create_lstm_backtest_chart(backtest_df, symbol, backtest_acc)
                st.plotly_chart(fig_backtest, use_container_width=True, key="lstm_backtest_chart_plotly")
                
                with st.expander("Detailed Daily Backtest Logs", expanded=False):
                    st.dataframe(
                        backtest_df[["Date", "Close", "Prediction", "Actual", "Correct"]].style.map(
                            lambda x: "background-color: rgba(0, 200, 83, 0.15); color: #00c853;" if x == "UP" or x is True else "background-color: rgba(255, 23, 68, 0.15); color: #ff1744;",
                            subset=["Prediction", "Actual", "Correct"]
                        ),
                        use_container_width=True
                    )
            else:
                st.info("LSTM Backtest: To see historical backtesting metrics, make sure you fetched enough historical data.")
                
        st.markdown("---")

        # Global LSTM training history (Epoch vs Loss/Accuracy)
        st.markdown("### Global LSTM Model Training Performance (Model-Wide)")
        st.caption(
            "These training history curves reflect the global performance of the trained neural network model "
            "across the historical dataset epochs. They represent the shared underlying model structure."
        )
        
        hist_tab1, hist_tab2 = st.tabs(["Loss History", "Accuracy History"])
        with hist_tab1:
            fig_loss = create_lstm_metric_chart("loss")
            st.plotly_chart(fig_loss, use_container_width=True, key="lstm_global_loss_history_plotly")
        with hist_tab2:
            fig_acc = create_lstm_metric_chart("accuracy")
            st.plotly_chart(fig_acc, use_container_width=True, key="lstm_global_acc_history_plotly")
            
        st.markdown("---")

        # LSTM Model info expander
        with st.expander("Model Information"):
            mi1, mi2 = st.columns(2)
            with mi1:
                st.markdown("**Model Details:**")
                st.write("- Model File: `lstm_model.keras`")
                st.write("- Model Type: LSTM (Long Short-Term Memory)")
                st.write(f"- Sequence Length: {lstm_res['sequence_length']} days")
                st.write("- Features: 21 technical indicators per day")
                lstm_meta_p = PROJECT_ROOT / "models" / "lstm_metadata.json"
                if lstm_meta_p.exists():
                    lm = json.loads(lstm_meta_p.read_text())
                    st.write(f"- Training Accuracy: {lm['metrics']['accuracy']*100:.2f}%")
                    st.write(f"- ROC AUC: {lm['metrics']['roc_auc']:.4f}")
                    st.write(f"- F1 Score: {lm['metrics']['f1_score']:.4f}")
                    arch = lm.get("architecture", {})
                    if arch.get("layers"):
                        st.markdown("**Architecture:**")
                        for layer in arch["layers"]:
                            st.write(f"  - {layer}")
            with mi2:
                st.markdown("**Prediction Details:**")
                st.write(f"- Symbol: {lstm_res['symbol']}")
                st.write(f"- Prediction: {lstm_res['prediction']}")
                st.write(f"- Confidence: {lstm_res['confidence']:.2f}%")
                st.write(f"- DOWN Probability: {lstm_res['probabilities']['DOWN']:.2f}%")
                st.write(f"- UP Probability: {lstm_res['probabilities']['UP']:.2f}%")
                st.write(f"- Data Source: {lstm_res['source']}")
                st.write(f"- Latest Date: {lstm_res['latest_date']}")

        with st.expander("Recent Prediction History (MongoDB)"):
            try:
                recent = fetch_recent_predictions(limit=15)
                if recent:
                    hdf = pd.DataFrame(recent)
                    dcols = ['symbol','prediction','confidence','timestamp','source','model']
                    if all(c in hdf.columns for c in dcols):
                        hdf = hdf[dcols]
                        hdf.columns = ['Symbol','Prediction','Confidence (%)','Timestamp','Source','Model']
                        hdf['Timestamp'] = pd.to_datetime(hdf['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        st.dataframe(hdf, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(pd.DataFrame(recent), use_container_width=True)
                else:
                    st.info("No records found or MongoDB unavailable.")
            except Exception as e:
                st.warning(f"Could not fetch history: {e}")

    else:
        # No prediction yet — show instructions
        st.info("Click **Predict with Random Forest** or **Predict with LSTM** to get a prediction.")
        st.markdown("")
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("#### Random Forest")
            st.markdown("""
- Uses **21 technical indicators** (snapshot)
- Fast prediction
- Works with any stock in dataset
- Shows feature importance
- Available in Live & Manual modes
            """)
        with col_info2:
            st.markdown("#### LSTM")
            st.markdown(f"""
- Uses **60-day sequence** of historical data
- Captures temporal patterns & trends
- {"Model trained and ready" if lstm_ready else "Not trained yet — run `python scripts/train_lstm_model.py`"}
- Shows training curves (loss & accuracy)
- Live mode only
            """)

# ---------------------------------------------------------------------------
# MANUAL MODE
# ---------------------------------------------------------------------------

elif mode == "Manual Input":
    st.markdown(f"### Manual Input — {selected_stock}")
    st.markdown("Enter OHLCV values and computed technical indicators:")
    st.caption("Using **Random Forest** — manual mode provides a single snapshot, which is exactly what RF needs.")

    col1, col2, col3 = st.columns(3)
    with col1:
        open_p = st.number_input("Open Price ($)", value=150.0, step=0.01, format="%.2f")
        high_p = st.number_input("High Price ($)", value=155.0, step=0.01, format="%.2f")
        low_p = st.number_input("Low Price ($)", value=148.0, step=0.01, format="%.2f")
        close_p = st.number_input("Close Price ($)", value=153.0, step=0.01, format="%.2f")
        volume = st.number_input("Volume", value=50000000, step=100000)
        ma5 = st.number_input("MA5", value=152.0, step=0.01, format="%.2f")
    with col2:
        ma10 = st.number_input("MA10", value=151.0, step=0.01, format="%.2f")
        ma20 = st.number_input("MA20", value=150.0, step=0.01, format="%.2f")
        daily_ret = st.number_input("Daily Returns (%)", value=0.5, step=0.01, format="%.4f")
        volatility = st.number_input("Volatility (H-L)", value=7.0, step=0.01, format="%.2f")
        price_chg = st.number_input("Price Change %", value=2.0, step=0.01, format="%.4f")
        lag1 = st.number_input("Lag 1 (Prev Close)", value=151.5, step=0.01, format="%.2f")
    with col3:
        lag3 = st.number_input("Lag 3 (3-day Close)", value=149.0, step=0.01, format="%.2f")
        rsi = st.number_input("RSI (14)", value=55.0, min_value=0.0, max_value=100.0, step=0.1)
        vol_chg = st.number_input("Volume Change %", value=5.0, step=0.01, format="%.4f")
        ema12 = st.number_input("EMA12", value=151.5, step=0.01, format="%.2f")
        bb_pos = st.number_input("Bollinger Position (0-1)", value=0.6, min_value=0.0, max_value=1.5, step=0.01)
        macd = st.number_input("MACD", value=0.5, step=0.01, format="%.4f")
        macd_signal = st.number_input("MACD Signal", value=0.4, step=0.01, format="%.4f")
        macd_hist = st.number_input("MACD Histogram", value=0.1, step=0.01, format="%.4f")
        bb_width = st.number_input("Bollinger Width (%)", value=4.0, step=0.01, format="%.4f")
        weekly_momentum = st.number_input("Weekly Momentum (%)", value=1.5, step=0.01, format="%.4f")
        avg_volume_trend = st.number_input("Average Volume Trend", value=1.0, step=0.01, format="%.4f")
        avg_5d_volume_trend = st.number_input("Average 5-Day Volume Trend", value=1.0, step=0.01, format="%.4f")
        trend_strength = st.number_input("Trend Strength (%)", value=0.8, step=0.01, format="%.4f")
        rolling_std_returns = st.number_input("Rolling Std Returns", value=1.2, step=0.01, format="%.4f")

    st.markdown("")

    if st.button("Predict", use_container_width=True, type="primary"):
        features = {
            "Open": open_p, "High": high_p, "Low": low_p, "Close": close_p,
            "Volume": volume, "MA5": ma5, "MA10": ma10, "MA20": ma20,
            "Daily_Returns": daily_ret, "Volatility": volatility,
            "Price_Change_Pct": price_chg, "Lag_1": lag1, "Lag_3": lag3,
            "RSI": rsi, "EMA12": ema12, "MACD": macd, "MACD_Signal": macd_signal,
            "BB_Width": bb_width, "Weekly_Momentum": weekly_momentum,
            "Avg_5D_Volume_Trend": avg_5d_volume_trend,
            "Trend_Strength": trend_strength,
        }
        
        # Make prediction
        direction, confidence, probs = predict_direction(features)
        
        # Display prediction
        st.markdown("")
        render_prediction(direction, confidence)
        st.markdown("")
        st.caption(f"DOWN probability: {probs[0]*100:.1f}% | UP probability: {probs[1]*100:.1f}%")
        
        # Save to MongoDB
        try:
            prediction_label = "UP" if direction == "UP" else "DOWN"
            mongo_id = save_prediction(
                symbol=symbol,
                prediction=prediction_label,
                confidence=confidence,
                source="manual_input",
                model="Random Forest (Final)",
                meta={
                    "input_type": "manual",
                    "down_probability": float(probs[0]),
                    "up_probability": float(probs[1]),
                    "quote": {
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": volume,
                    },
                    "features": features,
                },
            )
            
            if mongo_id:
                st.success(f"Prediction saved to MongoDB (ID: {mongo_id[:12]}...)")
            else:
                st.info("MongoDB save skipped or unavailable")
                
        except Exception as e:
            st.warning(f"Could not save to MongoDB: {e}")
            st.caption("Prediction completed successfully, but database storage failed (non-critical)")
        
        # --- Spark Processing for Manual Input ---
        st.markdown("---")
        st.markdown("### Spark Big Data Processing")
        
        with st.spinner("Running Spark distributed processing on training data..."):
            try:
                import threading
                from spark_processing import (
                    build_spark, read_input_data, validate_columns,
                    clean_data, generate_features as spark_generate_features,
                    save_output, compute_and_save_summary, visualize_dag,
                    demonstrate_rdd_operations, REQUIRED_COLUMNS,
                )
                
                # Build Spark session with optimized partitions for faster processing
                spark_session = build_spark(app_name="StockManualPrediction", use_hdfs=False)
                
                # CRITICAL: Reduce partitions from default 200 → 4 for fast processing
                spark_session.conf.set("spark.sql.shuffle.partitions", "4")
                spark_session.conf.set("spark.default.parallelism", "4")
                
                spark_ui_url = spark_session.sparkContext.uiWebUrl
                if spark_ui_url:
                    st.session_state['spark_ui_url'] = spark_ui_url
                
                # Read training data
                input_path = str(PROJECT_ROOT / "data" / "final_featured_data.csv")
                df = read_input_data(spark_session, input_path)
                
                # Validate columns
                validate_columns(df, REQUIRED_COLUMNS)
                
                # Clean data
                df = clean_data(df)
                
                # Generate Spark features (window functions, rolling calcs)
                df = spark_generate_features(df)
                
                # Cache DataFrame (visible in Spark UI Storage tab)
                df.cache()
                row_count = df.count()
                
                # Visualize DAG
                visualize_dag(df, str(PROJECT_ROOT / "spark_visualizations"))
                
                # Demonstrate RDD operations
                demonstrate_rdd_operations(spark_session, df, str(PROJECT_ROOT / "spark_visualizations"))
                
                # Save processed output
                save_output(df, str(PROJECT_ROOT / "spark_output"))
                
                # Compute and save summary statistics
                compute_and_save_summary(df, str(PROJECT_ROOT / "spark_summary"))
                
                # Show Spark UI info
                if spark_ui_url:
                    st.success(f"**Spark UI Available — [Open Spark UI]({spark_ui_url})**")
                    
                    col_ui1, col_ui2, col_ui3, col_ui4 = st.columns(4)
                    with col_ui1:
                        st.markdown(f"**[Overview]({spark_ui_url})**")
                        st.caption("Main dashboard")
                    with col_ui2:
                        st.markdown(f"**[Jobs]({spark_ui_url}/jobs/)**")
                        st.caption("All Spark jobs")
                    with col_ui3:
                        st.markdown(f"**[Stages]({spark_ui_url}/stages/)**")
                        st.caption("Stage execution")
                    with col_ui4:
                        st.markdown(f"**[Storage]({spark_ui_url}/storage/)**")
                        st.caption("Cached data")
                    
                    st.caption(f"Spark UI will remain available **until app restarts** at: {spark_ui_url}")
                    st.caption("Now run **Live Predict** — new Spark jobs will appear in the same UI!")
                    
                    with st.expander("More Spark UI Pages"):
                        col_more1, col_more2 = st.columns(2)
                        with col_more1:
                            st.markdown("**Monitoring:**")
                            st.markdown(f"- [Environment]({spark_ui_url}/environment/) - Spark configuration")
                            st.markdown(f"- [Executors]({spark_ui_url}/executors/) - Executor metrics")
                            st.markdown(f"- [SQL]({spark_ui_url}/SQL/) - SQL query execution")
                        with col_more2:
                            st.markdown("**Analysis:**")
                            st.markdown(f"- [Stages]({spark_ui_url}/stages/) - View completed stages")
                            st.markdown(f"- [Jobs]({spark_ui_url}/jobs/) - Click any job to see DAG")
                            st.markdown(f"- [Storage]({spark_ui_url}/storage/) - Cached data")
                        st.info("**Tip:** Click on any Job ID in the Jobs page to see its DAG visualization")
                
                # Show processing results
                st.markdown("")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Rows Processed", f"{row_count:,}")
                with col_s2:
                    st.metric("Spark Output", "Parquet Saved")
                with col_s3:
                    st.metric("DAG & RDD Demo", "Generated")
                
                # Keep Spark UI alive INDEFINITELY (no time limit)
                # This way when user switches to Live Predict, the Spark UI
                # is still active and new jobs will appear alongside Manual jobs
                # Spark will only stop when the Streamlit app is restarted
                
            except Exception as e:
                st.warning(f"Spark processing failed (non-critical): {e}")
                st.caption("Prediction completed successfully. Spark processing is optional for manual mode.")


