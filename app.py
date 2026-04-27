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
from model_config import FINAL_MODEL_FILE, MODEL_METADATA_FILE, get_expected_features
from predict_live import predict_live
from mongo_store import fetch_recent_predictions

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NASDAQ AI Predictor",
    page_icon="📈",
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

STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Netflix (NFLX)": "NFLX",
    "Intel (INTC)": "INTC",
    "AMD (AMD)": "AMD",
    "PayPal (PYPL)": "PYPL",
    "Adobe (ADBE)": "ADBE",
    "Salesforce (CRM)": "CRM",
    "Cisco (CSCO)": "CSCO",
    "Qualcomm (QCOM)": "QCOM",
    "Broadcom (AVGO)": "AVGO",
    "Texas Instruments (TXN)": "TXN",
    "Costco (COST)": "COST",
    "PepsiCo (PEP)": "PEP",
    "Starbucks (SBUX)": "SBUX",
}

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
    <h1>📈 NASDAQ AI Stock Predictor</h1>
    <p>Hybrid Machine Learning — Structured + Unstructured Data Analysis</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found. Run `python scripts/train_model.py` first to train the model.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    selected_stock = st.selectbox("Select Stock", list(STOCKS.keys()), index=0)
    symbol = STOCKS[selected_stock]

    st.markdown("---")
    mode = st.radio("Data Source", ["🔴 Live Data", "📝 Manual Input"], index=0)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    meta_path = MODEL_METADATA_FILE
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        st.metric("Accuracy", f"{meta['metrics']['accuracy']*100:.1f}%")
        st.metric("ROC AUC", f"{meta['metrics']['roc_auc']:.3f}")
        st.metric("F1 Score", f"{meta['metrics']['f1_score']:.3f}")
        st.caption(f"Model: {meta.get('selected_model', meta.get('model_type', 'RandomForest'))}")
        st.caption(f"Trained: {meta.get('trained_at', 'N/A')}")
    else:
        st.info("Train model to see metrics")

    st.markdown("---")
    st.markdown(
        "<p style='color:#64748b;font-size:0.75rem;text-align:center;'>"
        "Phase 2 · Final Random Forest Production<br>Final Year Project</p>",
        unsafe_allow_html=True,
    )

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
    cls = "pred-up" if direction == "UP" else "pred-down"
    color = "#00c853" if direction == "UP" else "#ff1744"
    arrow = "📈 STOCK WILL GO UP" if direction == "UP" else "📉 STOCK WILL GO DOWN"
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
# LIVE MODE
# ---------------------------------------------------------------------------

if mode == "🔴 Live Data":
    st.markdown(f"### 🔴 Live Data — {selected_stock}")

    with st.spinner(f"Fetching live data for {symbol}..."):
        try:
            from fetch_live_data import get_historical_for_chart
            api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            result = predict_live(symbol=symbol, source="auto", api_key=api_key)
            hist_df = get_historical_for_chart(symbol=symbol, api_key=api_key, source="auto")
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.info("💡 Ensure yfinance is installed or set ALPHA_VANTAGE_API_KEY env variable.")
            st.stop()

    # --- Quote Metrics ---
    latest_features = result["features"]
    q = {
        "open": latest_features["Open"],
        "high": latest_features["High"],
        "low": latest_features["Low"],
        "close": latest_features["Close"],
        "volume": int(latest_features["Volume"]),
    }
    c1, c2, c3, c4, c5 = st.columns(5)
    render_metric("Open", f"${q['open']:.2f}", c1)
    render_metric("High", f"${q['high']:.2f}", c2)
    render_metric("Low", f"${q['low']:.2f}", c3)
    render_metric("Close", f"${q['close']:.2f}", c4)
    render_metric("Volume", f"{q['volume']:,}", c5)

    st.markdown("")

    # --- Prediction ---
    direction = result["prediction"]
    confidence = result["confidence"]
    probs = [result["probabilities"]["DOWN"], result["probabilities"]["UP"]]
    render_prediction(direction, confidence)

    st.markdown("")
    st.caption(
        f"📅 Latest Trading Day: **{result['latest_date']}** | "
        f"Source: **{result['source']}** | MongoDB ID: **{result.get('mongo_id') or 'not saved'}**"
    )

    # --- Charts ---
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(make_candlestick(hist_df, f"{symbol} — Candlestick"), width="stretch")
    with col_r:
        st.plotly_chart(make_close_chart(hist_df, f"{symbol} — Close Price"), width="stretch")

    # --- Feature Details ---
    with st.expander("🔍 Feature Values Used for Prediction"):
        feat_df = pd.DataFrame([result["features"]])
        st.dataframe(feat_df.T.rename(columns={0: "Value"}).style.format("{:.4f}"))

    with st.expander("🗃️ Recent MongoDB Predictions"):
        recent = fetch_recent_predictions(limit=10)
        if recent:
            st.dataframe(pd.DataFrame(recent))
        else:
            st.info("No recent prediction records found or MongoDB unavailable.")


# ---------------------------------------------------------------------------
# MANUAL MODE
# ---------------------------------------------------------------------------

elif mode == "📝 Manual Input":
    st.markdown(f"### 📝 Manual Input — {selected_stock}")
    st.markdown("Enter OHLCV values and computed technical indicators:")

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

    if st.button("🚀 Predict", width="stretch", type="primary"):
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
        direction, confidence, probs = predict_direction(features)
        st.markdown("")
        render_prediction(direction, confidence)
        st.markdown("")
        st.caption(f"DOWN probability: {probs[0]*100:.1f}% | UP probability: {probs[1]*100:.1f}%")
