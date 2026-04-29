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
from fetch_live_data import get_historical_for_chart

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

# Get supported stocks from predict_live.py (production stocks with full pipeline support)
SUPPORTED_STOCKS_LIST = get_supported_stocks()

STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
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
# LIVE MODE - PRODUCTION PIPELINE
# ---------------------------------------------------------------------------

if mode == "🔴 Live Data":
    st.markdown(f"### 🔴 Live Prediction Pipeline — {selected_stock}")
    
    # Check if stock is supported for full pipeline
    if symbol not in SUPPORTED_STOCKS_LIST:
        st.warning(f"⚠️ {symbol} is not in the production pipeline. Supported stocks: {', '.join(SUPPORTED_STOCKS_LIST)}")
        st.info("Please select one of the supported stocks for full live prediction pipeline.")
        st.stop()
    
    st.markdown("---")
    
    # Prediction Controls
    col_btn, col_opts = st.columns([1, 2])
    
    with col_btn:
        predict_button = st.button(
            "🚀 Predict Live",
            type="primary",
            use_container_width=True,
            help="Run complete prediction pipeline: API → CSV → HDFS → ML → MongoDB"
        )
    
    with col_opts:
        col_src, col_hdfs, col_mongo = st.columns(3)
        with col_src:
            data_source = st.selectbox(
                "Data Source",
                ["auto", "alpha_vantage", "yfinance"],
                index=0,
                help="auto: tries Alpha Vantage first, falls back to Yahoo Finance"
            )
        with col_hdfs:
            skip_hdfs = st.checkbox("Skip HDFS", value=False, help="Skip HDFS upload (for testing)")
        with col_mongo:
            skip_mongo = st.checkbox("Skip MongoDB", value=False, help="Skip MongoDB save (for testing)")
    
    st.markdown("---")
    
    # Run prediction when button is clicked
    if predict_button or 'last_prediction' in st.session_state:
        
        if predict_button:
            # Clear previous prediction
            if 'last_prediction' in st.session_state:
                del st.session_state['last_prediction']
            
            # Run complete prediction pipeline
            with st.spinner(f"🔄 Running complete prediction pipeline for {symbol}..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Fetch live data
                    status_text.text("📡 Fetching live data from API...")
                    progress_bar.progress(20)
                    
                    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
                    
                    # Step 2: Run complete prediction pipeline
                    status_text.text("💾 Saving CSV and uploading to HDFS...")
                    progress_bar.progress(40)
                    
                    status_text.text("🔧 Generating features and validating schema...")
                    progress_bar.progress(60)
                    
                    status_text.text("🤖 Running Random Forest prediction...")
                    progress_bar.progress(80)
                    
                    # Call predict_live with all options
                    result = predict_live(
                        symbol=symbol,
                        source=data_source,
                        api_key=api_key,
                        save_to_db=not skip_mongo,
                        skip_hdfs=skip_hdfs
                    )
                    
                    status_text.text("✅ Prediction complete!")
                    progress_bar.progress(100)
                    
                    # Store result in session state
                    st.session_state['last_prediction'] = result
                    st.session_state['last_symbol'] = symbol
                    
                    # Fetch historical data for charts
                    hist_df = get_historical_for_chart(
                        symbol=symbol,
                        api_key=api_key,
                        source=data_source
                    )
                    st.session_state['hist_df'] = hist_df
                    
                    st.success("✅ Prediction pipeline completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.exception(e)
                    st.info("💡 Check that:\n- Internet connection is available\n- Model file exists\n- API keys are set (if using Alpha Vantage)")
                    st.stop()
                finally:
                    progress_bar.empty()
                    status_text.empty()
        
        # Display results if available
        if 'last_prediction' in st.session_state:
            result = st.session_state['last_prediction']
            hist_df = st.session_state.get('hist_df')
            
            # --- Pipeline Status ---
            st.markdown("### 📊 Pipeline Status")
            col_status1, col_status2, col_status3, col_status4 = st.columns(4)
            
            with col_status1:
                st.metric(
                    "Data Source",
                    result['source'].upper(),
                    delta="Live API" if result['source'] in ['alpha_vantage', 'yfinance'] else None
                )
            
            with col_status2:
                csv_status = "✅ Saved" if result.get('local_csv_path') else "❌ Failed"
                st.metric("CSV Dump", csv_status)
                if result.get('local_csv_path'):
                    st.caption(f"📁 {Path(result['local_csv_path']).name}")
                else:
                    st.caption("❌ CSV save failed")
            
            with col_status3:
                if result.get('hdfs_uploaded'):
                    st.metric("HDFS Upload", "✅ Uploaded")
                    st.caption("📂 /stock_data/live_api_dumps/")
                elif skip_hdfs:
                    st.metric("HDFS Upload", "⏭️ Skipped")
                    st.caption("Skipped by user")
                else:
                    st.metric("HDFS Upload", "⚠️ Failed")
                    st.caption("Check HDFS status below")
            
            with col_status4:
                mongo_status = "✅ Saved" if result.get('mongo_id') else ("⏭️ Skipped" if skip_mongo else "❌ Failed")
                st.metric("MongoDB", mongo_status)
                if result.get('mongo_id'):
                    st.caption(f"🗄️ ID: {result['mongo_id'][:12]}...")
                elif skip_mongo:
                    st.caption("Skipped by user")
                else:
                    st.caption("Check MongoDB status")
            
            st.markdown("---")
            
            # --- Quote Metrics ---
            st.markdown("### 💰 Latest Quote Data")
            q = result["quote"]
            c1, c2, c3, c4, c5 = st.columns(5)
            render_metric("Open", f"${q['open']:.2f}", c1)
            render_metric("High", f"${q['high']:.2f}", c2)
            render_metric("Low", f"${q['low']:.2f}", c3)
            render_metric("Close", f"${q['close']:.2f}", c4)
            render_metric("Volume", f"{q['volume']:,}", c5)
            
            st.markdown("")
            
            # --- Prediction Result ---
            st.markdown("### 🎯 Prediction Result")
            direction = result["prediction"]
            confidence = result["confidence"]
            render_prediction(direction, confidence)
            
            st.markdown("")
            
            # Detailed probabilities
            col_prob1, col_prob2, col_prob3 = st.columns(3)
            with col_prob1:
                st.metric(
                    "DOWN Probability",
                    f"{result['probabilities']['DOWN']:.2f}%",
                    delta=None
                )
            with col_prob2:
                st.metric(
                    "UP Probability",
                    f"{result['probabilities']['UP']:.2f}%",
                    delta=None
                )
            with col_prob3:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%",
                    delta="High" if confidence > 80 else ("Medium" if confidence > 60 else "Low")
                )
            
            st.markdown("")
            st.caption(
                f"📅 Latest Trading Day: **{result['latest_date']}** | "
                f"🤖 Model: **{Path(result['model_file']).name}** | "
                f"🔢 Features: **{result['feature_count']}** | "
                f"⏰ Timestamp: **{result['timestamp'][:19]}**"
            )
            
            st.markdown("---")
            
            # --- Charts ---
            if hist_df is not None and not hist_df.empty:
                st.markdown("### 📈 Historical Charts")
                col_l, col_r = st.columns(2)
                with col_l:
                    st.plotly_chart(
                        make_candlestick(hist_df, f"{symbol} — Candlestick Chart"),
                        use_container_width=True
                    )
                with col_r:
                    st.plotly_chart(
                        make_close_chart(hist_df, f"{symbol} — Close Price Trend"),
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            # --- Feature Details ---
            with st.expander("🔍 Feature Values Used for Prediction"):
                feat_df = pd.DataFrame([result["features"]])
                st.dataframe(
                    feat_df.T.rename(columns={0: "Value"}).style.format("{:.6f}"),
                    use_container_width=True
                )
            
            # --- Model Metadata ---
            with st.expander("🤖 Model Information"):
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    st.markdown("**Model Details:**")
                    st.write(f"- Model File: `{Path(result['model_file']).name}`")
                    st.write(f"- Model Type: Random Forest (Final)")
                    st.write(f"- Features Used: {result['feature_count']}")
                    
                    meta_path = MODEL_METADATA_FILE
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        st.write(f"- Training Accuracy: {meta['metrics']['accuracy']*100:.2f}%")
                        st.write(f"- ROC AUC Score: {meta['metrics']['roc_auc']:.4f}")
                        st.write(f"- F1 Score: {meta['metrics']['f1_score']:.4f}")
                
                with col_m2:
                    st.markdown("**Prediction Details:**")
                    st.write(f"- Symbol: {result['symbol']}")
                    st.write(f"- Prediction: {result['prediction']}")
                    st.write(f"- Confidence: {result['confidence']:.2f}%")
                    st.write(f"- Data Source: {result['source']}")
                    st.write(f"- Latest Date: {result['latest_date']}")
                    st.write(f"- Timestamp: {result['timestamp'][:19]}")
            
            # --- Storage Information ---
            with st.expander("💾 Data Storage Information"):
                col_s1, col_s2 = st.columns(2)
                
                with col_s1:
                    st.markdown("**Local Storage:**")
                    if result.get('local_csv_path'):
                        st.success(f"✅ CSV saved: `{Path(result['local_csv_path']).name}`")
                        st.caption(f"Full path: {result['local_csv_path']}")
                        
                        # Show file size if available
                        try:
                            csv_path = Path(result['local_csv_path'])
                            if csv_path.exists():
                                file_size = csv_path.stat().st_size
                                st.caption(f"File size: {file_size:,} bytes")
                        except:
                            pass
                    else:
                        st.error("❌ CSV save failed")
                
                with col_s2:
                    st.markdown("**HDFS Storage:**")
                    if result.get('hdfs_uploaded'):
                        st.success("✅ Uploaded to HDFS")
                        st.caption(f"Path: {result.get('hdfs_path', 'N/A')}")
                        st.caption("Verify: `hdfs dfs -ls /stock_data/live_api_dumps/`")
                    elif skip_hdfs:
                        st.info("⏭️ HDFS upload skipped by user")
                        st.caption("Enable HDFS upload by unchecking 'Skip HDFS'")
                    else:
                        st.warning("⚠️ HDFS upload failed (non-critical)")
                        st.caption("Prediction continues with local CSV")
                        
                        # HDFS troubleshooting
                        st.markdown("**HDFS Troubleshooting:**")
                        st.code("""
# Check if HDFS is running
jps

# Should show: NameNode, DataNode

# Start HDFS if not running
start-dfs.sh

# Create HDFS directory
hdfs dfs -mkdir -p /stock_data/live_api_dumps

# Verify directory
hdfs dfs -ls /stock_data/
                        """, language="bash")
                
                st.markdown("**MongoDB Storage:**")
                if result.get('mongo_id'):
                    st.success(f"✅ Prediction saved to MongoDB")
                    st.caption(f"Document ID: {result['mongo_id']}")
                    st.caption("Database: stock_predictor | Collection: predictions")
                elif skip_mongo:
                    st.info("⏭️ MongoDB save skipped by user")
                    st.caption("Enable MongoDB save by unchecking 'Skip MongoDB'")
                else:
                    st.warning("⚠️ MongoDB save failed (non-critical)")
                    st.caption("Prediction completed without database storage")
                    
                    # MongoDB troubleshooting
                    st.markdown("**MongoDB Troubleshooting:**")
                    st.code("""
# Check if MongoDB is running
ps aux | grep mongod

# Start MongoDB
mongod

# Or with config
mongod --config /path/to/mongod.conf
                    """, language="bash")
            
            # --- Recent Predictions History ---
            with st.expander("🗃️ Recent Prediction History (MongoDB)"):
                try:
                    recent = fetch_recent_predictions(limit=15)
                    if recent:
                        # Format the data
                        history_df = pd.DataFrame(recent)
                        
                        # Select and rename columns
                        display_cols = ['symbol', 'prediction', 'confidence', 'timestamp', 'source', 'model']
                        if all(col in history_df.columns for col in display_cols):
                            history_df = history_df[display_cols]
                            history_df.columns = ['Symbol', 'Prediction', 'Confidence (%)', 'Timestamp', 'Source', 'Model']
                            
                            # Format timestamp
                            if 'Timestamp' in history_df.columns:
                                history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                            
                            st.dataframe(
                                history_df,
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.dataframe(pd.DataFrame(recent), use_container_width=True)
                    else:
                        st.info("No recent prediction records found or MongoDB unavailable.")
                except Exception as e:
                    st.warning(f"Could not fetch prediction history: {e}")
    
    else:
        # Show instructions when no prediction has been made
        st.info("👆 Click **'Predict Live'** button to run the complete prediction pipeline")
        
        st.markdown("### 📋 What happens when you click Predict:")
        st.markdown("""
        1. **📡 Fetch Live Data** - Get latest stock data from Alpha Vantage or Yahoo Finance
        2. **💾 Save CSV** - Save raw data to `live_api_dumps/` directory
        3. **📂 Upload to HDFS** - Upload CSV to Hadoop HDFS (if not skipped)
        4. **🔧 Feature Engineering** - Compute 21 technical indicators
        5. **✅ Validate Schema** - Ensure features match training pipeline
        6. **🤖 ML Prediction** - Run Random Forest model prediction
        7. **📊 Calculate Confidence** - Compute prediction confidence score
        8. **🗄️ Save to MongoDB** - Store prediction result (if not skipped)
        9. **📈 Display Results** - Show prediction, charts, and metadata
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Supported Stocks for Full Pipeline:")
        cols = st.columns(4)
        for i, stock in enumerate(SUPPORTED_STOCKS_LIST):
            cols[i % 4].markdown(f"- **{stock}**")
        
        
        col_req1, col_req2 = st.columns(2)
        

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
        
        # Make prediction
        direction, confidence, probs = predict_direction(features)
        
        # Display prediction
        st.markdown("")
        render_prediction(direction, confidence)
        st.markdown("")
        st.caption(f"DOWN probability: {probs[0]*100:.1f}% | UP probability: {probs[1]*100:.1f}%")
        
        # Save to MongoDB
        try:
            prediction_label = "UP 📈" if direction == "UP" else "DOWN 📉"
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
                st.success(f"✅ Prediction saved to MongoDB (ID: {mongo_id[:12]}...)")
            else:
                st.info("ℹ️ MongoDB save skipped or unavailable")
                
        except Exception as e:
            st.warning(f"⚠️ Could not save to MongoDB: {e}")
            st.caption("Prediction completed successfully, but database storage failed (non-critical)")
