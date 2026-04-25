import streamlit as st
import joblib
import yfinance as yf
import plotly.graph_objects as go
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Stock Dashboard", layout="wide")

model = joblib.load("model.pkl")

API_KEY = "YOUR_API_KEY"  # 🔥 put your Alpha Vantage key

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;font-size:60px'>📈 Stock Dashboard</h1>", unsafe_allow_html=True)

# ---------------- MODE ----------------
mode = st.radio(
    "Choose Data Source",
    ["Manual", "Live (yfinance)", "Live (Alpha Vantage)"],
    horizontal=True
)

# ---------------- STOCK ----------------
stocks = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Google (GOOGL)": "GOOGL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN"
}

selected = st.selectbox("Select Stock", list(stocks.keys()))
symbol = stocks[selected]

# =========================================================
# ===================== MANUAL MODE ========================
# =========================================================
if mode == "Manual":

    st.markdown("## 📝 Manual Input")

    col1, col2 = st.columns(2)

    with col1:
        open_price = st.number_input("Open", value=100.0)
        low = st.number_input("Low", value=95.0)

    with col2:
        high = st.number_input("High", value=105.0)
        volume = st.number_input("Volume", value=1000000.0)

    if st.button("Predict"):
        pred = model.predict([[open_price, high, low, volume]])

        if pred[0] == 1:
            st.success("📈 STOCK WILL GO UP")
        else:
            st.error("📉 STOCK WILL GO DOWN")

# =========================================================
# ===================== YFINANCE MODE ======================
# =========================================================
elif mode == "Live (yfinance)":

    st.markdown("## 🔴 Live Data (yfinance)")

    data = yf.Ticker(symbol)
    hist = data.history(period="1mo")

    latest = hist.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("OPEN", round(latest["Open"],2))
    col2.metric("HIGH", round(latest["High"],2))
    col3.metric("LOW", round(latest["Low"],2))
    col4.metric("VOLUME", int(latest["Volume"]))

    if st.button("Predict"):
        pred = model.predict([[latest["Open"], latest["High"], latest["Low"], latest["Volume"]]])

        if pred[0] == 1:
            st.success("📈 STOCK WILL GO UP")
        else:
            st.error("📉 STOCK WILL GO DOWN")

    # graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'],
        mode='lines',
        name='Close'
    ))

    fig.update_layout(height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ================= ALPHA VANTAGE MODE =====================
# =========================================================
else:

    st.markdown("## 🔵 Live Data (Alpha Vantage API)")

    ts = TimeSeries(key=API_KEY, output_format='pandas')

    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        data = data.sort_index()
        data.columns = ["Open", "High", "Low", "Close", "Volume"]

    except:
        st.error("API Error (check key / limit reached)")
        st.stop()

    latest = data.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("OPEN", round(latest["Open"],2))
    col2.metric("HIGH", round(latest["High"],2))
    col3.metric("LOW", round(latest["Low"],2))
    col4.metric("VOLUME", int(latest["Volume"]))

    # prediction
    if st.button("Predict"):
        pred = model.predict([[latest["Open"], latest["High"], latest["Low"], latest["Volume"]]])
        if pred[0] == 1:
            st.success("📈 STOCK WILL GO UP")
        else:
            st.error("📉 STOCK WILL GO DOWN")

    # candlestick
    st.markdown("### 📊 Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    fig.update_layout(height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # forecast
    st.markdown("### 📈 7-Day Forecast")

    last_price = data['Close'].iloc[-1]
    future = []

    for i in range(7):
        last_price = last_price * (1 + 0.002)
        future.append(last_price)

    future_dates = pd.date_range(start=data.index[-1], periods=8)[1:]

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Actual"))
    fig2.add_trace(go.Scatter(x=future_dates, y=future, name="Forecast", line=dict(dash='dot')))

    fig2.update_layout(height=500, template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)
