# 📈 Stock Market Prediction Dashboard

## Features
- Manual Stock Prediction
- Live Data using yfinance
- Live Data using Alpha Vantage API
- Candlestick Charts
- 7-Day Forecast
- Big Data Processing using Apache Spark

---

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- PySpark
- Plotly
- yfinance
- Alpha Vantage API

---

## Installation

Clone the repository:

git clone https://github.com/Asmetha0205/Stock-Market-Dashboard.git

Install dependencies:

pip install -r requirements.txt

---

## Run the App

streamlit run app.py

---

## Dataset

Download dataset from Kaggle:

https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset?utm_so

After downloading, place it inside:

data/

---

## Alpha Vantage API Key

1. Get your free API key from:
   https://www.alphavantage.co

2. Open `app.py`

3. Replace:

API_KEY = "YOUR_API_KEY"

with your actual key.

---

## Notes

- Dataset is not included due to large size
- Ensure Java 8 is installed for Spark
- Keep folder structure intact

---

## Project Structure

StudioProject/
│
├── app.py
├── model.py
├── spark_processing.py
├── model.pkl
├── requirements.txt

---

## Future Improvements
- Advanced ML models (LSTM)
- Multi-stock comparison
- Buy/Sell signals
