# 📈 Hybrid NASDAQ Stock Market Prediction System

**AI-powered stock price direction prediction using structured + unstructured data, live APIs, and machine learning.**

> Final Year Major Project — Phase 1: ML Prediction Engine

---

## 🚀 How to Run the Project (Step-by-Step)

Follow these steps exactly in order to run the fully updated machine learning pipeline and launch the dashboard.

### Step 1: Install Dependencies
Open your terminal and run:
```bash
pip install -r requirements.txt
```

### Step 2: Setup API Key
1. Open the `.env` file in the root folder.
2. Replace `YOUR_KEY_HERE` with your Alpha Vantage API Key.
   *(Get a free key here: https://www.alphavantage.co/support/#api-key)*

### Step 3: Run the ML Pipeline (Data -> Features -> Model)
Run these commands one by one in the terminal. Wait for each to finish before starting the next.

1. **Merge the raw CSV data into one file:**
   ```bash
   python scripts/merge_stock_data.py
   ```
2. **Clean and preprocess the merged data:**
   ```bash
   python scripts/preprocessing.py
   ```
3. **Generate ML features (Moving Averages, RSI, Lag, Volatility, etc):**
   ```bash
   python scripts/feature_engineering.py
   ```
4. **Train the final Random Forest model:**
   ```bash
   python scripts/train_model.py
   ```

### Step 4: Launch the Dashboard
Once the model is successfully trained, start the premium UI:
```bash
streamlit run app.py
```

### Step 5: (Optional) Generate Candlestick Images
To generate bullish/bearish candlestick images for the future CNN phase:
```bash
python scripts/generate_candlestick_images.py
```

---

## 🗑️ Files That Are No Longer Used (Safe to Delete)

The following files are from the older basic version of the project. They have been entirely replaced by the new architecture and **are of no use anymore**. You can safely delete them:

* ❌ `model.py` *(Replaced by `scripts/train_model.py`)*
* ❌ `model.pkl` *(Replaced by `models/random_forest.pkl`)*
* ❌ `spark_processing.py` *(Old stub. Will recreate properly in Big Data Phase 3)*
* ❌ `dashboard/app.py` *(We moved the premium UI directly into `app.py` in the root folder)*

---

## 🏗️ New Project Architecture

```
stock_prediction_project/
│
├── data/                              # Generated datasets
│   ├── stock_market_dataset/          # Raw NASDAQ dataset
│   ├── merged_stock_data.csv          # Combined multi-stock data
│   ├── cleaned_stock_data.csv         # Preprocessed data
│   └── final_featured_data.csv        # Feature-engineered ML-ready data
│
├── scripts/                           # Pipeline modules
│   ├── merge_stock_data.py            
│   ├── preprocessing.py              
│   ├── feature_engineering.py        
│   ├── fetch_live_data.py            # Live data (Alpha Vantage + yfinance)
│   ├── train_model.py                
│   └── generate_candlestick_images.py 
│
├── models/                           # Saved model artifacts
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
│
├── images/                           # Candlestick chart images
│   ├── bullish/                      
│   └── bearish/                      
│
├── app.py                            # Streamlit prediction dashboard (Main entry point)
├── .env                              # API keys
├── requirements.txt
└── README.md
```

---

## 📊 Features Engineered

| Feature | Description |
|---------|-------------|
| MA5, MA10, MA20 | Simple Moving Averages (5, 10, 20 days) |
| EMA12 | Exponential Moving Average (12 days) |
| RSI | Relative Strength Index (14-day, Wilder's smoothing) |
| Daily Returns | Close-to-close percentage change |
| Volatility | Intraday range (High - Low) |
| Price Change % | Intraday return (Close - Open) / Open |
| Lag_1, Lag_3 | Previous day and 3-day-ago closing prices |
| Volume Change % | Day-over-day volume change |
| BB_Position | Position within Bollinger Bands (0-1 scale) |

**Target**: Binary classification — `1` (UP) if tomorrow's close > today's close, `0` (DOWN) otherwise.
