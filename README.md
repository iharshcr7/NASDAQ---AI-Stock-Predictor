# NASDAQ AI Stock Predictor

A Python-based stock direction prediction project with a Streamlit dashboard, live market data fetching, Random Forest and LSTM models, optional MongoDB prediction history, and optional Spark/HDFS processing for big-data workflow demonstration.

The app predicts whether a selected stock is likely to move **UP** or **DOWN** using engineered market features such as moving averages, RSI, MACD, volatility, lag values, volume trends, and momentum indicators.

> Note: This project is for learning, experimentation, and demonstration. It is not financial advice.

## Main Features

- Interactive Streamlit dashboard in `app.py`
- Live stock data from Alpha Vantage with yfinance fallback
- Manual feature-entry prediction mode
- Production Random Forest model using 21 engineered technical indicators
- LSTM model using 60-day time sequences
- Stock symbol search and company-name mapping
- MongoDB storage for prediction history
- Spark processing for local/HDFS data pipelines
- Optional HDFS upload for datasets, live API dumps, and candlestick images
- Optional fusion model that combines structured stock features with CNN image features

## Tech Stack

Core Python:

- `pandas`, `numpy`
- `scikit-learn`, `joblib`
- `tensorflow==2.15.0`
- `pyspark==3.5.8`

Dashboard and visualization:

- `streamlit`
- `plotly`
- `matplotlib`
- `mplfinance`

Live data and configuration:

- `yfinance`
- `requests`
- `alpha_vantage`
- `python-dotenv`

Storage:

- `pymongo`
- Optional local MongoDB server
- Optional Hadoop HDFS

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- .env
|-- .gitignore
|-- data/
|   |-- stock_market_dataset/
|   |   |-- stocks/
|   |   |-- etfs/
|   |   `-- symbols_valid_meta.csv
|   |-- merged_stock_data.csv
|   |-- cleaned_stock_data.csv
|   |-- final_featured_data.csv
|   |-- cnn_features.csv
|   |-- fusion_features.csv
|   `-- image_labels.csv
|-- models/
|   |-- final_random_forest.pkl
|   |-- model_metadata.json
|   |-- lstm_model.keras
|   |-- lstm_scaler.pkl
|   |-- lstm_metadata.json
|   |-- lstm_history.json
|   |-- fusion_random_forest.pkl
|   |-- fusion_metadata.json
|   |-- random_forest.pkl
|   `-- scaler.pkl
|-- scripts/
|   |-- merge_stock_data.py
|   |-- preprocessing.py
|   |-- feature_engineering.py
|   |-- final_random_forest_model.py
|   |-- train_lstm_model.py
|   |-- predict_live.py
|   |-- predict_lstm.py
|   |-- fetch_live_data.py
|   |-- mongo_store.py
|   |-- model_config.py
|   |-- symbol_mapper.py
|   |-- spark_processing.py
|   |-- spark_live_processing.py
|   |-- hdfs_upload.py
|   |-- generate_candlestick_images.py
|   |-- generate_image_labels.py
|   |-- upload_images_to_hdfs.py
|   |-- fusion_feature_engineering.py
|   `-- train_fusion_model.py
|-- live_api_dumps/
|-- metadata/
|-- spark_output/
|-- spark_summary/
|-- spark_visualizations/
`-- test_spark_output.parquet/
```

## Current Local Artifacts

This checkout already contains generated data and model artifacts:

- `data/stock_market_dataset/stocks/`: 5,884 stock CSV files
- `data/stock_market_dataset/etfs/`: 2,165 ETF CSV files
- `models/final_random_forest.pkl`: main dashboard Random Forest model
- `models/lstm_model.keras`: trained LSTM model
- `models/fusion_random_forest.pkl`: trained fusion model
- `spark_output/`: processed Spark Parquet output
- `spark_summary/`: Spark summary CSV output
- `spark_visualizations/`: Spark DAG and RDD explanation files
- `live_api_dumps/`: saved live API CSV responses

## Environment Variables

The project loads environment variables from `.env`.

Required for Alpha Vantage live data:

```env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

Optional MongoDB settings:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=stock_predictor
MONGODB_COLLECTION_NAME=predictions
```

Optional Spark UI keep-alive:

```env
SPARK_UI_KEEPALIVE_SECONDS=300
```

Values:

- `300`: keep Spark UI alive for 5 minutes
- `0`: stop immediately
- `-1`: keep alive indefinitely during development

## Installation

Use Python 3.10 or 3.11 for best compatibility with TensorFlow 2.15.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## How to Run the Dashboard

From the project root:

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

The dashboard requires:

- `models/final_random_forest.pkl`
- `models/model_metadata.json`
- the `scripts/` helper modules
- live data access through Alpha Vantage or yfinance for live mode

The LSTM section additionally requires:

- `models/lstm_model.keras`
- `models/lstm_scaler.pkl`
- `models/lstm_metadata.json`
- `models/lstm_history.json`

## Dashboard Modes

### Live Data Mode

Live mode fetches recent stock history, computes the same technical indicators used during training, and predicts direction.

Flow:

```text
app.py
  -> predict_live.py
    -> validate symbol using symbol_mapper.py
    -> fetch live data using fetch_live_data.py
    -> compute live features
    -> load final_random_forest.pkl using model_config.py
    -> predict UP/DOWN and confidence
    -> save live CSV in live_api_dumps/
    -> optionally upload to HDFS
    -> optionally trigger spark_live_processing.py
  -> mongo_store.py saves prediction history when MongoDB is available
```

For LSTM predictions:

```text
app.py
  -> predict_lstm.py
    -> fetch historical data through fetch_live_data.py
    -> compute LSTM feature sequence
    -> load lstm_model.keras and lstm_scaler.pkl
    -> predict UP/DOWN using a 60-day window
```

### Manual Input Mode

Manual mode lets the user enter OHLCV and technical indicator values directly.

Flow:

```text
app.py
  -> builds a one-row feature DataFrame
  -> uses final_random_forest.pkl
  -> predicts UP/DOWN
  -> optionally saves result to MongoDB
  -> optionally runs spark_processing.py on data/final_featured_data.csv
```

Manual mode only uses Random Forest because the LSTM needs a 60-day sequence, not one snapshot.

## Command Line Usage

### Live Random Forest Prediction

```powershell
python scripts\predict_live.py --symbol AAPL --source auto --skip-hdfs
```

Other examples:

```powershell
python scripts\predict_live.py --symbols AAPL MSFT GOOGL --source yfinance --skip-hdfs
python scripts\predict_live.py --symbol TSLA --no-save --skip-hdfs --no-spark
python scripts\predict_live.py --list-stocks
```

### Live LSTM Prediction

```powershell
python scripts\predict_lstm.py --symbol AAPL --source auto
```

With yfinance only:

```powershell
python scripts\predict_lstm.py --symbol TSLA --source yfinance
```

### Fetch Live Features Only

```powershell
python scripts\fetch_live_data.py --symbol AAPL
python scripts\fetch_live_data.py --symbol TSLA --source yfinance
```

### Search or Resolve Symbols

```powershell
python scripts\symbol_mapper.py --count
python scripts\symbol_mapper.py --list
python scripts\symbol_mapper.py --search Tesla
python scripts\symbol_mapper.py AAPL
```

## Full Training Pipeline

The main structured-data pipeline is:

```text
raw stock CSV files
  -> merge_stock_data.py
  -> data/merged_stock_data.csv
  -> preprocessing.py
  -> data/cleaned_stock_data.csv
  -> feature_engineering.py
  -> data/final_featured_data.csv
  -> final_random_forest_model.py
  -> models/final_random_forest.pkl
  -> app.py / predict_live.py
```

Run it step by step:

```powershell
python scripts\merge_stock_data.py
python scripts\preprocessing.py
python scripts\feature_engineering.py
python scripts\final_random_forest_model.py
```

To train the LSTM:

```powershell
python scripts\train_lstm_model.py
```

The LSTM training script reads:

```text
data/final_featured_data.csv
```

and writes:

```text
models/lstm_model.keras
models/lstm_scaler.pkl
models/lstm_metadata.json
models/lstm_history.json
```

## Model Details

### Random Forest

Main files:

- training script: `scripts/final_random_forest_model.py`
- model file: `models/final_random_forest.pkl`
- metadata file: `models/model_metadata.json`
- shared config: `scripts/model_config.py`

Expected feature columns:

```text
Open, High, Low, Close, Volume,
MA5, MA10, MA20, EMA12,
RSI, MACD, MACD_Signal,
Daily_Returns, Volatility, Price_Change_Pct,
Weekly_Momentum, Trend_Strength, BB_Width,
Avg_5D_Volume_Trend, Lag_1, Lag_3
```

Current metadata reports:

- model type: `RandomForestClassifier`
- split strategy: chronological 80/20
- CV strategy: `TimeSeriesSplit`, 5 splits
- accuracy: about 52.6%
- ROC AUC: about 0.532
- F1 score: about 0.581

### LSTM

Main files:

- training script: `scripts/train_lstm_model.py`
- prediction script: `scripts/predict_lstm.py`
- model file: `models/lstm_model.keras`
- scaler file: `models/lstm_scaler.pkl`
- metadata file: `models/lstm_metadata.json`
- training curves: `models/lstm_history.json`

Current architecture:

```text
LSTM(128, return_sequences=True)
Dropout(0.2)
LSTM(64)
Dropout(0.2)
Dense(32, relu)
Dense(1, sigmoid)
```

Current metadata reports:

- sequence length: 60 days
- accuracy: about 53.2%
- ROC AUC: about 0.537
- F1 score: about 0.666

### Fusion Model

Main files:

- feature builder: `scripts/fusion_feature_engineering.py`
- trainer: `scripts/train_fusion_model.py`
- model file: `models/fusion_random_forest.pkl`
- metadata file: `models/fusion_metadata.json`

The fusion model combines:

- 21 structured stock features
- 128 CNN feature columns named `cnn_feature_0` through `cnn_feature_127`

It reads:

```text
data/final_featured_data.csv
data/cnn_features.csv
```

and writes:

```text
data/fusion_features.csv
models/fusion_random_forest.pkl
models/fusion_metadata.json
```

The current Streamlit app primarily uses the final Random Forest and LSTM models. The fusion model is available as an experimental/offline pipeline artifact.

## Data Pipeline Scripts

### `scripts/merge_stock_data.py`

Discovers CSV files in:

```text
data/stock_market_dataset/stocks/
```

Adds symbols, standardizes input records, and writes:

```text
data/merged_stock_data.csv
```

Useful commands:

```powershell
python scripts\merge_stock_data.py
python scripts\merge_stock_data.py --symbols AAPL MSFT GOOGL TSLA
python scripts\merge_stock_data.py --limit 100
```

### `scripts/preprocessing.py`

Reads:

```text
data/merged_stock_data.csv
```

Cleans dates, numeric columns, duplicates, missing values, price sanity, low-data symbols, and writes:

```text
data/cleaned_stock_data.csv
```

### `scripts/feature_engineering.py`

Reads:

```text
data/cleaned_stock_data.csv
```

Computes moving averages, returns, RSI, MACD, Bollinger values, lag features, volume trends, target labels, and writes:

```text
data/final_featured_data.csv
```

### `scripts/final_random_forest_model.py`

Reads:

```text
data/final_featured_data.csv
```

Trains the production Random Forest model and writes:

```text
models/final_random_forest.pkl
models/model_metadata.json
```

### `scripts/train_lstm_model.py`

Reads engineered features, builds 60-day sequences per symbol, trains the LSTM, and writes LSTM model artifacts to `models/`.

## Live Prediction Scripts

### `scripts/fetch_live_data.py`

Fetches recent market data from:

- Alpha Vantage
- yfinance fallback

It normalizes OHLCV data and computes the live feature set expected by the trained models.

### `scripts/predict_live.py`

Main Random Forest live prediction engine.

Responsibilities:

- validate stock symbol
- load the production Random Forest model
- fetch and feature-engineer live data
- save live CSV dumps
- optionally upload to HDFS
- optionally trigger live Spark processing
- return prediction, confidence, probabilities, feature values, and metadata

### `scripts/predict_lstm.py`

LSTM live prediction engine.

Responsibilities:

- load LSTM model and scaler
- fetch enough historical data for a 60-day sequence
- compute features
- scale and reshape input
- return LSTM prediction and probabilities

### `scripts/mongo_store.py`

Handles MongoDB connection and prediction history.

Used by:

- `app.py`

Default MongoDB settings:

```text
URI: mongodb://localhost:27017
Database: stock_predictor
Collection: predictions
```

## Spark and HDFS Scripts

Spark and HDFS are optional. The app can still make predictions without HDFS if `skip_hdfs` is enabled.

### `scripts/spark_processing.py`

Batch Spark pipeline for CSV input.

It can:

- start a Spark session
- read CSV data
- validate required OHLCV columns
- clean data
- generate Spark features
- save Parquet output
- save summary statistics
- write DAG and RDD demonstration files

Example:

```powershell
python scripts\spark_processing.py --input data/final_featured_data.csv --output spark_output --summary spark_summary --visualize-dag --demonstrate-rdd
```

With HDFS:

```powershell
python scripts\spark_processing.py --input hdfs://localhost:9000/stock_data/final_featured_data.csv --output hdfs://localhost:9000/stock_data/spark_output --use-hdfs
```

### `scripts/spark_live_processing.py`

Processes newly uploaded live API files from HDFS.

Flow:

```text
detect latest file in /stock_data/live_api_dumps/
  -> read CSV with Spark
  -> compute Spark features
  -> save Parquet to /stock_data/live_processed/
  -> update metadata/last_processed.txt
```

Example:

```powershell
python scripts\spark_live_processing.py --local
```

or with Spark:

```powershell
spark-submit scripts\spark_live_processing.py --force
```

### `scripts/hdfs_upload.py`

Uploads datasets and assets to HDFS.

Default HDFS root:

```text
/stock_data
```

Example:

```powershell
python scripts\hdfs_upload.py --hdfs-root /stock_data
```

### `scripts/upload_images_to_hdfs.py`

Uploads generated candlestick image folders to HDFS.

Example:

```powershell
python scripts\upload_images_to_hdfs.py --hdfs-path /stock_data/images/
```

## Image and Fusion Pipeline

These scripts support the optional unstructured-data path.

### `scripts/generate_candlestick_images.py`

Reads:

```text
data/cleaned_stock_data.csv
```

Generates candlestick chart images into:

```text
images/bullish/
images/bearish/
```

### `scripts/generate_image_labels.py`

Reads image files from:

```text
images/bullish/
images/bearish/
```

Writes:

```text
data/image_labels.csv
```

### `scripts/fusion_feature_engineering.py`

Combines structured features and CNN image features.

Reads:

```text
data/final_featured_data.csv
data/cnn_features.csv
```

Writes:

```text
data/fusion_features.csv
```

### `scripts/train_fusion_model.py`

Trains a Random Forest on the combined structured + CNN feature table.

Writes:

```text
models/fusion_random_forest.pkl
models/fusion_metadata.json
```

## Shared Helper Files

### `scripts/model_config.py`

Central source for:

- project root
- `models/` paths
- final Random Forest file path
- metadata path
- expected 21 feature columns
- schema validation

Used by:

- `app.py`
- `fetch_live_data.py`
- `predict_live.py`
- `final_random_forest_model.py`

### `scripts/symbol_mapper.py`

Discovers available stock symbols from:

```text
data/stock_market_dataset/stocks/
data/stock_market_dataset/etfs/
```

Provides:

- symbol validation
- company-name lookup
- search by symbol or name
- popular company-name mappings such as Apple, Tesla, Microsoft, NVIDIA

Used by:

- `app.py`
- `predict_live.py`

## How Files Connect

High-level dependency map:

```text
data/stock_market_dataset/stocks/*.csv
  -> scripts/merge_stock_data.py
  -> data/merged_stock_data.csv
  -> scripts/preprocessing.py
  -> data/cleaned_stock_data.csv
  -> scripts/feature_engineering.py
  -> data/final_featured_data.csv
```

Model training:

```text
data/final_featured_data.csv
  -> scripts/final_random_forest_model.py
  -> models/final_random_forest.pkl
  -> models/model_metadata.json
```

```text
data/final_featured_data.csv
  -> scripts/train_lstm_model.py
  -> models/lstm_model.keras
  -> models/lstm_scaler.pkl
  -> models/lstm_metadata.json
  -> models/lstm_history.json
```

Dashboard:

```text
app.py
  -> scripts/model_config.py
  -> scripts/predict_live.py
  -> scripts/predict_lstm.py
  -> scripts/fetch_live_data.py
  -> scripts/symbol_mapper.py
  -> scripts/mongo_store.py
  -> models/*
  -> live_api_dumps/
  -> spark_output/, spark_summary/, spark_visualizations/
```

Live Random Forest prediction:

```text
app.py or scripts/predict_live.py
  -> scripts/symbol_mapper.py
  -> scripts/fetch_live_data.py
  -> models/final_random_forest.pkl
  -> live_api_dumps/*.csv
  -> optional HDFS upload
  -> optional scripts/spark_live_processing.py
```

MongoDB history:

```text
app.py
  -> scripts/mongo_store.py
  -> MongoDB stock_predictor.predictions
```

Spark batch processing:

```text
data/final_featured_data.csv
  -> scripts/spark_processing.py
  -> spark_output/
  -> spark_summary/
  -> spark_visualizations/
```

Fusion model:

```text
data/cleaned_stock_data.csv
  -> scripts/generate_candlestick_images.py
  -> images/bullish/, images/bearish/
  -> scripts/generate_image_labels.py
  -> data/image_labels.csv
```

```text
data/final_featured_data.csv + data/cnn_features.csv
  -> scripts/fusion_feature_engineering.py
  -> data/fusion_features.csv
  -> scripts/train_fusion_model.py
  -> models/fusion_random_forest.pkl
```

## Important Generated Folders

### `live_api_dumps/`

Stores timestamped CSV files created during live prediction.

Example file:

```text
AAPL_live_2026_05_21_095656.csv
```

### `metadata/`

Stores local state for Spark live processing.

Current file:

```text
metadata/last_processed.txt
```

### `spark_output/`

Spark batch output in Parquet format.

### `spark_summary/`

Spark-computed summary statistics.

### `spark_visualizations/`

Text outputs explaining Spark execution:

- `spark_dag_plan.txt`
- `spark_rdd_demonstration.txt`

## Optional Services

### MongoDB

MongoDB is optional. If it is unavailable, predictions still run, but prediction history may not save or display.

Start a local MongoDB server, then use:

```env
MONGODB_URI=mongodb://localhost:27017
```

### HDFS

HDFS is optional and mainly used for big-data demonstration.

Expected NameNode:

```text
hdfs://localhost:9000
```

Expected project paths:

```text
/stock_data/live_api_dumps/
/stock_data/live_processed/
/stock_data/images/
```

If HDFS is not running, use dashboard options or CLI flags such as:

```powershell
--skip-hdfs
--no-spark
```

## Troubleshooting

### Model not found

If the dashboard says the model is missing, train the production model:

```powershell
python scripts\final_random_forest_model.py
```

Make sure this file exists afterward:

```text
models/final_random_forest.pkl
```

### LSTM not available

Train the LSTM:

```powershell
python scripts\train_lstm_model.py
```

Required files:

```text
models/lstm_model.keras
models/lstm_scaler.pkl
```

### Live API fails

Check:

- `.env` contains `ALPHA_VANTAGE_API_KEY`
- internet access is available
- Alpha Vantage rate limits have not been exceeded
- try yfinance fallback:

```powershell
python scripts\predict_live.py --symbol AAPL --source yfinance --skip-hdfs
```

### TensorFlow install issues

Use Python 3.10 or 3.11 and reinstall:

```powershell
pip install --upgrade pip
pip install tensorflow==2.15.0
```

### Spark fails on Windows

Check:

- Java is installed and available on `PATH`
- `JAVA_HOME` is set
- PySpark is installed from `requirements.txt`
- HDFS flags are only used when Hadoop/HDFS is actually running

For local-only prediction, skip HDFS:

```powershell
python scripts\predict_live.py --symbol AAPL --skip-hdfs --no-spark
```

## Development Notes

- `data/`, `models/`, `images/`, and `live_api_dumps/` are generated/local artifacts and are listed in `.gitignore`.
- The app inserts `scripts/` into `sys.path`, so modules inside `scripts/` can be imported directly by `app.py`.
- The Random Forest model is the main production model for both live and manual dashboard predictions.
- The LSTM model is live-mode only because it requires historical sequences.
- Spark/HDFS processing is optional and does not need to work for the core prediction UI to run.
