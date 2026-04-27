"""
predict_live.py
===============
Live production prediction engine.
Flow:
API fetch -> feature engineering -> strict schema validation -> model inference
-> confidence score -> MongoDB save.
"""

from __future__ import annotations

import os
import json
import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from fetch_live_data import fetch_live_stock_data
from mongo_store import save_prediction
from model_config import FINAL_MODEL_FILE, get_expected_features, validate_feature_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(model_file: Path = FINAL_MODEL_FILE):
    if not model_file.exists():
        raise FileNotFoundError(f"Final model not found: {model_file}")
    return joblib.load(model_file)


def predict_live(symbol: str, source: str = "auto", api_key: str = "") -> dict:
    model = load_model()
    live = fetch_live_stock_data(symbol=symbol, api_key=api_key, source=source)
    expected_features = get_expected_features()
    validate_feature_schema(expected_features)

    row = {c: live["features"][c] for c in expected_features}
    X = pd.DataFrame([row], columns=expected_features)
    pred = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    confidence = float(probs.max() * 100)
    label = "UP" if pred == 1 else "DOWN"

    mongo_id = save_prediction(
        symbol=symbol,
        prediction=label,
        confidence=confidence,
        source=live["source"],
        model="Random Forest",
        meta={
            "latest_date": live["latest_date"],
            "down_probability": float(probs[0]),
            "up_probability": float(probs[1]),
        },
    )

    return {
        "symbol": symbol,
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": {"DOWN": float(probs[0]), "UP": float(probs[1])},
        "latest_date": live["latest_date"],
        "source": live["source"],
        "features": row,
        "mongo_id": mongo_id,
        "model_file": str(FINAL_MODEL_FILE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live production prediction with final Random Forest model")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--source", type=str, default="auto", choices=["alpha_vantage", "yfinance", "auto"])
    parser.add_argument("--api-key", type=str, default=os.environ.get("ALPHA_VANTAGE_API_KEY", ""))
    args = parser.parse_args()

    result = predict_live(symbol=args.symbol, source=args.source, api_key=args.api_key)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

