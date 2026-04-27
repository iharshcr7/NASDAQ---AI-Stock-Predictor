"""
mongo_store.py
==============
MongoDB helpers for saving and reading live prediction records.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

DEFAULT_DB_NAME = os.environ.get("MONGODB_DB_NAME", "stock_predictor")
DEFAULT_COLLECTION_NAME = os.environ.get("MONGODB_COLLECTION_NAME", "predictions")


def connect_mongodb(uri: str | None = None, db_name: str | None = None, collection_name: str | None = None) -> Collection:
    mongo_uri = uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    db = db_name or DEFAULT_DB_NAME
    collection = collection_name or DEFAULT_COLLECTION_NAME
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[db][collection]


def save_prediction(
    symbol: str,
    prediction: str,
    confidence: float,
    source: str,
    model: str = "Random Forest",
    meta: dict | None = None,
    collection: Collection | None = None,
) -> str:
    payload = {
        "symbol": symbol,
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "model": model,
    }
    if meta:
        payload["meta"] = meta

    try:
        col = collection or connect_mongodb()
        result = col.insert_one(payload)
        return str(result.inserted_id)
    except PyMongoError as exc:
        logger.warning("Failed to save prediction in MongoDB: %s", exc)
        return ""


def fetch_recent_predictions(limit: int = 20, collection: Collection | None = None) -> list[dict]:
    try:
        col = collection or connect_mongodb()
        rows = list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        return rows
    except PyMongoError as exc:
        logger.warning("Failed to fetch recent predictions from MongoDB: %s", exc)
        return []

