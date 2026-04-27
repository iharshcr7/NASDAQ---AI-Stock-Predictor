"""
spark_processing.py
===================
Distributed preprocessing and feature generation with PySpark on HDFS data.
"""

from __future__ import annotations

import argparse
import logging

from pyspark.sql import SparkSession, functions as F, Window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_spark(app_name: str = "StockBigDataProcessing") -> SparkSession:
    return (
    SparkSession.builder 
    .appName("StockMarketSparkProcessing") 
    .master("local[*]") 
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") 
    .getOrCreate()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed Spark pipeline for stock data")
    parser.add_argument("--input", default="hdfs://localhost:9000/stock_data/processed_data/final_featured_data.csv")
    parser.add_argument("--output", default="hdfs://localhost:9000/stock_data/spark_output/")
    args = parser.parse_args()

    spark = build_spark()
    logger.info("Reading from %s", args.input)
    df = spark.read.option("header", True).option("inferSchema", True).csv(args.input)

    # distributed cleaning
    df = df.dropDuplicates(["Symbol", "Date"])
    df = df.dropna(subset=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.withColumn("Date", F.to_date("Date"))

    # distributed feature generation (core)
    w = Window.partitionBy("Symbol").orderBy("Date")
    w5 = w.rowsBetween(-4, 0)
    w10 = w.rowsBetween(-9, 0)
    w20 = w.rowsBetween(-19, 0)

    df = df.withColumn("Daily_Returns_Spark", (F.col("Close") / F.lag("Close", 1).over(w) - 1.0) * 100.0)
    df = df.withColumn("Volatility_Spark", F.col("High") - F.col("Low"))
    df = df.withColumn("MA5_Spark", F.avg("Close").over(w5))
    df = df.withColumn("MA10_Spark", F.avg("Close").over(w10))
    df = df.withColumn("MA20_Spark", F.avg("Close").over(w20))
    df = df.withColumn("Weekly_Momentum_Spark", (F.col("Close") / F.lag("Close", 5).over(w) - 1.0) * 100.0)

    logger.info("Writing processed data to %s", args.output)
    df.write.mode("overwrite").parquet(args.output)

    logger.info("Computing distributed summary statistics")
    summary = df.select(
        F.count("*").alias("rows"),
        F.countDistinct("Symbol").alias("symbols"),
        F.min("Date").alias("min_date"),
        F.max("Date").alias("max_date"),
        F.avg("Close").alias("avg_close"),
        F.stddev("Close").alias("std_close"),
    )
    summary.show(truncate=False)

    spark.stop()
    logger.info("Spark distributed processing complete.")


if __name__ == "__main__":
    main()

