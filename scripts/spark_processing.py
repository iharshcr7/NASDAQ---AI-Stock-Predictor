"""
spark_processing.py
===================
Production-ready distributed preprocessing and feature generation with PySpark on HDFS data.

This module provides:
- Robust exception handling for production environments
- Feature column validation
- Advanced technical indicators (MA, Volatility, Returns, Momentum)
- Summary statistics export to HDFS
- Flexible input handling (single file or directory)
- Professional logging with reduced Spark verbosity
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from pyspark.sql import SparkSession, DataFrame, functions as F, Window
from pyspark.sql.utils import AnalysisException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Required columns for processing
REQUIRED_COLUMNS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]


def build_spark(app_name: str = "StockBigDataProcessing") -> SparkSession:
    """
    Build and configure Spark session with HDFS support and optimized logging.
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        Configured SparkSession instance
    """
    try:
        spark = (
            SparkSession.builder 
            .appName(app_name) 
            .master("local[*]") 
            .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate()
        )
        
        # Reduce Spark logging verbosity for cleaner output
        spark.sparkContext.setLogLevel("ERROR")
        logger.info("Spark session initialized successfully")
        return spark
        
    except Exception as e:
        logger.exception("Failed to initialize Spark session")
        raise


def validate_columns(df: DataFrame, required_cols: List[str]) -> None:
    """
    Validate that all required columns exist in the DataFrame.
    
    Args:
        df: Input DataFrame to validate
        required_cols: List of required column names
        
    Raises:
        ValueError: If any required columns are missing
    """
    actual_cols = set(df.columns)
    required_set = set(required_cols)
    missing_cols = required_set - actual_cols
    
    if missing_cols:
        error_msg = f"Missing required columns: {sorted(missing_cols)}"
        logger.error(error_msg)
        logger.error(f"Available columns: {sorted(actual_cols)}")
        raise ValueError(error_msg)
    
    logger.info("✓ All required columns validated successfully")


def read_input_data(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Read input data from HDFS with robust error handling.
    Supports both single files and directories.
    
    Args:
        spark: Active SparkSession
        input_path: HDFS path to input file or directory
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        AnalysisException: If input path doesn't exist or is invalid
    """
    try:
        logger.info(f"Reading input from: {input_path}")
        
        # Attempt to read as CSV (handles both files and directories)
        df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)
        
        row_count = df.count()
        logger.info(f"✓ Successfully loaded {row_count:,} rows from input")
        
        return df
        
    except AnalysisException as e:
        logger.exception(f"Failed to read input from {input_path}")
        logger.error("Possible causes: Path doesn't exist, HDFS not running, or invalid format")
        raise
    except Exception as e:
        logger.exception("Unexpected error while reading input data")
        raise


def clean_data(df: DataFrame) -> DataFrame:
    """
    Perform distributed data cleaning operations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    
    initial_count = df.count()
    
    # Remove duplicates based on Symbol and Date
    df = df.dropDuplicates(["Symbol", "Date"])
    
    # Remove rows with missing critical values
    df = df.dropna(subset=REQUIRED_COLUMNS)
    
    # Ensure Date column is properly formatted
    df = df.withColumn("Date", F.to_date("Date"))
    
    final_count = df.count()
    removed_count = initial_count - final_count
    
    logger.info(f"✓ Data cleaning complete: {removed_count:,} rows removed, {final_count:,} rows remaining")
    
    return df


def generate_features(df: DataFrame) -> DataFrame:
    """
    Generate advanced technical indicators using distributed window functions.
    
    Features generated:
    - Moving Averages (MA5, MA10, MA20)
    - Daily Returns (percentage change)
    - Weekly Momentum (5-day percentage change)
    - Volatility (normalized price range)
    - Rolling Standard Deviation (20-day)
    - Volume Moving Average (20-day)
    - Bollinger Band Width
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with additional feature columns
    """
    logger.info("Generating features...")
    
    # Define window specifications
    w = Window.partitionBy("Symbol").orderBy("Date")
    w5 = w.rowsBetween(-4, 0)
    w10 = w.rowsBetween(-9, 0)
    w20 = w.rowsBetween(-19, 0)
    
    # Price-based features
    df = df.withColumn("Daily_Returns_Spark", 
                       (F.col("Close") / F.lag("Close", 1).over(w) - 1.0) * 100.0)
    
    df = df.withColumn("Weekly_Momentum_Spark", 
                       (F.col("Close") / F.lag("Close", 5).over(w) - 1.0) * 100.0)
    
    # Improved volatility formula: (High - Low) / Close * 100
    df = df.withColumn("Volatility_Spark", 
                       ((F.col("High") - F.col("Low")) / F.col("Close")) * 100.0)
    
    # Moving averages
    df = df.withColumn("MA5_Spark", F.avg("Close").over(w5))
    df = df.withColumn("MA10_Spark", F.avg("Close").over(w10))
    df = df.withColumn("MA20_Spark", F.avg("Close").over(w20))
    
    # Rolling standard deviation (20-day)
    df = df.withColumn("Rolling_Std_20_Spark", F.stddev("Close").over(w20))
    
    # Volume features
    df = df.withColumn("Volume_MA20_Spark", F.avg("Volume").over(w20))
    df = df.withColumn("Volume_Ratio_Spark", 
                       F.col("Volume") / F.col("Volume_MA20_Spark"))
    
    # Bollinger Band Width (simplified): 2 * std / MA20
    df = df.withColumn("Bollinger_Width_Spark", 
                       (2.0 * F.col("Rolling_Std_20_Spark")) / F.col("MA20_Spark") * 100.0)
    
    # Lag features for time series
    df = df.withColumn("Close_Lag1_Spark", F.lag("Close", 1).over(w))
    df = df.withColumn("Close_Lag5_Spark", F.lag("Close", 5).over(w))
    
    logger.info("✓ Feature generation complete")
    
    return df


def save_output(df: DataFrame, output_path: str) -> None:
    """
    Save processed data to HDFS in Parquet format.
    
    Args:
        df: Processed DataFrame
        output_path: HDFS path for output
        
    Raises:
        Exception: If write operation fails
    """
    try:
        logger.info(f"Writing processed data to: {output_path}")
        df.write.mode("overwrite").parquet(output_path)
        logger.info("✓ Successfully saved processed data in Parquet format")
        
    except Exception as e:
        logger.exception("Failed to write output data")
        raise


def compute_and_save_summary(df: DataFrame, summary_path: str) -> None:
    """
    Compute comprehensive summary statistics and save to HDFS.
    
    Args:
        df: Processed DataFrame
        summary_path: HDFS path for summary statistics
    """
    try:
        logger.info("Computing distributed summary statistics...")
        
        summary = df.select(
            F.count("*").alias("total_rows"),
            F.countDistinct("Symbol").alias("unique_symbols"),
            F.min("Date").alias("date_range_start"),
            F.max("Date").alias("date_range_end"),
            F.avg("Close").alias("avg_close_price"),
            F.stddev("Close").alias("std_close_price"),
            F.min("Close").alias("min_close_price"),
            F.max("Close").alias("max_close_price"),
            F.avg("Volume").alias("avg_volume"),
            F.avg("Daily_Returns_Spark").alias("avg_daily_returns"),
            F.avg("Volatility_Spark").alias("avg_volatility"),
        )
        
        # Display summary
        logger.info("Summary Statistics:")
        summary.show(truncate=False)
        
        # Save summary to HDFS
        logger.info(f"Saving summary statistics to: {summary_path}")
        summary.write.mode("overwrite").csv(summary_path, header=True)
        logger.info("✓ Summary statistics saved successfully")
        
    except Exception as e:
        logger.exception("Failed to compute or save summary statistics")
        # Don't raise - summary is non-critical


def main() -> None:
    """
    Main execution function with comprehensive error handling.
    """
    parser = argparse.ArgumentParser(
        description="Production-ready distributed Spark pipeline for stock market data processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python spark_processing.py --input hdfs://localhost:9000/stock_data/processed_data/final_featured_data.csv
  
  # Process entire directory
  python spark_processing.py --input hdfs://localhost:9000/stock_data/stock_market_dataset/stocks/
  
  # Custom output location
  python spark_processing.py --input <input_path> --output hdfs://localhost:9000/custom_output/
        """
    )
    parser.add_argument(
        "--input", 
        default="hdfs://localhost:9000/stock_data/processed_data/final_featured_data.csv",
        help="HDFS path to input CSV file or directory"
    )
    parser.add_argument(
        "--output", 
        default="hdfs://localhost:9000/stock_data/spark_output/",
        help="HDFS path for output Parquet files"
    )
    parser.add_argument(
        "--summary", 
        default="hdfs://localhost:9000/stock_data/spark_summary/",
        help="HDFS path for summary statistics"
    )
    
    args = parser.parse_args()
    
    spark = None
    exit_code = 0
    
    try:
        # Initialize Spark session
        spark = build_spark()
        
        # Read input data
        df = read_input_data(spark, args.input)
        
        # Validate required columns
        validate_columns(df, REQUIRED_COLUMNS)
        
        # Clean data
        df = clean_data(df)
        
        # Generate features
        df = generate_features(df)
        
        # Save processed output
        save_output(df, args.output)
        
        # Compute and save summary statistics
        compute_and_save_summary(df, args.summary)
        
        logger.info("=" * 70)
        logger.info("✓ Spark distributed processing completed successfully")
        logger.info("=" * 70)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        exit_code = 1
        
    except AnalysisException as e:
        logger.error(f"HDFS/Spark analysis error: {e}")
        logger.error("Please verify HDFS is running and paths are correct")
        exit_code = 2
        
    except Exception as e:
        logger.exception("Unexpected error during Spark processing")
        exit_code = 3
        
    finally:
        # Always stop Spark session gracefully
        if spark is not None:
            try:
                spark.stop()
                logger.info("Spark session stopped gracefully")
            except Exception as e:
                logger.error(f"Error stopping Spark session: {e}")
        
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

