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
- DAG visualization for educational purposes
- RDD demonstrations
- Spark UI keep-alive for debugging
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from typing import List
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame, functions as F, Window
from pyspark.sql.utils import AnalysisException
from pyspark import RDD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Required columns for processing
REQUIRED_COLUMNS = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]


def build_spark(app_name: str = "StockBigDataProcessing", use_hdfs: bool = False) -> SparkSession:
    """
    Build and configure Spark session with optional HDFS support and optimized logging.
    
    Args:
        app_name: Name of the Spark application
        use_hdfs: Whether to configure HDFS (default: False for local files)
        
    Returns:
        Configured SparkSession instance
    """
    try:
        # Fix Python version mismatch between driver and worker
        python_exe = sys.executable
        os.environ["PYSPARK_PYTHON"] = python_exe
        os.environ["PYSPARK_DRIVER_PYTHON"] = python_exe
        logger.info(f"Python path set: {python_exe}")
        
        builder = (
            SparkSession.builder 
            .appName(app_name) 
            .master("local[*]")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        )
        
        # Only configure HDFS if requested
        if use_hdfs:
            builder = builder.config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
            logger.info("HDFS enabled: hdfs://localhost:9000")
        else:
            logger.info("HDFS disabled: using local filesystem")
        
        spark = builder.getOrCreate()
        
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
    
    logger.info("All required columns validated successfully")


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
        logger.info(f"Successfully loaded {row_count:,} rows from input")
        
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
    
    logger.info(f"Data cleaning complete: {removed_count:,} rows removed, {final_count:,} rows remaining")
    
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
    
    logger.info("Feature generation complete")
    
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
        logger.info("Successfully saved processed data in Parquet format")
        
    except Exception as e:
        logger.exception("Failed to write output data")
        raise


def visualize_dag(df: DataFrame, output_dir: str = "spark_visualizations") -> None:
    """
    Visualize the Spark DAG (Directed Acyclic Graph) for the DataFrame.
    Saves the DAG visualization to a local file.
    
    Args:
        df: DataFrame to visualize DAG for
        output_dir: Directory to save visualization files
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("SPARK DAG VISUALIZATION")
        logger.info("=" * 70)
        
        # Get the logical plan
        logical_plan = df._jdf.queryExecution().logical()
        logger.info("Logical Plan:")
        logger.info(str(logical_plan))
        
        # Get the physical plan
        physical_plan = df._jdf.queryExecution().executedPlan()
        logger.info("\nPhysical Plan:")
        logger.info(str(physical_plan))
        
        # Save plans to file
        dag_file = output_path / "spark_dag_plan.txt"
        with open(dag_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("SPARK DAG - LOGICAL PLAN\n")
            f.write("=" * 70 + "\n")
            f.write(str(logical_plan) + "\n\n")
            f.write("=" * 70 + "\n")
            f.write("SPARK DAG - PHYSICAL PLAN\n")
            f.write("=" * 70 + "\n")
            f.write(str(physical_plan) + "\n")
        
        logger.info(f"DAG plan saved to: {dag_file}")
        
        # Print DAG structure explanation
        logger.info("\nDAG Structure Explanation:")
        logger.info("-" * 70)
        logger.info("The DAG shows the sequence of transformations applied to the data:")
        logger.info("1. Read: Load data from HDFS (CSV)")
        logger.info("2. Filter: Remove null values and duplicates")
        logger.info("3. Project: Select and compute columns (feature generation)")
        logger.info("4. Window: Apply window functions for rolling calculations")
        logger.info("5. Write: Save output to HDFS (Parquet)")
        logger.info("-" * 70)
        
    except Exception as e:
        logger.warning(f"Failed to visualize DAG: {e}")
        # Don't raise - visualization is non-critical


def demonstrate_rdd_operations(spark: SparkSession, df: DataFrame, output_dir: str = "spark_visualizations") -> None:
    """
    Demonstrate RDD transformations and actions for educational purposes.
    Shows the difference between DataFrame API and RDD API.
    
    Args:
        spark: Active SparkSession
        df: DataFrame to convert to RDD
        output_dir: Directory to save RDD demonstration files
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("SPARK RDD DEMONSTRATION")
        logger.info("=" * 70)
        
        # Convert DataFrame to RDD
        rdd = df.rdd
        logger.info(f"DataFrame converted to RDD")
        logger.info(f"  RDD partitions: {rdd.getNumPartitions()}")
        
        # RDD Transformation 1: map
        logger.info("\nRDD Transformation - map:")
        logger.info("  Operation: Extract Close price from each row")
        close_rdd = rdd.map(lambda row: row.Close)
        logger.info(f"  Result: First 5 close prices: {close_rdd.take(5)}")
        
        # RDD Transformation 2: filter
        logger.info("\nRDD Transformation - filter:")
        logger.info("  Operation: Filter rows with Volume > 1,000,000")
        high_volume_rdd = rdd.filter(lambda row: row.Volume > 1000000 if row.Volume else False)
        logger.info(f"  Result: Count of high volume rows: {high_volume_rdd.count()}")
        
        # RDD Transformation 3: reduceByKey
        logger.info("\nRDD Transformation - reduceByKey:")
        logger.info("  Operation: Group by Symbol and count rows")
        symbol_rdd = rdd.map(lambda row: (row.Symbol, 1))
        symbol_counts = symbol_rdd.reduceByKey(lambda a, b: a + b)
        logger.info(f"  Result: Symbol counts: {symbol_counts.take(10)}")
        
        # RDD Action 1: count
        logger.info("\nRDD Action - count:")
        logger.info(f"  Total rows in RDD: {rdd.count()}")
        
        # RDD Action 2: take
        logger.info("\nRDD Action - take:")
        logger.info(f"  First 3 rows: {rdd.take(3)}")
        
        # RDD Action 3: collect (sample)
        logger.info("\nRDD Action - collect (sample):")
        sample_data = rdd.sample(False, 0.001).collect()
        logger.info(f"  Sample size: {len(sample_data)} rows")
        
        # Save RDD demonstration to file
        rdd_file = output_path / "spark_rdd_demonstration.txt"
        with open(rdd_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("SPARK RDD TRANSFORMATIONS AND ACTIONS DEMONSTRATION\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("RDD Information:\n")
            f.write(f"  Partitions: {rdd.getNumPartitions()}\n")
            f.write(f"  Total Rows: {rdd.count()}\n\n")
            
            f.write("RDD Transformations:\n")
            f.write("  1. map: Extract Close price\n")
            f.write(f"     First 5 close prices: {close_rdd.take(5)}\n\n")
            
            f.write("  2. filter: High volume rows (>1M)\n")
            f.write(f"     Count: {high_volume_rdd.count()}\n\n")
            
            f.write("  3. reduceByKey: Count rows per symbol\n")
            f.write(f"     Symbol counts: {symbol_counts.take(10)}\n\n")
            
            f.write("RDD Actions:\n")
            f.write("  1. count: Total rows\n")
            f.write(f"     Result: {rdd.count()}\n\n")
            
            f.write("  2. take: First N rows\n")
            f.write(f"     First 3 rows: {rdd.take(3)}\n\n")
            
            f.write("  3. collect: Collect all data (sampled)\n")
            f.write(f"     Sample size: {len(sample_data)} rows\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("RDD vs DataFrame API Comparison\n")
            f.write("=" * 70 + "\n")
            f.write("RDD API:\n")
            f.write("  - Lower-level, more control\n")
            f.write("  - Manual optimization needed\n")
            f.write("  - Python/Java/Scala native\n")
            f.write("  - Good for unstructured data\n\n")
            f.write("DataFrame API:\n")
            f.write("  - Higher-level, easier to use\n")
            f.write("  - Catalyst optimizer optimizes automatically\n")
            f.write("  - SQL-like operations\n")
            f.write("  - Better for structured data\n")
            f.write("=" * 70 + "\n")
        
        logger.info(f"RDD demonstration saved to: {rdd_file}")
        
        # Print RDD explanation
        logger.info("\nRDD vs DataFrame Explanation:")
        logger.info("-" * 70)
        logger.info("RDD (Resilient Distributed Dataset):")
        logger.info("  - Fundamental data structure in Spark")
        logger.info("  - Immutable, partitioned collection of records")
        logger.info("  - Transformations: map, filter, flatMap, reduceByKey")
        logger.info("  - Actions: count, take, collect, reduce")
        logger.info("  - Lazy evaluation (transformations)")
        logger.info("  - Eager evaluation (actions)")
        logger.info("-" * 70)
        
    except Exception as e:
        logger.warning(f"Failed to demonstrate RDD operations: {e}")
        # Don't raise - demonstration is non-critical


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
        logger.info("Summary statistics saved successfully")
        
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
        default="data/final_featured_data.csv",
        help="Path to input CSV file or directory"
    )
    parser.add_argument(
        "--output", 
        default="spark_output/",
        help="Path for output Parquet files"
    )
    parser.add_argument(
        "--summary", 
        default="spark_summary/",
        help="Path for summary statistics"
    )
    parser.add_argument(
        "--visualize-dag",
        action="store_true",
        help="Enable DAG visualization (saves to spark_visualizations/)"
    )
    parser.add_argument(
        "--demonstrate-rdd",
        action="store_true",
        help="Enable RDD demonstration (saves to spark_visualizations/)"
    )
    parser.add_argument(
        "--use-hdfs",
        action="store_true",
        help="Enable HDFS configuration (required for HDFS paths)"
    )
    parser.add_argument(
        "--keep-ui",
        type=int,
        default=0,
        help="Keep Spark UI running for N seconds after completion (for demo purposes)"
    )
    parser.add_argument(
        "--keep-ui-forever",
        action="store_true",
        help="Keep Spark UI running indefinitely until Ctrl+C (for demo purposes)"
    )
    
    args = parser.parse_args()
    
    spark = None
    exit_code = 0
    
    try:
        # Initialize Spark session
        spark = build_spark(use_hdfs=args.use_hdfs)
        
        # Read input data
        df = read_input_data(spark, args.input)
        
        # Validate required columns
        validate_columns(df, REQUIRED_COLUMNS)
        
        # Clean data
        df = clean_data(df)
        
        # Generate features
        df = generate_features(df)
        
        # Cache DataFrame so it appears in Spark UI Storage tab
        df.cache()
        df.count()  # Trigger cache (action required)
        logger.info("DataFrame cached (visible in Spark UI Storage tab)")
        
        # Visualize DAG if requested
        if args.visualize_dag:
            visualize_dag(df)
        
        # Demonstrate RDD operations if requested
        if args.demonstrate_rdd:
            demonstrate_rdd_operations(spark, df)
        
        # Save processed output
        save_output(df, args.output)
        
        # Compute and save summary statistics
        compute_and_save_summary(df, args.summary)
        
        logger.info("=" * 70)
        logger.info("Spark distributed processing completed successfully")
        logger.info("=" * 70)
        
        # Keep Spark UI running for demo if requested
        if args.keep_ui_forever:
            logger.info("Keeping Spark UI running indefinitely...")
            logger.info("Access Spark UI at: http://localhost:4040")
            logger.info("Press Ctrl+C to stop")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt, stopping...")
        elif args.keep_ui > 0:
            logger.info(f"Keeping Spark UI running for {args.keep_ui} seconds...")
            logger.info(f"Access Spark UI at: http://localhost:4040")
            import time
            time.sleep(args.keep_ui)
        
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
