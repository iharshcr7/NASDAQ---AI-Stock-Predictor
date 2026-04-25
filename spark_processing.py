from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name

spark = SparkSession.builder.appName("Stock Analysis").getOrCreate()

df = spark.read.csv(
    "data/stock_market_dataset/stocks/*.csv",
    header=True,
    inferSchema=True
)

df = df.withColumn("file", input_file_name())

print("===== SCHEMA =====")
df.printSchema()

print("===== SAMPLE DATA =====")
df.show(5)

print("===== TOTAL ROWS =====")
print(df.count())

print("===== STOCK FILE COUNT =====")
df.groupBy("file").count().show()

input("Press Enter to stop Spark...")
spark.stop()