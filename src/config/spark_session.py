from pyspark.sql import SparkSession
from pathlib import Path
def get_spark_session(app_name="Customer Segmentation and Predictive Modeling"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.executor.memory", "6g")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.cores", "4")
        .config("spark.sql.shuffle.partitions", "1000")
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    return spark

spark = get_spark_session()

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ← racine du projet
data_path = str(BASE_DIR / "data" / "raw" / "Online Retail.csv")

def data_load(data_path= data_path):

    df = spark.read.csv(
        data_path,
        header=True,
        inferSchema=True
    )

    df.printSchema()
    return df
