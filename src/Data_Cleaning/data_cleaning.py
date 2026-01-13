from pyspark.sql.functions import (
    col, when, count, to_timestamp, year, month, dayofmonth, dayofweek, isnan
)
from pyspark.sql.window import Window

from config.spark_session import get_spark_session, data_load

spark = get_spark_session()

df = data_load()

def mode_description(df):
    mode = df.groupBy("Description").count().orderBy(col("count").desc()).first()[0]
    return mode

def imputation_valeurs_manquante(df):
    print("avant l'imputation")
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    mode_descriptions = mode_description(df)
    df_clean = df.fillna({"Description": mode_descriptions})

    #Suppression des lignes avec CustomerID manquant
    df_clean = df.filter(col("CustomerID").isNotNull())

    print("aprés l'imputation")
    df_clean.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_clean.columns]).show()

    df_clean = df_clean.withColumn(
    "TotalPrice",
    col("Quantity") * col("UnitPrice")
        )
    return df_clean

def enrichissement_InvoiceDate(df_clean):

    print("avant l'enrichissement")
    df_clean.select('InvoiceDate').show(10)
    df_clean = df_clean.withColumn(
    "InvoiceDate",
    to_timestamp(col("InvoiceDate"), "MM/dd/yyyy HH:mm")
    )

    df_clean = (
        df_clean
        .withColumn("Year", year(col("InvoiceDate")))
        .withColumn("Month", month(col("InvoiceDate")))
        .withColumn("Day", dayofmonth(col("InvoiceDate")))
        .withColumn("DayOfWeek", dayofweek(col("InvoiceDate")))
    )
    print("aprés l'enrichissement")
    df_clean.select('InvoiceDate','Year','Month','Day','DayOfWeek').show(10)

    return df_clean

