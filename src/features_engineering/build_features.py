from pyspark.sql.functions import max, datediff, col, log
from pyspark.sql import functions as F
from pyspark.sql.functions import col, sum as _sum, countDistinct, datediff, to_date, lit, when, avg, udf, isnan, count
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import VectorAssembler, StandardScaler


from config.spark_session import get_spark_session, data_load

spark = get_spark_session()


def read_parquet(path):
    df_cleaned = spark.read.parquet(path)
    return df_cleaned


def rfm(df_cleaned, reference_date):
    rfm = df_cleaned.groupBy("CustomerID").agg(
        datediff(lit(reference_date), F.max(col("InvoiceDate"))).alias("Recency"),
        countDistinct("InvoiceNo").alias("Frequency"),
        _sum("TotalPrice").alias("Monetary")
    )
    rfm.show(10)
    return rfm


def rfm_visualization(rfm):

    # Convert Spark DataFrame to Pandas for visualization
    rfm_pd = rfm.toPandas()

    print("Distribution des RFM")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(data=rfm_pd, x='Recency', bins=30, kde=True)
    plt.title('Distribution of Recency')

    plt.subplot(1, 3, 2)
    sns.histplot(data=rfm_pd, x='Frequency', bins=30, kde=True)
    plt.title('Distribution of Frequency')

    plt.subplot(1, 3, 3)
    sns.histplot(data=rfm_pd, x='Monetary', bins=30, kde=True)
    plt.title('Distribution of Monetary')

    plt.tight_layout()
    plt.show()
    print("### Interpretation : ")
    print(". Recency : Très skewée droite, majorité clients récents (pic ~0-50 jours), queue longue d'inactifs → beaucoup de clients récents ou one-time.")
    print(". Frequency : Fortement skewée, pic élevé à faible fréquence (1-10 achats), très peu de clients fidèles → plupart achètent rarement.")
    print(". Monetary : Extrêmement skewée, pic à bas montants, longue queue (quelques gros dépensiers) → valeur client concentrée sur minorité.")

    print("Matrice de confusion")
    # Correlation matrix
    corr_matrix = rfm_pd[['Recency', 'Frequency', 'Monetary']].corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('RFM Correlation Matrix')
    plt.show()

def Log_transformation(rfm):

    # Apply log transformation to the Frequency and Monetary values
    rfm = rfm.withColumn("Log_Frequency", log(rfm["Frequency"] + 1)) \
            .withColumn("Log_Monetary", log(rfm["Monetary"] + 1))

    # Calculate RFM score (sum of Recency + Log_Frequency + Log_Monetary)
    rfm = rfm.withColumn("RFM_Score", col("Recency") + col("Log_Frequency") + col("Log_Monetary"))

    print("Shape of the DataFrame:", (rfm.count(), len(rfm.columns)))
    rfm.show(5)

    return rfm

def VectorAssembler_Transformation(rfm):
    # Assemble features for clustering
    assembler = VectorAssembler(inputCols=["Recency", "Log_Frequency", "Log_Monetary", "RFM_Score"],
                                    outputCol="features",
                                    handleInvalid="skip")
    rfm_features = assembler.transform(rfm)

    # Scale the features
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    rfm_scaled = scaler.fit(rfm_features).transform(rfm_features)
    rfm_scaled.select("scaledFeatures").show(5, truncate=False)
    return rfm_scaled