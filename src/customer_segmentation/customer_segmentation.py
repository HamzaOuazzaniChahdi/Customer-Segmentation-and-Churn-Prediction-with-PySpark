# Grid Search or Random Search for optimizing k in K-Means
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import numpy as np
from pyspark.ml.clustering import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors

from config.spark_session import get_spark_session, data_load

spark = get_spark_session()

def read_parquet(path):
    rfm_scaled = spark.read.parquet(path)
    return rfm_scaled

# Function to calculate Euclidean distance squared
def squared_euclidean_distance(point, center):
    return float(np.sum((np.array(point) - np.array(center)) ** 2))

def evaluate_kmeans(rfm_scaled, k_values):
    results = []

    for k in k_values:
        kmeans = KMeans(
            k=k,
            seed=42,
            featuresCol="scaledFeatures",
            predictionCol="cluster"
        )

        model = kmeans.fit(rfm_scaled)
        clustered_df = model.transform(rfm_scaled)

        # ✅ WCSS natif Spark
        wcss = model.summary.trainingCost

        evaluator = ClusteringEvaluator(
            featuresCol="scaledFeatures",
            predictionCol="cluster",
            metricName="silhouette"
        )

        silhouette = evaluator.evaluate(clustered_df)

        results.append({
            "k": k,
            "WCSS": wcss,
            "SilhouetteScore": silhouette
        })

    return results


def train_kmeans(rfm_scaled, best_k): 
    """ Train final KMeans model """ 
    kmeans = KMeans( k=best_k, seed=42, featuresCol="scaledFeatures", predictionCol="cluster" )
    model = kmeans.fit(rfm_scaled) 
    clustered_df = model.transform(rfm_scaled) 
    return model, clustered_df



def compute_mean_rfm_per_cluster(rfm_clusters):
    """
    Compute mean RFM metrics per cluster
    """
    mean_rfm = rfm_clusters.groupBy("cluster").agg(
        F.avg("Recency").alias("Mean_Recency"),
        F.avg("Log_Frequency").alias("Mean_Frequency"),
        F.avg("Log_Monetary").alias("Mean_Monetary"),
        F.avg("RFM_Score").alias("Mean_RFM_Score")
    )

    return mean_rfm


def label_customers(rfm_clusters, mean_rfm_per_cluster):

    rfm_clusters = rfm_clusters.join(
            mean_rfm_per_cluster,
            on="cluster",
            how="left"
        )

        # Ajouter la colonne Customer_Type directement à rfm_clusters
    rfm_clusters = rfm_clusters.withColumn(
            "Customer_Type",
            F.when(
                F.col("RFM_Score") > F.col("Mean_RFM_Score") * 1.2, "Elite"
            ).when(
                F.col("RFM_Score") < F.col("Mean_RFM_Score") * 0.8, "Dormant"
            ).otherwise("Standard")
        )

    return rfm_clusters







def plot_kmeans_metrics(metrics_df):
    """
    Plot WCSS and Silhouette Score
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.lineplot(x="k", y="WCSS", data=metrics_df, ax=ax1, marker="o")
    ax1.set_title("WCSS for Different k Values")

    sns.lineplot(x="k", y="SilhouetteScore", data=metrics_df, ax=ax2, marker="o")
    ax2.set_title("Silhouette Score for Different k Values")

    plt.tight_layout()
    plt.show()
