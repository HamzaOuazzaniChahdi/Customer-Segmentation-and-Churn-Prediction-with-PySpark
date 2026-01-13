from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.sql.functions import col, isnan, count, when

from config.spark_session import get_spark_session

spark = get_spark_session()


def data_load(data_path):

    df = spark.read.csv(
        data_path,
        header=True,
        inferSchema=True
    )

    df.printSchema()
    return df

def visualize(df):


    print("#################### Répartition des types de variables ############# ")
    schema_df = pd.DataFrame(
    [(field.name, field.dataType.simpleString()) for field in df.schema.fields],
    columns=["Variable", "Type"]
    )

    print(schema_df)

    schema_df["Type"].value_counts().plot(kind="bar")
    plt.title("Répartition des types de variables")
    plt.show()



    print("######################## Valeurs manquantes ###########################")
    n_rows = df.count()
    n_cols = len(df.columns)

    print(f"Nombre de lignes : {n_rows}")
    print(f"Nombre de colonnes : {n_cols}")
    #Analyse des valeurs manquantes
    missing_df = df.select([
        count(
            when(col(c).isNull() | isnan(col(c)), c)
        ).alias(c)
        for c in df.columns
    ])

    missing_pd = missing_df.toPandas().T
    missing_pd.columns = ["Missing_Count"]
    missing_pd["Missing_%"] = (missing_pd["Missing_Count"] / n_rows) * 100
    missing_pd.sort_values("Missing_%", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=missing_pd["Missing_%"].sort_values(ascending=False).head(10),
        y=missing_pd.sort_values("Missing_%", ascending=False).head(10).index
    )
    plt.title("Top 10 variables avec valeurs manquantes")
    plt.xlabel("Pourcentage de valeurs manquantes")
    plt.show()

    print("########################### Matrice de confusion ###########################")
    # Matrice de correlation
    #Échantillonnage contrôlé pour Pandas

    sample_df = df.sample(fraction=0.1, seed=42).toPandas()
    #Distribution des variables numériques
    numeric_cols = sample_df.select_dtypes(include=np.number).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(sample_df[numeric_cols].corr(), cmap="coolwarm")
    plt.title("Matrice de corrélation")
    plt.show()
    print("###################### Interpretation : ##########################")
    print("Il n’existe aucune corrélation linéaire significative entre les variables quantitatives étudiées.CustomerID n’apporte pas d’information analytique et doit être exclu des corrélations.")