from visualization.visualize import *
from Data_Cleaning.data_cleaning import *
from features_engineering.build_features import *
from customer_segmentation.customer_segmentation import *
from models.predict_model import *
from pathlib import Path


print("##################################################### VISUALISATION   #####################################################")
df = data_load()

visualize(df)

print("################################################ DATA_CLEANING   ################################################")

# Imputation des valeurs manquantes

mode= mode_description(df)

df_clean = imputation_valeurs_manquante(df)

# Enrichissement de la colonne InvoiceDate

df_clean= enrichissement_InvoiceDate(df_clean)

df_clean.printSchema()
df_clean.show(5)
#Enregistrer la dataframe sous forme parquet
BASE_DIR = Path(__file__).resolve().parent.parent
output_path = str(BASE_DIR / "data" / "processed" / "online_retail_cleaned")
#df_clean.write.mode("overwrite").parquet(output_path)


print("Nombre de lignes après nettoyage :", df_clean.count())

print("###################################### FEATURES_ENGINEERING   ##################################################")


df_cleaned = read_parquet(output_path)


reference_date = df_cleaned.select(max("InvoiceDate")).first()[0]
df_cleaned.printSchema()
print("la date de reference:", reference_date)

rfm= rfm(df_cleaned, reference_date)

# visualisation de la distribution des RFM
rfm_visualization(rfm)

# appliquant le log transformation pour reduire les valeurs extremes
rfm = Log_transformation(rfm)

rfm_scaled = VectorAssembler_Transformation(rfm)

#Enregistrer la dataframe sous forme parquet

#output_path_rfmScaled = "/Users/yassineoc/Desktop/hamza/Projet_Spark/Customer Segmentation and Churn Prediction with PySpark/data/processed/rfm_scaled"
#rfm_scaled.write.mode("overwrite").parquet(output_path_rfmScaled)

# Evaluate KMeans for different k
k_values = range(2, 11)
metrics_df = evaluate_kmeans(rfm_scaled, k_values)
metrics_df = pd.DataFrame(metrics_df)  

plot_kmeans_metrics(metrics_df)
# Visualize evaluation metrics
plot_kmeans_metrics(metrics_df)

# Select optimal k (from analysis)
best_k = 3
print("Optimal number of clusters:", best_k)

# Train final model
model, rfm_clusters = train_kmeans(rfm_scaled, best_k)

# Analyze clusters
mean_rfm = compute_mean_rfm_per_cluster(rfm_clusters)

# Label customers
rfm_labeled = label_customers(rfm_clusters, mean_rfm)

# Show sample output
rfm_labeled.select(
    "CustomerID", "RFM_Score", "cluster", "Customer_Type"
).show(10)

rfm_clusters_path = str(BASE_DIR / "data" / "processed" / "rfm_clusters"/"*")

rfm_clusterss = read_parquet(rfm_clusters_path)

rfm_clusterss.printSchema()


rfm_clusterss = create_churn_column(rfm_clusterss)


# 4️⃣ Visualiser les corrélations des features

plot_feature_correlation(rfm_clusterss)


# 5️⃣ Préparer train/test pour les modèles

train_data, test_data = prepare_train_test(rfm_clusterss)


# 6️⃣ Random Forest Classifier

best_rf_model, rf_predictions, rf_auc, rf_report, rf_conf_matrix = train_random_forest(train_data, test_data)
print(f"Random Forest AUC: {rf_auc:.4f}")
print("Random Forest Classification Report:\n", rf_report)
plot_confusion_matrix(rf_conf_matrix,parameter="Random Forest Classifier")


# 7️⃣ Gradient Boosted Tree Classifier

best_gbt_model, gbt_predictions, gbt_auc, gbt_report, gbt_conf_matrix = train_gradient_boosted_tree(train_data, test_data)
print(f"GBT Classifier AUC: {gbt_auc:.4f}")
print("GBT Classification Report:\n", gbt_report)
plot_confusion_matrix(gbt_conf_matrix,parameter="Gradient Boosted Tree Classifier")


# 8️⃣ Logistic Regression avec pipeline

best_lr_model, lr_predictions, lr_auc, lr_auc_sklearn, lr_report, lr_cm = train_logistic_regression(rfm_clusterss)
print(f"Logistic Regression AUC (Spark): {lr_auc:.4f}")
print(f"Logistic Regression AUC (sklearn): {lr_auc_sklearn:.4f}")
print("Logistic Regression Classification Report:\n", lr_report)
plot_sklearn_confusion_matrix(lr_predictions)
plot_probability_distribution(lr_predictions)


print("✅ Tous les modèles ont été entraînés et évalués avec succès !")


