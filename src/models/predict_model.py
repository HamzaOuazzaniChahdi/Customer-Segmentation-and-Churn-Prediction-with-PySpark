from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


from config.spark_session import get_spark_session, data_load

spark = get_spark_session()

def create_churn_column(rfm_clusters):
    """Créer la colonne Churn à partir de Customer_Type"""
    def create_churn_dummy(Customer_Type):
        return 1 if Customer_Type == 'Dormant' else 0

    churn_udf = F.udf(create_churn_dummy, IntegerType())
    return rfm_clusters.withColumn("Churn", churn_udf(rfm_clusters["Customer_Type"]))


def prepare_train_test(rfm_clusters, feature_col="scaledFeatures"):
    """Préparer les features et split train/test"""
    df = rfm_clusters.select(feature_col, "Churn")
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    return train_data, test_data


def train_random_forest(train_data, test_data, num_trees=[10,15], max_depth=[3,5]):
    """Entraîner un Random Forest avec CrossValidation"""
    rf_classifier = RandomForestClassifier(labelCol="Churn", featuresCol="scaledFeatures")
    param_grid = (ParamGridBuilder()
                  .addGrid(rf_classifier.numTrees, num_trees)
                  .addGrid(rf_classifier.maxDepth, max_depth)
                  .build())
    evaluator = BinaryClassificationEvaluator(labelCol="Churn")
    crossval = CrossValidator(estimator=rf_classifier,
                              estimatorParamMaps=param_grid,
                              evaluator=evaluator,
                              numFolds=3)
    cv_rf_model = crossval.fit(train_data)
    best_rf_model = cv_rf_model.bestModel
    predictions = best_rf_model.transform(test_data)

    # Metrics
    auc = evaluator.evaluate(predictions)
    predictions_and_labels = predictions.select("prediction", "Churn").rdd.map(lambda row: (float(row.prediction), float(row.Churn)))
    predicted_labels = [int(x[0]) for x in predictions_and_labels.collect()]
    true_labels = [int(x[1]) for x in predictions_and_labels.collect()]
    report = classification_report(true_labels, predicted_labels, target_names=["Active", "Churned"])
    conf_matrix = MulticlassMetrics(predictions_and_labels).confusionMatrix().toArray()

    return best_rf_model, predictions, auc, report, conf_matrix


def train_gradient_boosted_tree(train_data, test_data, max_iter=[5,10], max_depth=[3,5], step_size=[0.1]):
    """Entraîner un GBTClassifier avec CrossValidation"""
    gbt_classifier = GBTClassifier(labelCol="Churn", featuresCol="scaledFeatures")
    param_grid = (ParamGridBuilder()
                  .addGrid(gbt_classifier.maxIter, max_iter)
                  .addGrid(gbt_classifier.maxDepth, max_depth)
                  .addGrid(gbt_classifier.stepSize, step_size)
                  .build())
    evaluator = BinaryClassificationEvaluator(labelCol="Churn")
    crossval = CrossValidator(estimator=gbt_classifier,
                              estimatorParamMaps=param_grid,
                              evaluator=evaluator,
                              numFolds=3)
    cv_gbt_model = crossval.fit(train_data)
    best_gbt_model = cv_gbt_model.bestModel
    predictions = best_gbt_model.transform(test_data)

    # Metrics
    auc = evaluator.evaluate(predictions)
    predictions_and_labels = predictions.select("prediction", "Churn").rdd.map(lambda row: (float(row.prediction), float(row.Churn)))
    predicted_labels = [int(x[0]) for x in predictions_and_labels.collect()]
    true_labels = [int(x[1]) for x in predictions_and_labels.collect()]
    report = classification_report(true_labels, predicted_labels, target_names=["Active", "Churned"])
    conf_matrix = MulticlassMetrics(predictions_and_labels).confusionMatrix().toArray()

    return best_gbt_model, predictions, auc, report, conf_matrix


def train_logistic_regression(rfm_clusters, feature_cols=None):
    """Entraîner une régression logistique avec pipeline et CrossValidation"""
    if feature_cols is None:
        feature_cols = ["Recency", "Log_Frequency", "Log_Monetary", "RFM_Score", "cluster"]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
    scaler = StandardScaler(inputCol="features_vec", outputCol="scaledFeatures_lr")
    lr = LogisticRegression(featuresCol="scaledFeatures_lr", labelCol="Churn", maxIter=50)
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    evaluator = BinaryClassificationEvaluator(labelCol="Churn", metricName="areaUnderROC")
    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.01, 0.1, 0.5])
                 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                 .build())

    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=5,
                        parallelism=2)

    train_df, test_df = rfm_clusters.randomSplit([0.8, 0.2], seed=42)
    cv_model = cv.fit(train_df)
    predictions = cv_model.transform(test_df)
    auc = evaluator.evaluate(predictions)
    best_lr_model = cv_model.bestModel.stages[-1]

    # Convertir en Pandas pour sklearn metrics
    pred_pd = predictions.select("Churn", "prediction", "probability").toPandas()
    pred_pd['prob_churn'] = pred_pd['probability'].apply(lambda x: x[1])
    report = classification_report(pred_pd['Churn'], pred_pd['prediction'], target_names=['Active','Churned'])
    auc_sklearn = roc_auc_score(pred_pd['Churn'], pred_pd['prob_churn'])
    cm = confusion_matrix(pred_pd['Churn'], pred_pd['prediction'])

    return best_lr_model, predictions, auc, auc_sklearn, report, cm


def plot_confusion_matrix(conf_matrix, labels=["Active","Churned"], parameter=""):
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt=".4f", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f"Confusion Matrix - {parameter}")
    plt.show()


def plot_feature_correlation(rfm_clusters, columns=None):
    if columns is None:
        columns = ["Recency", "Frequency", "Monetary", "Log_Frequency", "Log_Monetary", "RFM_Score"]
    df = rfm_clusters.select(columns).toPandas()
    correlation_matrix = df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Features")
    plt.show()


def plot_probability_distribution(predictions):
    pred_pd = predictions.select("Churn", "prediction", "probability").sample(fraction=0.8, seed=42).toPandas()
    pred_pd['prob_churn'] = pred_pd['probability'].apply(lambda x: x[1])
    plt.figure(figsize=(12,5))
    sns.histplot(data=pred_pd, x='prob_churn', bins=30, hue='Churn', multiple='stack')
    plt.title("Distribution des probabilités prédites par classe")
    plt.xlabel("Probabilité de churn")
    plt.ylabel("Nombre de clients")
    plt.show()

def plot_sklearn_confusion_matrix(predictions):
    pred_pd = predictions.select("Churn", "prediction").toPandas()
    cm = confusion_matrix(pred_pd['Churn'], pred_pd['prediction'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion")
    plt.show()




