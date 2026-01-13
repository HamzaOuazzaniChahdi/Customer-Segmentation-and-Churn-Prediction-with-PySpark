# Customer Segmentation and Churn Prediction with PySpark

## 📌 Présentation du projet
Ce projet de **Big Data Analytics** vise à exploiter **PySpark** pour analyser des données transactionnelles à grande échelle afin de :

- **Segmenter les clients** selon leur comportement d’achat à l’aide de l’approche **RFM** (Récence, Fréquence, Montant).
- **Prédire le churn client** (risque de désengagement) à l’aide de modèles de **Machine Learning supervisés**.

L’objectif est de fournir une aide à la décision permettant aux entreprises d’optimiser leurs stratégies de **fidélisation**, de **marketing ciblé** et de **gestion de la relation client**.

---

## 🗂️ Jeu de données utilisé
- **Nom** : Online Retail Dataset  
- **Source** : UCI Machine Learning Repository / Kaggle  
- **Période** : Décembre 2010 – Décembre 2011  
- **Taille** : 541 909 transactions, 8 variables  

### Variables principales
- `InvoiceNo` : Identifiant de la facture  
- `StockCode` : Code produit  
- `Description` : Description du produit  
- `Quantity` : Quantité achetée  
- `InvoiceDate` : Date et heure de la transaction  
- `UnitPrice` : Prix unitaire  
- `CustomerID` : Identifiant unique du client  
- `Country` : Pays du client  

---

## 🔧 Étapes principales du projet

### 1️⃣ Prétraitement des données
- Imputation des valeurs manquantes, enrichissement temporel

### 2️⃣ Feature Engineering – Métriques RFM
- **Recency** : Temps écoulé depuis le dernier achat
- **Frequency** : Nombre de transactions distinctes
- **Monetary** : Montant total dépensé


---

### 3️⃣ Segmentation client avec K-Means
- Normalisation des variables RFM
- Sélection du nombre optimal de clusters :
- Méthode du **coude (WCSS)**
- **Score de silhouette**


---

### 4️⃣ Prédiction du churn client
- Variables explicatives :
- Métriques RFM
- Cluster K-Means
- Modèles testés :
- Régression Logistique
- Random Forest
- Gradient Boosting Machine (GBM)

#### Métriques d’évaluation
- Accuracy
- Precision
- Recall
- F1-score
- AUC (ROC)


---


## 🛠️ Technologies utilisées
- **PySpark** : Traitement distribué des données
- **Python** : Développement et orchestration
- **Spark MLlib** : Clustering et classification
- **Pandas / NumPy / Matplotlib / Seaborn** : Analyse et visualisation
- **UCI / Kaggle** : Source de données

---

## 🚀 Instructions pour exécuter le projet


## Cloner le dépôt
```bash
git clone https://github.com/HamzaOuazzaniChahdi/Customer-Segmentation-and-Churn-Prediction-with-PySpark.git
cd Customer-Segmentation-and-Churn-Prediction-with-PySpark
```
```bash
# Créer un environnement virtuel
python -m venv venv
# Activation
venv\Scripts\activate      #Windows
source venv/bin/activate   # Linux/mac
```

```bash
# Installer les dépendances
pip install -r requirements.txt
# Lancer le projet
python src/main.py
