#!/bin/bash

echo "📦 Création de l'environnement virtuel..."
python3 -m venv OUAZZANI.C_HAMZA


# Activer le venv
source venv/bin/activate

echo "📥 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Lancement du projet..."
python src/main.py
