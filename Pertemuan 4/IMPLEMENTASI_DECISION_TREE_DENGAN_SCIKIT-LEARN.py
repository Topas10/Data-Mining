import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Load dataset (contoh: Iris)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Buat model Decision Tree
dt = DecisionTreeClassifier(
    criterion='gini',      # 'gini' atau 'entropy'
    max_depth=3,           # batasi kedalaman untuk hindari overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Training
dt.fit(X_train, y_train)

# Prediksi
y_pred = dt.predict(X_test)

# Evaluasi
print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))