import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Single Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Bagging dengan Decision Tree
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',  # sqrt(p) untuk klasifikasi
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Perbandingan Akurasi
print("=" * 50)
print("Perbandingan Akurasi:")
print(f"Decision Tree : {accuracy_score(y_test, dt_pred):.4f}")
print(f"Bagging       : {accuracy_score(y_test, bagging_pred):.4f}")
print(f"Random Forest : {accuracy_score(y_test, rf_pred):.4f}")

# Feature Importance dari Random Forest
importance = pd.DataFrame({
    'feature': data.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Feature Importance:")
print(importance.head())