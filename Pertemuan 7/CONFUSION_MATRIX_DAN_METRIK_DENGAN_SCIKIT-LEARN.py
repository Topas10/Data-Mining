import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,
    precision_score, recall_score,
                             f1_score,
    confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate dataset imbalanced
X, y = make_classification(n_samples=1000, n_features=20,
    weights=[0.9, 0.1],  # 90% kelas 0, 10% kelas 1
    random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Hitung metrik
print("=" * 50)
print("EVALUASI MODEL")
print("=" * 50)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("(Format: [[TN, FP], [FN, TP]])")

# Classification Report (lengkap)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
    target_names=['Class 0', 'Class 1']))

# Interpretasi manual dari CM
tn, fp, fn, tp = cm.ravel()
print("\nInterpretasi:")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")