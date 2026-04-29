import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# ======================
# Load Dataset
# ======================
iris = load_iris()
X = iris.data
y = iris.target

# ======================
# Split Data
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ======================
# Model Decision Tree
# ======================
dt = DecisionTreeClassifier(
    criterion='gini',      # bisa juga 'entropy'
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# ======================
# Training
# ======================
dt.fit(X_train, y_train)

# ======================
# Prediksi
# ======================
y_pred = dt.predict(X_test)

# ======================
# Evaluasi
# ======================
print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ======================
# Visualisasi Decision Tree
# ======================
plt.figure(figsize=(15, 10))
plot_tree(
    dt,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree - Iris Dataset")
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================
# Feature Importance
# ======================
importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance)