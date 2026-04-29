from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Normalisasi SANGAT PENTING untuk k-NN!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42,
    stratify=y
)

# k-NN dengan k=5
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)
print(f"Akurasi (k=5): {accuracy_score(y_test, y_pred):.4f}")

# Tuning parameter k dengan cross-validation
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"\nParameter terbaik: {grid_search.best_params_}")
print(f"Akurasi CV terbaik: {grid_search.best_score_:.4f}")

# Evaluasi model terbaik
best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test)
print(f"Akurasi test set: {accuracy_score(y_test, y_pred_best):.4f}")