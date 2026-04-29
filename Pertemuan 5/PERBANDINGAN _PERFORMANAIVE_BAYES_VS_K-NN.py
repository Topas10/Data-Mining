from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd

# Generate berbagai jenis dataset
datasets = {
    'Linear': make_classification(n_samples=500, n_features=10,
                                  n_informative=8, random_state=42),
    'Moons': make_moons(n_samples=500, noise=0.1, random_state=42),
    'Circles': make_circles(n_samples=500, noise=0.05,
                            factor=0.5, random_state=42)
}

# Model
models = {
    'Naive Bayes': GaussianNB(),
    'k-NN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'k-NN (k=7)': KNeighborsClassifier(n_neighbors=7),
    'k-NN (k=15)': KNeighborsClassifier(n_neighbors=15)
}

# Bandingkan performa
results = []

for dataset_name, (X, y) in datasets.items():
    # Normalisasi untuk k-NN
    X_scaled = StandardScaler().fit_transform(X)

    for model_name, model in models.items():
        if 'k-NN' in model_name:
            X_use = X_scaled
        else:
            X_use = X  # NB tidak perlu normalisasi untuk data ini

        scores = cross_val_score(model, X_use, y, cv=5, scoring='accuracy')
        results.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Mean Accuracy': scores.mean(),
            'Std': scores.std()
        })

results_df = pd.DataFrame(results)
print(results_df.round(4))

# Pivot table untuk perbandingan mudah
pivot = results_df.pivot(index='Dataset', columns='Model', values='Mean Accuracy')
print("\nPerbandingan Akurasi Rata-rata:")
print(pivot.round(4))