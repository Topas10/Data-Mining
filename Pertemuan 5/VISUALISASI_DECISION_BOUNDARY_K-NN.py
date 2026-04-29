import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler # Tambahan agar tidak error
from matplotlib.colors import ListedColormap

# 1. Generate dataset 2D untuk visualisasi
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# 2. Normalisasi
X = StandardScaler().fit_transform(X)

# 3. Buat model untuk berbagai nilai k
k_values = [1, 5, 15, 30]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

# Warna untuk plot
cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_points = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 4. Loop untuk melakukan fitting dan plotting berdasarkan nilai k
for idx, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Menggambar background warna
    axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap=cmap_background)
    
    # Menggambar titik data (scatter plot)
    axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolor='black', s=30)
    
    axes[idx].set_title(f'k-NN Decision Boundary (k={k})')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Observasi (berdasarkan catatan di gambar):
# - k=1:  boundary sangat kompleks (overfitting)
# - k=5:  boundary lebih halus
# - k=15: semakin smooth
# - k=30: terlalu smooth (underfitting)