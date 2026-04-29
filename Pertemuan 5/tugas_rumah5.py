import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# ============================================================
#  BAGIAN 1: k-NN dari Scratch
# ============================================================

class KNNScratch:
    """
    k-Nearest Neighbors Classifier implementasi dari scratch.
    Menggunakan Euclidean Distance untuk menghitung jarak antar sampel.
    """

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_one(self, x):
        # Hitung jarak ke semua data training
        distances = [self._euclidean(x, xt) for xt in self.X_train]
        # Ambil indeks k tetangga terdekat
        k_indices = np.argsort(distances)[: self.k]
        # Ambil label k tetangga
        k_labels = self.y_train[k_indices]
        # Voting mayoritas
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in np.array(X)])

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ============================================================
#  BAGIAN 2: Persiapan Data
# ============================================================

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("  k-NN: SCRATCH vs SCIKIT-LEARN  |  Dataset Iris")
print("=" * 60)
print(f"  Train  : {len(X_train)} sampel")
print(f"  Test   : {len(X_test)} sampel")
print(f"  Fitur  : {iris.feature_names}")
print(f"  Kelas  : {list(iris.target_names)}")

# ============================================================
#  BAGIAN 3: Eksperimen Berbagai Nilai k
# ============================================================

k_values = [1, 3, 5, 7, 9, 11]
results = []

print(f"\n{'='*60}")
print(f"  {'k':>3} | {'Scratch':>10} | {'Sklearn':>10} | {'Selisih':>8} | {'t Scratch':>10} | {'t Sklearn':>10}")
print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

for k in k_values:
    # Scratch
    t0 = time.perf_counter()
    knn_s = KNNScratch(k=k)
    knn_s.fit(X_train, y_train)
    pred_s = knn_s.predict(X_test)
    t1 = time.perf_counter()
    acc_s  = accuracy_score(y_test, pred_s)
    time_s = (t1 - t0) * 1000

    # Scikit-Learn
    t0 = time.perf_counter()
    knn_sk = KNeighborsClassifier(n_neighbors=k)
    knn_sk.fit(X_train, y_train)
    pred_sk = knn_sk.predict(X_test)
    t1 = time.perf_counter()
    acc_sk  = accuracy_score(y_test, pred_sk)
    time_sk = (t1 - t0) * 1000

    delta = abs(acc_s - acc_sk) * 100
    results.append((k, acc_s, acc_sk, delta, time_s, time_sk, pred_s, pred_sk))

    print(f"  {k:>3} | {acc_s*100:>9.2f}% | {acc_sk*100:>9.2f}% | {delta:>7.2f}% | {time_s:>8.2f} ms | {time_sk:>8.2f} ms")

# ============================================================
#  BAGIAN 4: Laporan Detail per k
# ============================================================

for k, acc_s, acc_sk, delta, time_s, time_sk, pred_s, pred_sk in results:
    print(f"\n{'='*60}")
    print(f"  k = {k}")
    print(f"{'='*60}")

    for label, pred, acc, elapsed in [
        ("SCRATCH",      pred_s,  acc_s,  time_s),
        ("SCIKIT-LEARN", pred_sk, acc_sk, time_sk),
    ]:
        print(f"\n  [{label}]  Akurasi: {acc*100:.2f}%  |  Waktu: {elapsed:.2f} ms")
        print(classification_report(
            y_test, pred,
            target_names=iris.target_names,
            digits=4,
            zero_division=0
        ))
        print("  Confusion Matrix:")
        print(confusion_matrix(y_test, pred))

# ============================================================
#  BAGIAN 5: Analisis Pengaruh Nilai k
# ============================================================

print(f"\n{'='*60}")
print("  ANALISIS PENGARUH NILAI k TERHADAP AKURASI")
print(f"{'='*60}")

best_k_s  = max(results, key=lambda r: r[1])
best_k_sk = max(results, key=lambda r: r[2])

print(f"\n  k terbaik (Scratch)     : k={best_k_s[0]}  -> Akurasi {best_k_s[1]*100:.2f}%")
print(f"  k terbaik (Scikit-Learn): k={best_k_sk[0]}  -> Akurasi {best_k_sk[2]*100:.2f}%")

print("""
  Kesimpulan Analisis:
  ┌─────────────────────────────────────────────────────┐
  │  k kecil (k=1)  : Model sangat sensitif terhadap   │
  │                   noise (overfitting).              │
  │  k sedang (3-7) : Keseimbangan bias-variance yang  │
  │                   baik, akurasi umumnya tertinggi.  │
  │  k besar (9-11) : Model mulai underfitting karena  │
  │                   batas keputusan terlalu halus.   │
  │                                                     │
  │  Scratch vs Sklearn: Akurasi identik, sklearn       │
  │  lebih cepat karena optimasi C/Cython di baliknya. │
  └─────────────────────────────────────────────────────┘
""")