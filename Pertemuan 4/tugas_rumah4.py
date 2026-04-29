import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# ============================================================
#  BAGIAN 1: Decision Tree dari Scratch
# ============================================================

class Node:
    """Simpul (node) dalam pohon keputusan."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature   = feature    # indeks fitur untuk split
        self.threshold = threshold  # nilai threshold split
        self.left      = left       # subtree kiri (≤ threshold)
        self.right     = right      # subtree kanan (> threshold)
        self.value     = value      # label kelas (hanya untuk leaf node)

    def is_leaf(self):
        return self.value is not None


class DecisionTreeScratch:
    """
    Decision Tree Classifier implementasi dari scratch.
    Menggunakan kriteria Gini Impurity untuk memilih split terbaik.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.root              = None

    # ----- Kriteria Impurity -----

    def _gini(self, y):
        """Hitung Gini Impurity untuk sekumpulan label."""
        n = len(y)
        if n == 0:
            return 0.0
        counts = Counter(y)
        impurity = 1.0 - sum((c / n) ** 2 for c in counts.values())
        return impurity

    def _information_gain(self, y, y_left, y_right):
        """Hitung information gain dari sebuah split."""
        n = len(y)
        gain = self._gini(y) - (len(y_left) / n) * self._gini(y_left) \
                              - (len(y_right) / n) * self._gini(y_right)
        return gain

    # ----- Mencari Split Terbaik -----

    def _best_split(self, X, y):
        """Cari fitur & threshold yang memberikan information gain tertinggi."""
        best_gain      = -1
        best_feature   = None
        best_threshold = None

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                mask   = X[:, feature] <= threshold
                y_left  = y[mask]
                y_right = y[~mask]

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # ----- Membangun Pohon (Rekursif) -----

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Kondisi berhenti (stopping criteria)
        if (self.max_depth is not None and depth >= self.max_depth) \
           or n_classes == 1 \
           or n_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Cari split terbaik
        feature, threshold = self._best_split(X, y)

        if feature is None:                           # tidak ada split yang valid
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        mask  = X[:, feature] <= threshold
        left  = self._build_tree(X[mask],  y[mask],  depth + 1)
        right = self._build_tree(X[~mask], y[~mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    # ----- Prediksi -----

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    # ----- API Publik -----

    def fit(self, X, y):
        self.root = self._build_tree(np.array(X), np.array(y))
        return self

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in np.array(X)])

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    # ----- Visualisasi Teks -----

    def print_tree(self, node=None, depth=0, feature_names=None):
        if node is None:
            node = self.root
        indent = "    " * depth
        if node.is_leaf():
            print(f"{indent}→ Prediksi: Kelas {node.value}")
        else:
            fname = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
            print(f"{indent}[{fname} ≤ {node.threshold:.3f}]")
            print(f"{indent}  KIRI (Ya):")
            self.print_tree(node.left,  depth + 1, feature_names)
            print(f"{indent}  KANAN (Tidak):")
            self.print_tree(node.right, depth + 1, feature_names)


# ============================================================
#  BAGIAN 2: Persiapan Data
# ============================================================

def load_and_split():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, iris


# ============================================================
#  BAGIAN 3: Training & Evaluasi
# ============================================================

def evaluate(name, y_true, y_pred, elapsed, target_names):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Akurasi       : {acc*100:.2f}%")
    print(f"  Waktu training: {elapsed*1000:.2f} ms")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    return acc


def main():
    print("=" * 55)
    print("  DECISION TREE: SCRATCH vs SCIKIT-LEARN (Iris)")
    print("=" * 55)

    X_train, X_test, y_train, y_test, iris = load_and_split()

    print(f"\n  Dataset  : Iris")
    print(f"  Train    : {len(X_train)} sampel")
    print(f"  Test     : {len(X_test)} sampel")
    print(f"  Fitur    : {iris.feature_names}")
    print(f"  Kelas    : {list(iris.target_names)}")

    # ---- Model dari Scratch ----
    t0 = time.perf_counter()
    dt_scratch = DecisionTreeScratch(max_depth=4)
    dt_scratch.fit(X_train, y_train)
    t1 = time.perf_counter()
    y_pred_scratch = dt_scratch.predict(X_test)
    acc_scratch = evaluate("Decision Tree SCRATCH", y_test, y_pred_scratch,
                           t1 - t0, iris.target_names)

    # ---- Model Scikit-Learn ----
    t0 = time.perf_counter()
    dt_sklearn = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
    dt_sklearn.fit(X_train, y_train)
    t1 = time.perf_counter()
    y_pred_sklearn = dt_sklearn.predict(X_test)
    acc_sklearn = evaluate("Decision Tree SCIKIT-LEARN", y_test, y_pred_sklearn,
                           t1 - t0, iris.target_names)

    # ---- Ringkasan Perbandingan ----
    print(f"\n{'='*55}")
    print("  RINGKASAN PERBANDINGAN")
    print(f"{'='*55}")
    print(f"  Akurasi Scratch     : {acc_scratch*100:.2f}%")
    print(f"  Akurasi Scikit-Learn: {acc_sklearn*100:.2f}%")
    delta = abs(acc_scratch - acc_sklearn) * 100
    print(f"  Selisih Akurasi     : {delta:.2f}%")
    print()

    # ---- Visualisasi Pohon (Scratch) ----
    print("  Struktur Pohon (Scratch, 3 level pertama):")
    dt_scratch.print_tree(feature_names=iris.feature_names)

    print("\n  ✓ Selesai!\n")


if __name__ == "__main__":
    main()