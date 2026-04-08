import numpy as np
from sklearn.metrics import jaccard_score

# Data soal nomor 2
X = [1, 0, 1, 0, 1, 1]
Y = [1, 1, 0, 0, 1, 0]

# SMC (Simple Matching Coefficient)
def simple_matching_coefficient(p, q):
    """Menghitung SMC untuk dua vektor biner"""
    p = np.array(p)
    q = np.array(q)
    m11 = np.sum((p == 1) & (q == 1))
    m00 = np.sum((p == 0) & (q == 0))
    total = len(p)
    return (m11 + m00) / total

smc = simple_matching_coefficient(X, Y)
print(f"SMC: {smc:.4f}")  # 0.5000

# Jaccard (hanya kehadiran)
p = np.array(X)
q = np.array(Y)

m11 = np.sum((p == 1) & (q == 1))
m10 = np.sum((p == 1) & (q == 0))
m01 = np.sum((p == 0) & (q == 1))

jaccard = m11 / (m11 + m10 + m01)
print(f"Jaccard (manual): {jaccard:.4f}")  # 0.4000

# Alternatif dengan scikit-learn
jaccard_sklearn = jaccard_score(X, Y, average='binary')
print(f"Jaccard (sklearn): {jaccard_sklearn:.4f}")  # 0.4000