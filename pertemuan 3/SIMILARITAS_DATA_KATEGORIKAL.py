import numpy as np
from sklearn.metrics import jaccard_score

def simple_matching_coefficient(p, q):
    """Menghitung SMC untuk dua vektor biner"""
    p = np.array(p)
    q = np.array(q)

    m11 = np.sum((p == 1) & (q == 1))
    m00 = np.sum((p == 0) & (q == 0))
    total = len(p)
    return (m11 + m00) / total

# Data contoh
p = [1, 0, 1, 1, 0]
q = [1, 0, 0, 1, 1]

# Hitung SMC
smc = simple_matching_coefficient(p, q)
print(f"SMC: {smc:.2f}")  # 0.60

# Hitung Jaccard (hanya kehadiran)
# M11 = 2, M10 = 1, M01 = 1
m11 = np.sum((np.array(p) == 1) & (np.array(q) == 1))
m10 = np.sum((np.array(p) == 1) & (np.array(q) == 0))
m01 = np.sum((np.array(p) == 0) & (np.array(q) == 1))

jaccard = m11 / (m11 + m10 + m01)
print(f"Jaccard (manual): {jaccard:.2f}")  # 0.50

# Alternatif dengan scikit-learn (average='binary')
jaccard_sklearn = jaccard_score(p, q, average='binary')
print(f"Jaccard (sklearn): {jaccard_sklearn:.2f}")  # 0.50