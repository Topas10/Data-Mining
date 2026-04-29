import numpy as np
from scipy.spatial.distance import euclidean, cityblock, minkowski

# Data contoh
A = np.array([2, 3])
B = np.array([5, 7])

# Perhitungan berbagai jarak
euc_dist = euclidean(A, B)
man_dist = cityblock(A, B)
min_dist = minkowski(A, B, p=3) # r=3
print(f"Euclidean Distance: {euc_dist:.2f}")
print(f"Manhattan Distance: {man_dist:.2f}")
print(f"Minkowski (r=3): {min_dist:.2f}")

# Perbandingan dengan data berskala berbeda
A_scaled = np.array([2, 2000000])
B_scaled = np.array([5, 7000000])

print("\nTanpa normalisasi:")
print(f"Euclidean: {euclidean(A_scaled, B_scaled):.2f} ")
# Output akan didominasi oleh fitur pendapatan!