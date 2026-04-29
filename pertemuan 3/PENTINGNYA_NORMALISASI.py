import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Data dengan skala berbeda
data = np.array([[2, 2000000],
                 [5, 7000000],
                 [3, 5000000],
                 [8, 12000000]])

# Min-Max Normalization
scaler_minmax = MinMaxScaler()
data_minmax = scaler_minmax.fit_transform(data)

# Z-Score Normalization
scaler_zscore = StandardScaler()
data_zscore = scaler_zscore.fit_transform(data)

print("Data Asli:")
print(data)
print("\nSetelah Min-Max:")
print(data_minmax)
print("\nSetelah Z-Score:")
print(data_zscore)

# Hitung jarak Euclidean setelah normalisasi
from scipy.spatial.distance import euclidean

print(f"\nJarak Euclidean (data asli): {euclidean(data[0], data[1]):.2f}")
print(f"Jarak Euclidean (Min-Max): {euclidean(data_minmax[0], data_minmax[1]):.2f}")