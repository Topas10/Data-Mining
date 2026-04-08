import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock, minkowski
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# Load dan normalisasi dataset Iris
iris      = load_iris()
iris_data = iris.data
labels    = iris.target
feature_names = iris.feature_names

scaler = MinMaxScaler()
data   = scaler.fit_transform(iris_data)

print("Jumlah data :", data.shape[0])
print("Jumlah fitur:", data.shape[1])
print("Fitur       :", feature_names)
# Implementasi fungsi untuk menghitung ketiga jenis jarak
def hitung_euclidean(a, b):
    """Jarak Euclidean: akar dari jumlah kuadrat selisih"""
    return euclidean(a, b)

def hitung_manhattan(a, b):
    """Jarak Manhattan: jumlah nilai absolut selisih"""
    return cityblock(a, b)

def hitung_minkowski(a, b, p=3):
    """Jarak Minkowski: generalisasi Euclidean & Manhattan"""
    return minkowski(a, b, p=p)

# Fungsi untuk membangun matriks jarak
def bangun_matriks_jarak(data, jenis='euclidean'):
    n = len(data)
    matriks = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if jenis == 'euclidean':
                matriks[i][j] = hitung_euclidean(data[i], data[j])
            elif jenis == 'manhattan':
                matriks[i][j] = hitung_manhattan(data[i], data[j])
            elif jenis == 'minkowski':
                matriks[i][j] = hitung_minkowski(data[i], data[j])
    return matriks

# Bangun ketiga matriks jarak
mat_euc = bangun_matriks_jarak(data, 'euclidean')
mat_man = bangun_matriks_jarak(data, 'manhattan')
mat_mink= bangun_matriks_jarak(data, 'minkowski')

print("\nMatriks Euclidean (5x5):")
print(pd.DataFrame(mat_euc[:5,:5]).round(4))
print("\nMatriks Manhattan (5x5):")
print(pd.DataFrame(mat_man[:5,:5]).round(4))
print("\nMatriks Minkowski p=3 (5x5):")
print(pd.DataFrame(mat_mink[:5,:5]).round(4))
# Pengelompokan sederhana: assign setiap data ke centroid terdekat
def kelompokkan(data, labels_asli, matriks_jarak, nama_jarak):
    # Hitung centroid per kelas dari data asli
    centroids = []
    for kelas in [0, 1, 2]:
        idx = np.where(labels_asli == kelas)[0]
        centroids.append(data[idx].mean(axis=0))
    centroids = np.array(centroids)

    # Assign setiap data ke centroid dengan jarak terkecil
    prediksi = []
    for i in range(len(data)):
        jarak_ke_centroid = [
            hitung_euclidean(data[i], centroids[0]),
            hitung_euclidean(data[i], centroids[1]),
            hitung_euclidean(data[i], centroids[2])
        ]
        prediksi.append(np.argmin(jarak_ke_centroid))
    prediksi = np.array(prediksi)

    # Hitung akurasi sederhana
    akurasi = np.mean(prediksi == labels_asli) * 100
    print(f"\n[{nama_jarak}] Akurasi pengelompokan : {akurasi:.2f}%")

    # Tampilkan distribusi per kelas
    for kelas in [0, 1, 2]:
        idx = np.where(labels_asli == kelas)[0]
        benar = np.sum(prediksi[idx] == kelas)
        print(f"  Kelas {kelas} ({iris.target_names[kelas]:12s}): {benar}/50 benar")
    return prediksi

print("== PERBANDINGAN PENGELOMPOKAN SEDERHANA ==")
pred_euc  = kelompokkan(data, labels, mat_euc,  "Euclidean")
pred_man  = kelompokkan(data, labels, mat_man,  "Manhattan")
pred_mink = kelompokkan(data, labels, mat_mink, "Minkowski")
# Ringkasan perbandingan ketiga jarak
hasil = {
    'Euclidean'  : np.mean(pred_euc  == labels) * 100,
    'Manhattan'  : np.mean(pred_man  == labels) * 100,
    'Minkowski'  : np.mean(pred_mink == labels) * 100,
}

df_hasil = pd.DataFrame(
    list(hasil.items()),
    columns=['Jenis Jarak', 'Akurasi (%)']
).sort_values('Akurasi (%)', ascending=False)

print("\n== RANGKUMAN PERBANDINGAN ==")
print(df_hasil.to_string(index=False))

terbaik = df_hasil.iloc[0]['Jenis Jarak']
print(f"\nJarak terbaik untuk Iris : {terbaik}")