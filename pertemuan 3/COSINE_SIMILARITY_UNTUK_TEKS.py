from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Kumpulan dokumen
documents = [
    "data mining machine learning",
    "machine learning data mining",
    "artificial intelligence",
    "deep learning neural network"
]

# Konversi ke vektor TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Hitung cosine similarity antara semua dokumen
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Cosine Similarity Matrix:")
print(similarity_matrix.round(3))
print("\nFitur (kata-kata):")
print(vectorizer.get_feature_names_out())

# Cari dokumen paling mirip dengan dokumen pertama
print(f"\nDokumen 0 paling mirip dengan dokumen {similarity_matrix[0].argsort()[-2]}")