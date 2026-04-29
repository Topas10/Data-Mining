from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

# Contoh data teks sederhana
emails = [
    "Free money click here win prize",
    "Meeting schedule for tomorrow",
    "Win lottery prize money free",
    "Project deadline extension request",
    "Click link to claim your prize",
    "Quarterly report attachment",
    "Free gift card for you",
    "Team lunch tomorrow"
]

# Label: 1 = spam, 0 = not spam
labels = [1, 0, 1, 0, 1, 0, 1, 0]

# Split manual
X_train = emails[:6]
X_test = emails[6:]
y_train = labels[:6]
y_test = labels[6:]

# Pipeline: Vectorizer + Naive Bayes
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # atau CountVectorizer()
    ('classifier', MultinomialNB())
])

# Training
pipeline.fit(X_train, y_train)

# Prediksi
y_pred = pipeline.predict(X_test)

print("Data Test:")
for email, pred, actual in zip(X_test, y_pred, y_test):
    status = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"Email: {email}")
    print(f"Prediksi: {status} (Actual: {'SPAM' if actual == 1 else 'NOT SPAM'})")
    print()

# Lihat feature importance (log probabilities)
vectorizer = pipeline.named_steps['vectorizer']
classifier = pipeline.named_steps['classifier']
feature_names = vectorizer.get_feature_names_out()

# Kata dengan probabilitas tertinggi untuk spam
spam_log_prob = classifier.feature_log_prob_[1]
top_spam_idx = np.argsort(spam_log_prob)[-10:]

print("Top 10 kata indikator SPAM:")
for idx in top_spam_idx:
    print(f"{feature_names[idx]}: {np.exp(spam_log_prob[idx]):.4f}")