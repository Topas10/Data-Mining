from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier  # ← tambahkan ini
from sklearn.metrics import roc_curve, roc_auc_score  # ← tambahkan ini
from sklearn.model_selection import train_test_split  # ← tambahkan ini
from sklearn.datasets import load_breast_cancer       # ← tambahkan ini
import matplotlib.pyplot as plt                       # ← tambahkan ini

# ============ LOAD DATA ============
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3, random_state=42)

# Definisikan model
models = {
    'Logistic Regression': LogisticRegression(
        random_state=42),
    'Random Forest': RandomForestClassifier(
        random_state=42),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SVM (RBF)': SVC(probability=True, random_state=42)
}

# Train dan plot ROC
plt.figure(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} ('
        f'AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label=
    'Random (AUC=0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# Model mana yang terbaik?
# Semakin kiri-atas kurva, semakin baik