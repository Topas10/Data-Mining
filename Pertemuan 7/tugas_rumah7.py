from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedKFold,
    cross_val_score, train_test_split)
from sklearn.metrics import (roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# SIMPAN GAMBAR DI FOLDER YANG SAMA DENGAN FILE INI
# ============================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LOAD DATA
# ============================================================
data = load_iris()
X = data.data
y = data.target
class_names = data.target_names

print("=" * 60)
print("TUGAS RUMAH - KOMPARASI ALGORITMA KLASIFIKASI")
print("Dataset: Iris (Multi-Class)")
print("=" * 60)
print(f"Jumlah sampel : {X.shape[0]}")
print(f"Jumlah fitur  : {X.shape[1]}")
print(f"Kelas         : {list(class_names)}")

# ============================================================
# DEFINISI MODEL
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=200, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SVM (RBF)': SVC(probability=True, random_state=42)
}

# ============================================================
# STRATIFIED 10-FOLD CV
# ============================================================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\n" + "=" * 60)
print("EVALUASI DENGAN STRATIFIED 10-FOLD CROSS VALIDATION")
print("=" * 60)

results = {}
for name, model in models.items():
    acc  = cross_val_score(model, X, y, cv=skf,
               scoring='accuracy').mean()
    prec = cross_val_score(model, X, y, cv=skf,
               scoring='precision_macro').mean()
    rec  = cross_val_score(model, X, y, cv=skf,
               scoring='recall_macro').mean()
    f1   = cross_val_score(model, X, y, cv=skf,
               scoring='f1_macro').mean()
    roc  = cross_val_score(model, X, y, cv=skf,
               scoring='roc_auc_ovr').mean()

    results[name] = {
        'Accuracy' : acc,
        'Precision': prec,
        'Recall'   : rec,
        'F1'       : f1,
        'ROC-AUC'  : roc
    }

    print(f"\n{name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")

# ============================================================
# PLOT ROC CURVE SEMUA MODEL
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

colors = ['blue', 'green', 'red', 'purple']
plt.figure(figsize=(10, 8))

for (name, model), color in zip(models.items(), colors):
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)

    fpr_list, tpr_list, auc_list = [], [], []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i],
            y_proba[:, i])
        auc_val = roc_auc_score(y_test_bin[:, i],
            y_proba[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc_val)

    mean_auc = np.mean(auc_list)
    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr,
            fpr_list[i], tpr_list[i])
    mean_tpr /= n_classes

    plt.plot(all_fpr, mean_tpr, linewidth=2, color=color,
        label=f'{name} (AUC = {mean_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1,
    label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison - Iris Dataset\n'
    '(Macro Average, One-vs-Rest)', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'ROC_Curve_Comparison.png'),
    dpi=150)
plt.show()
print("Gambar disimpan: ROC_Curve_Comparison.png")

# ============================================================
# PLOT PERBANDINGAN METRIK
# ============================================================
metric_names = ['Accuracy', 'Precision', 'Recall',
    'F1', 'ROC-AUC']
x = np.arange(len(metric_names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, vals) in enumerate(results.items()):
    scores = [vals[m] for m in metric_names]
    ax.bar(x + i * width, scores, width, label=name)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Metrik Semua Model\n'
    '(Stratified 10-Fold CV)', fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_names)
ax.set_ylim(0.85, 1.02)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'Metric_Comparison.png'),
    dpi=150)
plt.show()
print("Gambar disimpan: Metric_Comparison.png")

# ============================================================
# LAPORAN ANALISIS
# ============================================================
print("\n" + "=" * 60)
print("LAPORAN ANALISIS")
print("=" * 60)

best_model = max(results, key=lambda x: results[x]['F1'])
best_scores = results[best_model]

print(f"\nModel Terbaik: {best_model}")
print(f"  Accuracy  : {best_scores['Accuracy']:.4f}")
print(f"  Precision : {best_scores['Precision']:.4f}")
print(f"  Recall    : {best_scores['Recall']:.4f}")
print(f"  F1        : {best_scores['F1']:.4f}")
print(f"  ROC-AUC   : {best_scores['ROC-AUC']:.4f}")

print("""
KESIMPULAN:
-----------
1. Semua model dievaluasi menggunakan Stratified 10-Fold CV
   untuk memastikan distribusi kelas seimbang di setiap fold.

2. Dataset Iris memiliki 3 kelas sehingga metrik dihitung
   dengan rata-rata macro (macro average).

3. ROC Curve digambar menggunakan pendekatan One-vs-Rest
   (OvR) dan dirata-rata secara macro.

4. Model terbaik dipilih berdasarkan nilai F1-Score tertinggi
   karena F1 menyeimbangkan Precision dan Recall.
""")