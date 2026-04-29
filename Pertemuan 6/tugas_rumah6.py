import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  ENSEMBLE METHODS COMPARISON  |  Dataset Titanic")
print("=" * 65)

# ============================================================
#  BAGIAN 1: Generate Dataset Titanic (langsung di memori)
# ============================================================

print("\n  [1/5] Memuat dataset Titanic ...")

np.random.seed(42)
n = 891
pclass   = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
sex      = np.random.choice([0, 1],    n, p=[0.65, 0.35])   # 0=male, 1=female
age      = np.clip(np.random.normal(29.7, 14.5, n), 0.5, 80)
sibsp    = np.random.choice([0,1,2,3,4,5], n, p=[0.68,0.23,0.05,0.02,0.01,0.01])
parch    = np.random.choice([0,1,2,3,4,5], n, p=[0.76,0.13,0.08,0.01,0.01,0.01])
fare     = np.clip(np.random.lognormal(3.0, 1.0, n), 0, 512)
embarked = np.random.choice([0, 1, 2], n, p=[0.72, 0.19, 0.09])
logit    = (-1.5 + (3 - pclass) * 0.5 + sex * 2.5
            - (age - 30) * 0.015 - sibsp * 0.15
            - parch * 0.1 + np.log1p(fare) * 0.1)
survived = (np.random.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)

df = pd.DataFrame({
    "pclass": pclass, "sex": sex, "age": age, "sibsp": sibsp,
    "parch": parch, "fare": fare, "embarked": embarked, "survived": survived
})

features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = SimpleImputer(strategy="median").fit_transform(df[features].values)
y = df["survived"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Sampel: {len(y)} | Train: {len(y_train)} | Test: {len(y_test)} | Survival rate: {y.mean():.2%}")

# ============================================================
#  BAGIAN 2: Helper Evaluasi
# ============================================================

def evaluate(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    p  = model.predict(Xte)
    cv = cross_val_score(model, Xtr, ytr, cv=5, scoring="accuracy").mean()
    return {
        "Model":     name,
        "Accuracy":  accuracy_score(yte, p),
        "Precision": precision_score(yte, p, zero_division=0),
        "Recall":    recall_score(yte, p, zero_division=0),
        "F1-Score":  f1_score(yte, p, zero_division=0),
        "CV-Mean":   cv,
        "y_pred":    p,
    }

results = []

# ============================================================
#  BAGIAN 3: Single Decision Tree
# ============================================================

print("\n  [2/5] Training Single Decision Tree ...")
results.append(evaluate(
    "Single Decision Tree",
    DecisionTreeClassifier(max_depth=5, random_state=42),
    X_train, y_train, X_test, y_test
))

# ============================================================
#  BAGIAN 4: Random Forest — Tuning n_estimators & max_depth
# ============================================================

print("  [3/5] Tuning Random Forest ...")
rf_grid = [(50,5),(50,10),(100,5),(100,10),(200,10),(200,None)]
best_rf, best_rf_acc, best_rf_p = None, 0, None
rf_rows = []
for ne, d in rf_grid:
    m = RandomForestClassifier(n_estimators=ne, max_depth=d, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    a = accuracy_score(y_test, m.predict(X_test))
    rf_rows.append((ne, d, a))
    if a > best_rf_acc:
        best_rf_acc, best_rf, best_rf_p = a, m, (ne, d)

print(f"\n  RF Tuning  {'n_est':>8} | {'max_depth':>10} | {'Accuracy':>10}")
print(f"  " + "-"*8 + "-+-" + "-"*10 + "-+-" + "-"*10)
for ne, d, a in rf_rows:
    mark = "  ◀ best" if (ne, d) == best_rf_p else ""
    print(f"  {ne:>10} | {str(d):>10} | {a*100:>9.2f}%{mark}")

results.append(evaluate(
    f"Random Forest (n={best_rf_p[0]}, d={best_rf_p[1]})",
    best_rf, X_train, y_train, X_test, y_test
))

# ============================================================
#  BAGIAN 5: Gradient Boosting — Tuning learning_rate & n_estimators
# ============================================================

print("\n  [4/5] Tuning Gradient Boosting ...")
gb_grid = [(0.01,100),(0.05,100),(0.1,100),(0.1,200),(0.2,100),(0.2,200)]
best_gb, best_gb_acc, best_gb_p = None, 0, None
gb_rows = []
for lr, ne in gb_grid:
    m = GradientBoostingClassifier(learning_rate=lr, n_estimators=ne,
                                   max_depth=3, random_state=42)
    m.fit(X_train, y_train)
    a = accuracy_score(y_test, m.predict(X_test))
    gb_rows.append((lr, ne, a))
    if a > best_gb_acc:
        best_gb_acc, best_gb, best_gb_p = a, m, (lr, ne)

print(f"\n  GB Tuning  {'lr':>6} | {'n_est':>6} | {'Accuracy':>10}")
print(f"  " + "-"*6 + "-+-" + "-"*6 + "-+-" + "-"*10)
for lr, ne, a in gb_rows:
    mark = "  ◀ best" if (lr, ne) == best_gb_p else ""
    print(f"  {lr:>8} | {ne:>6} | {a*100:>9.2f}%{mark}")

results.append(evaluate(
    f"Gradient Boosting (lr={best_gb_p[0]}, n={best_gb_p[1]})",
    best_gb, X_train, y_train, X_test, y_test
))

# ============================================================
#  BAGIAN 6: Stacking — minimal 3 base model
# ============================================================

print("\n  [5/5] Training Stacking Classifier ...")
stacking = StackingClassifier(
    estimators=[
        ("dt",  DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=100, max_depth=10,
                                       random_state=42, n_jobs=-1)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("svc", SVC(probability=True, kernel="rbf", random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=500, random_state=42),
    cv=5, passthrough=False, n_jobs=-1,
)
results.append(evaluate(
    "Stacking (DT+RF+KNN+SVC → LR)",
    stacking, X_train, y_train, X_test, y_test
))

# ============================================================
#  BAGIAN 7: Laporan Perbandingan
# ============================================================

print(f"\n{'='*65}")
print("  LAPORAN PERBANDINGAN SEMUA MODEL")
print(f"{'='*65}")
print(f"  {'Model':<40} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'CV':>6}")
print("  " + "-"*40 + "-+-" + ("-"*6+"-+-")*3 + "-"*6)
for r in results:
    print(f"  {r['Model']:<40} | {r['Accuracy']*100:>5.2f}% | "
          f"{r['Precision']*100:>5.2f}% | {r['Recall']*100:>5.2f}% | "
          f"{r['F1-Score']*100:>5.2f}% | {r['CV-Mean']*100:>5.2f}%")

print(f"\n{'='*65}")
print("  CLASSIFICATION REPORT DETAIL")
print(f"{'='*65}")
for r in results:
    print(f"\n  [{r['Model']}]")
    print(classification_report(
        y_test, r["y_pred"],
        target_names=["Tidak Selamat", "Selamat"], digits=4
    ))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, r["y_pred"]))

# ============================================================
#  BAGIAN 8: Analisis Kapan Model Lebih Unggul
# ============================================================

df_r     = pd.DataFrame([{k:v for k,v in r.items() if k!="y_pred"} for r in results])
best_acc = df_r.loc[df_r["Accuracy"].idxmax(), "Model"]
best_f1  = df_r.loc[df_r["F1-Score"].idxmax(), "Model"]
best_cv  = df_r.loc[df_r["CV-Mean"].idxmax(),  "Model"]

print(f"\n{'='*65}")
print("  ANALISIS: KAPAN MASING-MASING MODEL LEBIH UNGGUL?")
print(f"{'='*65}")
print(f"""
  Peringkat Akhir:
    Akurasi tertinggi  : {best_acc}
    F1-Score tertinggi : {best_f1}
    CV-Mean tertinggi  : {best_cv}

  +-----------------------------------------------------------+
  |  Single Decision Tree                                     |
  |  + Cepat, mudah diinterpretasi (bisa divisualisasi).     |
  |  + Cocok sebagai baseline / data kecil sederhana.        |
  |  - Rentan overfitting, variance tinggi.                  |
  |                                                           |
  |  Random Forest                                            |
  |  + Mengurangi variance via bagging ratusan DT.           |
  |  + Tahan noise, outlier, & missing value.                |
  |  + Unggul saat data besar & fitur banyak.                |
  |  - Kurang optimal untuk data time-series.                |
  |                                                           |
  |  Gradient Boosting                                        |
  |  + Akurasi tinggi via boosting (koreksi error iteratif). |
  |  + Sangat kuat untuk data tabular.                       |
  |  - Lambat, perlu tuning learning_rate hati-hati.         |
  |  - Rentan overfitting jika lr terlalu besar.             |
  |                                                           |
  |  Stacking                                                 |
  |  + Menggabungkan keunggulan banyak model (diversity).    |
  |  + Generalisasi terbaik -- meta-learner belajar kapan    |
  |    masing-masing base model bisa dipercaya.              |
  |  - Paling kompleks & lambat, risiko data leakage.        |
  +-----------------------------------------------------------+
""")