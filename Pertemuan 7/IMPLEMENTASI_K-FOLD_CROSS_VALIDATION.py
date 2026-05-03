from sklearn.model_selection import (cross_val_score,
    KFold, StratifiedKFold, train_test_split)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np

# ============ LOAD DATA ============
iris = load_iris()
X = iris.data
y = iris.target

# ============ HOLD-OUT ============
print("=" * 50)
print("HOLD-OUT VALIDATION")
print("=" * 50)

scores_holdout = []
for i in range(10):  # 10 kali ulangan
    X_tr, X_te, y_tr, y_te = train_test_split(X, y,
        test_size=0.3, random_state=i)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_tr, y_tr)
    scores_holdout.append(model.score(X_te, y_te))

print(f"Mean accuracy: {np.mean(scores_holdout):.4f}")
print(f"Std: {np.std(scores_holdout):.4f}")

# ============ K-FOLD CV ============
print("\n" + "=" * 50)
print("K-FOLD CROSS VALIDATION")
print("=" * 50)

# Standard k-Fold (tanpa stratified)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kfold = cross_val_score(model, X, y, cv=kfold,
    scoring='accuracy')
print(f"k-Fold (k=5) - Mean: {scores_kfold.mean():.4f}"
      f" (+/- {scores_kfold.std():.4f})")

# ============ STRATIFIED K-FOLD ============
print("\n" + "=" * 50)
print("STRATIFIED K-FOLD CV")
print("=" * 50)

strat_kfold = StratifiedKFold(n_splits=5, shuffle=True,
    random_state=42)
scores_strat = cross_val_score(model, X, y, cv=
    strat_kfold, scoring='accuracy')
print(f"Stratified k-Fold - Mean: {scores_strat.mean():.4f}"
      f" (+/- {scores_strat.std():.4f})")

# ============ PERBANDINGAN NILAI K ============
print("\n" + "=" * 50)
print("PERBANDINGAN BERBAGAI NILAI K")
print("=" * 50)

k_values = [3, 5, 7, 10]
for k in k_values:
    strat_kfold = StratifiedKFold(n_splits=k, shuffle=True,
        random_state=42)
    scores = cross_val_score(model, X, y, cv=strat_kfold,
        scoring='accuracy')
    print(f"k={k}: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}")