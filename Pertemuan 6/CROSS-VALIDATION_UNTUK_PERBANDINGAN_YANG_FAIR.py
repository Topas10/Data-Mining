from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.datasets import load_breast_cancer
import pandas as pd
import time

# --- Load Dataset ---
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Stacking (diperlukan sebelum models dict) ---
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42))
]
meta_learner = LogisticRegression(max_iter=1000)
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba'
)

# --- List Model yang Akan Dibandingkan ---
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Stacking': stacking
}

# --- Cross-Validation dengan 5-Fold ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    elapsed_time = time.time() - start_time

    results.append({
        'Model': name,
        'Mean Accuracy': scores.mean(),
        'Std': scores.std(),
        'Training Time (s)': elapsed_time
    })

# --- Tampilkan Hasil ---
results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))

# --- Kesimpulan ---
best_model = results_df.loc[results_df['Mean Accuracy'].idxmax(), 'Model']
print(f"\nModel terbaik berdasarkan CV: {best_model}")