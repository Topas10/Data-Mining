from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier  # ← tambahkan ini
from sklearn.datasets import load_breast_cancer      # ← tambahkan ini
import numpy as np                                   # ← tambahkan ini
import matplotlib.pyplot as plt                      # ← tambahkan ini

# ============ LOAD DATA ============
data = load_breast_cancer()
X = data.data
y = data.target

# Hitung learning curve
train_sizes, train_scores, test_scores = \
    learning_curve(
        RandomForestClassifier(random_state=42),
        X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True,
        random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )

# Hitung mean dan std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std,
    train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std,
    test_mean + test_std, alpha=0.1, color='orange')
plt.plot(train_sizes, train_mean, 'o-', color='blue',
    label='Training Score')
plt.plot(train_sizes, test_mean, 'o-', color='orange',
    label='Cross-Validation Score')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.show()

# Interpretasi
gap = train_mean[-1] - test_mean[-1]
if gap > 0.1:
    print("MODEL OVERFITTING: Training >> Validation")
elif gap < -0.05:
    print("MODEL UNDERFITTING: Training < Validation")
else:
    print("MODEL GOOD: Training ~ Validation")