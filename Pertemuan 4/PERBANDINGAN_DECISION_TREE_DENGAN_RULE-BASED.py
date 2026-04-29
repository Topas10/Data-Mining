from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data (contoh: Titanic)
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'survived']]
titanic = titanic.dropna()
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

X = titanic.drop('survived', axis=1)
y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Decision Tree (Gini)': DecisionTreeClassifier(criterion='gini', max_depth=5),
    'Decision Tree (Entropy)': DecisionTreeClassifier(criterion='entropy', max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print(results_df.round(4))