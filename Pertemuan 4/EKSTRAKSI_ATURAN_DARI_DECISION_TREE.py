# Listing 4: extract_rules_from_tree.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np

def extract_rules(tree, feature_names, class_names, node_id=0, current_rule=None):
    # Ekstrak aturan IF-THEN dari decision tree
    if current_rule is None:
        current_rule = []

    rules = []

    # Cek apakah node daun (leaf)
    if tree.children_left[node_id] == tree.children_right[node_id]:
        # Node daun: output kelas
        class_id = np.argmax(tree.value[node_id][0])
        rule_str = "IF " + " AND ".join(current_rule)
        rule_str += f" THEN {class_names[class_id]}"
        rules.append(rule_str)
    else:
        # Node internal: rekursif ke kiri dan kanan
        feature = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]

        # Cabang kiri (<= threshold)
        left_rule = current_rule + [f"{feature} <= {threshold:.2f}"]
        rules.extend(extract_rules(tree, feature_names, class_names,
                                   tree.children_left[node_id], left_rule))

        # Cabang kanan (> threshold)
        right_rule = current_rule + [f"{feature} > {threshold:.2f}"]
        rules.extend(extract_rules(tree, feature_names, class_names,
                                   tree.children_right[node_id], right_rule))

    return rules

# Contoh ekstraksi dari model sebelumnya
dt = DecisionTreeClassifier(max_depth=2, random_state=42)
iris = load_iris()
X_train, y_train = iris.data, iris.target
dt.fit(X_train, y_train)

rules = extract_rules(dt.tree_, iris.feature_names, iris.target_names)
print("Aturan yang diekstrak:")
for i, rule in enumerate(rules):
    print(f"{i+1}. {rule}")