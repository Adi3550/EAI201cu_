# Dataset
data = [
    {'Student': 1, 'Study Hours': 2, 'Pass/Fail': 0},
    {'Student': 2, 'Study Hours': 4, 'Pass/Fail': 0},
    {'Student': 3, 'Study Hours': 6, 'Pass/Fail': 1},
    {'Student': 4, 'Study Hours': 8, 'Pass/Fail': 1},
    {'Student': 5, 'Study Hours': 10, 'Pass/Fail': 1}
]

def calculate_gini(group):
    if len(group) == 0:
        return 0
    total = len(group)
    count_0 = sum(1 for item in group if item['Pass/Fail'] == 0)
    count_1 = total - count_0
    proportion_0 = count_0 / total
    proportion_1 = count_1 / total
    return 1 - (proportion_0 ** 2 + proportion_1 ** 2)


def split_data(data, split_value):
    left = [item for item in data if item['Study Hours'] <= split_value]
    right = [item for item in data if item['Study Hours'] > split_value]
    return left, right

def evaluate_split(data, split_values):
    best_gini = float('inf')
    best_split = None
    for split in split_values:
        left, right = split_data(data, split)
        gini_left = calculate_gini(left)
        gini_right = calculate_gini(right)
        weight_left = len(left) / len(data)
        weight_right = len(right) / len(data)
        gini = (weight_left * gini_left) + (weight_right * gini_right)
        print(f"Split at {split}: Gini = {gini:.4f} (Left: {len(left)}, Right: {len(right)})")
        if gini < best_gini:
            best_gini = gini
            best_split = split
    return best_split, best_gini

def build_tree(data, split_values, depth=0):
    gini = calculate_gini(data)
    print(f"{'  ' * depth}Node: Gini = {gini:.4f}, Samples = {len(data)}")
    if gini == 0 or depth >= 2:  
        return
    best_split, _ = evaluate_split(data, split_values)
    if best_split:
        left, right = split_data(data, best_split)
        print(f"{'  ' * depth}Split at {best_split}")
        build_tree(left, split_values, depth + 1)
        build_tree(right, split_values, depth + 1)


split_values = [3, 5, 7, 9]


print("Evaluating Splits:")
best_split, best_gini = evaluate_split(data, split_values)
print(f"Best split at {best_split} with Gini impurity {best_gini:.4f}")

print("\nDecision Tree:")
build_tree(data, split_values)


from sklearn.tree import DecisionTreeClassifier
import numpy as np
X = np.array([[item['Study Hours']] for item in data])
y = np.array([item['Pass/Fail'] for item in data])
model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)
print("\nscikit-learn Decision Tree Cross-Check:")
print(f"Split at: {model.tree_.threshold[0]}")
print(f"Predicted classes: {model.predict(X)}")
