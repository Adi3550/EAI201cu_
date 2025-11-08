# task2.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -------------------------------------------------
# 1. Load cleaned data (from Task 1)
# -------------------------------------------------
data = pd.read_csv('cleaned_team_data.csv')
print("=== Task 2: Model Building and Training ===")
print(f"Loaded: {data.shape[0]} rows, {data.shape[1]} cols")
print("Finalists:", data['finalist'].sum(), "(0.2%)")

# Features you will use (keep the same order as in the screenshot)
features = ['goal_diff_avg', 'win_rate', 'rank', 'participations']
print("Features:", features)

X = data[features]
y = data['finalist']

# -------------------------------------------------
# 2. Train / test split (80/20) + scaling
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# save scaler for later tasks
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
print("scaler saved → models/scaler.pkl")

# -------------------------------------------------
# 3. Logistic Regression – GridSearchCV
# -------------------------------------------------
print("\nTuning Logistic Regression...")
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    lr_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
lr_grid.fit(X_train_sc, y_train)

print(f"Fitting {lr_grid.n_splits_} folds for each of {len(lr_grid.param_grid['C'])} candidates, totalling {lr_grid.cv_results_['mean_test_score'].size} fits")
print("Best LR:", lr_grid.best_params_)
print(f"CV F1: {lr_grid.best_score_:.3f}")

lr_best = lr_grid.best_estimator_
joblib.dump(lr_best, 'models/lr_best.pkl')
print("LR model saved → models/lr_best.pkl")

# -------------------------------------------------
# 4. Random Forest – GridSearchCV
# -------------------------------------------------
print("\nTuning Random Forest...")
rf_param_grid = {
    'n_estimators': [100],
    'max_depth'   : [5],
    'min_samples_split': [2],
    'min_samples_leaf' : [1]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
rf_grid.fit(X_train_sc, y_train)

print(f"Fitting {rf_grid.n_splits_} folds for each of {len(rf_grid.param_grid['n_estimators'])} candidates, totalling {rf_grid.cv_results_['mean_test_score'].size} fits")
print("Best RF:", rf_grid.best_params_)
print(f"CV F1: {rf_grid.best_score_:.3f}")

rf_best = rf_grid.best_estimator_
joblib.dump(rf_best, 'models/rf_best.pkl')
print("RF model saved → models/rf_best.pkl")

# -------------------------------------------------
# 5. Test‑set preview (exactly like your screenshot)
# -------------------------------------------------
print("\n--- Test Set Preview ---")
for name, model in [('Logistic', lr_best), ('Random Forest', rf_best)]:
    y_pred = model.predict(X_test_sc)
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred, digits=2))

# -------------------------------------------------
# 6. Save test split for Task 3 (optional but handy)
# -------------------------------------------------
X_test.to_csv('X_test.csv', index=False)
pd.DataFrame({'finalist': y_test}).to_csv('y_test.csv', index=False)