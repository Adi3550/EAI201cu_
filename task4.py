# task4.py
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# ------------------------------------------------------------------
# 1. Load model & data
# ------------------------------------------------------------------
lr = joblib.load('models/lr_best.pkl')
rf = joblib.load('models/rf_best.pkl')
data = pd.read_csv('cleaned_team_data.csv')

features = ['goal_diff_avg', 'win_rate', 'rank', 'participations']
print("\n=== Task 4: Feature Importance & Interpretation ===")
print("Features:", features)

# ------------------------------------------------------------------
# 2. Logistic Regression – absolute coefficients
# ------------------------------------------------------------------
lr_coef = abs(lr.coef_[0])
lr_importance = pd.Series(lr_coef, index=features).sort_values(ascending=False)

print("\nLogistic Regression – Absolute Coefficients")
print(lr_importance.round(4))

# ------------------------------------------------------------------
# 3. Random Forest – feature importances (Gini)
# ------------------------------------------------------------------
rf_importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

print("\nRandom Forest – Gini Importance")
print(rf_importance.round(4))

# ------------------------------------------------------------------
# 4. Combine & Plot
# ------------------------------------------------------------------
importance_df = pd.DataFrame({
    'Logistic (abs coef)': lr_coef,
    'Random Forest (Gini)': rf.feature_importances_
}, index=features)

ax = importance_df.plot.barh(figsize=(9,5))
plt.title('Feature Importance – Logistic Regression vs Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nPlot saved → plots/feature_importance.png")
