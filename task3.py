# task3.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Create folder for plots
os.makedirs('plots', exist_ok=True)

# Load everything from Task 2
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')['finalist']
scaler = joblib.load('models/scaler.pkl')
lr = joblib.load('models/lr_best.pkl')
rf = joblib.load('models/rf_best.pkl')

X_test_scaled = scaler.transform(X_test)

print("\n" + "="*50)
print("Task 3: Model Evaluation")
print("="*50)

# Test both models
for name, model in [('Logistic Regression', lr), ('Random Forest', rf)]:
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n{name} - Test Set Results")
    print("-" * 40)
    print(classification_report(y_test, y_pred, digits=3))
    
    # ROC-AUC
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f'plots/confusion_{name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.savefig(f'plots/roc_{name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nAll plots saved in 'plots/' folder")
print("confusion_logistic_regression.png")
print("roc_logistic_regression.png")
print("confusion_random_forest.png")
print("roc_random_forest.png")