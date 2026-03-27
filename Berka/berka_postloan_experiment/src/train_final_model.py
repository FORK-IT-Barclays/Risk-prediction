import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score, confusion_matrix
import joblib

# Paths
DATA_PATH = r"D:\fork_it\Risk-prediction\Berka\berka_postloan_experiment\outputs\postloan_vector_features.csv"
MODEL_DIR = r"D:\fork_it\Risk-prediction\Berka\berka_postloan_experiment\models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    'income_erosion_v', 'liquidity_momentum_v', 'overdraft_v', 'overdraft_t2',
    'salary_drift_v', 'tx_freq_v', 'avg_balance_t2', 'min_balance_t2', 'total_out_t2'
]

# Business Logic Lock-in
FINAL_THRESHOLD = 0.368  # Target: 85% Recall with ~30% Precision

def load_and_prep_data():
    df = pd.read_csv(DATA_PATH)
    df['salary_drift_v'] = df['salary_drift_v'].fillna(0.0)
    
    X = df[FEATURES]
    y = df['default']
    groups = df['account_id']
    return X, y, groups

def train_and_evaluate():
    X, y, groups = load_and_prep_data()

    # The Tuned Balanced Bagging Engine
    clf = BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=10,
        max_features='sqrt',
        sampling_strategy='auto', 
        replacement=True,
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_proba = np.zeros(len(X))

    print("\n--- Running 5-Fold Evaluation ---")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val     = X.iloc[val_idx], y.iloc[val_idx]

        clf.fit(X_train, y_train)
        oof_preds_proba[val_idx] = clf.predict_proba(X_val)[:, 1]

    # Evaluate using the locked-in Business Threshold
    print("\n=======================================================")
    print(f"FINAL BUSINESS EVALUATION (Threshold: {FINAL_THRESHOLD})")
    print("=======================================================")
    
    print(f"OOF ROC AUC: {roc_auc_score(y, oof_preds_proba):.4f}")
    
    final_preds = (oof_preds_proba >= FINAL_THRESHOLD).astype(int)
    cm = confusion_matrix(y, final_preds)
    
    recall = cm[1,1] / (cm[1,0] + cm[1,1])
    precision = cm[1,1] / (cm[0,1] + cm[1,1])
    
    print(f"\nDefault Catch Rate (Recall): {recall * 100:.1f}%")
    print(f"Precision on Flags:          {precision * 100:.1f}%")
    
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=['Actual Good', 'Actual Default'], columns=['Pred Good', 'Pred Default']))
    print("\nClassification Report:")
    print(classification_report(y, final_preds))

    # Train the final production model on 100% of the data
    print("\n--- Training Final Production Model ---")
    clf.fit(X, y)
    
    final_model_path = os.path.join(MODEL_DIR, "final_balanced_engine.pkl")
    joblib.dump(clf, final_model_path)
    print(f"SUCCESS: Saved final model to {final_model_path}")

if __name__ == "__main__":
    train_and_evaluate()
