"""
VECTOR: Behavioral Expert Model
================================
Trains an XGBoost classifier on the engineered Physics-of-Risk features.
Outputs:
  - behavioral_engine.pkl   (trained model weights)
  - roc_curve.png           (AUC evaluation chart)
  - feature_importance.png  (SHAP-style importance for hackathon judges)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)
import xgboost as xgb
import joblib

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(WORKSPACE, "dataset")

# ─────────────────────────────────────────────
# 1. Load preserved feature matrix
# ─────────────────────────────────────────────
print("[1/5] Loading vector_features_v4.csv...")
df = pd.read_csv(os.path.join(DATA_DIR, "vector_features_v4.csv"))
df["salary_drift_v"] = df["salary_drift_v"].fillna(0)

FEATURE_COLS = [
    "income_erosion_v", "liquidity_momentum_v",
    "overdraft_v", "overdraft_t2",
    "salary_drift_v", "tx_freq_v",
    "avg_balance_t2", "min_balance_t2", "total_out_t2"
]
X      = df[FEATURE_COLS].values
y      = df["default"].values
groups = df["account_id"].values   # used to prevent same account leaking across folds

print(f"   Matrix: {X.shape}  |  Default rate: {y.mean()*100:.1f}%")

# ─────────────────────────────────────────────
# 2. Stratified Group K-Fold Cross Validation
#    (Ensures windows from the same account never
#     straddle train/test — prevents data leakage)
# ─────────────────────────────────────────────
print("\n[2/5] Running StratifiedGroupKFold (5-fold)...")

scale_weight = (y == 0).sum() / (y == 1).sum()   # handles 1:8 class imbalance

model = xgb.XGBClassifier(
    n_estimators      = 300,
    learning_rate     = 0.05,
    max_depth         = 4,
    scale_pos_weight  = scale_weight,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 3,
    random_state      = 42,
    eval_metric       = "auc",
    verbosity         = 0,
)

sgkf    = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores, all_y_true, all_y_prob = [], [], []

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, proba)
    auc_scores.append(auc)
    all_y_true.extend(y_te)
    all_y_prob.extend(proba)
    print(f"   Fold {fold+1}: ROC-AUC = {auc:.4f}")

mean_auc = np.mean(auc_scores)
std_auc  = np.std(auc_scores)
print(f"\n   Mean ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")

# ─────────────────────────────────────────────
# 3. Final model trained on full dataset
# ─────────────────────────────────────────────
print("\n[3/5] Training final model on full dataset...")
model.fit(X, y)

# ─────────────────────────────────────────────
# 4. Save model weights
# ─────────────────────────────────────────────
model_path = os.path.join(WORKSPACE, "Output_Artifacts", "behavioral_engine.pkl")
joblib.dump(model, model_path)
print(f"[4/5] Model saved -> {model_path}")

# ─────────────────────────────────────────────
# 5. Visualisations
# ─────────────────────────────────────────────
print("[5/5] Generating charts...")

# --- ROC Curve (aggregate across all folds) ---
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"VECTOR Behavioral Expert (AUC = {mean_auc:.3f})")
plt.plot([0,1],[0,1], color="navy", linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — VECTOR Behavioral Specialist")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(WORKSPACE, "Output_Artifacts", "roc_curve.png"), dpi=150)
plt.close()

# --- Feature Importance ---
importance = model.feature_importances_
feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importance})
feat_df = feat_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(9, 5))
colors = ["#ef4444" if imp > feat_df["Importance"].median() else "#3b82f6"
          for imp in feat_df["Importance"]]
plt.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color=colors[::-1])
plt.xlabel("Information Gain (XGBoost Importance)")
plt.title("VECTOR — What Drives Default? (Behavioral Expert Feature Importance)")
plt.tight_layout()
plt.savefig(os.path.join(WORKSPACE, "Output_Artifacts", "feature_importance.png"), dpi=150)
plt.close()

print("\n=== VECTOR Behavioral Expert — Training Complete ===")
print(f"  ROC-AUC (5-fold CV): {mean_auc:.4f} ± {std_auc:.4f}")
print(f"  Model:               {model_path}")
print(f"  ROC Chart:           {WORKSPACE}\\roc_curve.png")
print(f"  Importance Chart:    {WORKSPACE}\\feature_importance.png")

# --- Live Risk Score demo ---
print("\n--- Sample Behavioural Risk Scores (Probability) ---")
sample = df.sample(8, random_state=42)
sample["VECTOR_Risk_Score"] = model.predict_proba(sample[FEATURE_COLS].values)[:, 1]
print(sample[["account_id","window","default","VECTOR_Risk_Score"]].to_string(index=False))
