"""
VECTOR: Algorithm Race
=================================================
Tests alternative predictive models against XGBoost using the canonical
vector_features_v4.csv dataset. Validates using 5-Fold StratifiedGroupKFold.
Models tested:
  1. XGBoost (The existing champion)
  2. CatBoost (Ordered Boosting)
  3. LightGBM (Gradient Boosting)
  4. Balanced Random Forest (Bagging Ensemble)
  5. Logistic Regression (Linear Benchmark with Class Weights)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier

import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = WORKSPACE + r"\dataset"
THRESHOLD = 0.46

# ── 1. Load Data ──
df = pd.read_csv(DATA_DIR + r"\vector_features_v4.csv")
df["salary_drift_v"] = df["salary_drift_v"].fillna(0)

FEATURES = ['income_erosion_v','liquidity_momentum_v','overdraft_v','overdraft_t2',
            'salary_drift_v','tx_freq_v','avg_balance_t2','min_balance_t2','total_out_t2']

X = df[FEATURES].values
y = df["default"].values
groups = df["account_id"].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
scale_w = (y == 0).sum() / (y == 1).sum()

# ── 2. Define Competitors ──
models = {
    "XGBoost (Champion Base)": xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, 
        scale_pos_weight=scale_w, random_state=42, eval_metric="auc", verbosity=0
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        scale_pos_weight=scale_w, random_state=42, verbose=-1, force_col_wise=True
    ),
    "Balanced Random Forest": BalancedRandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, sampling_strategy='all', n_jobs=-1
    )
}

if CatBoostClassifier is not None:
    models["CatBoost"] = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=4,
        auto_class_weights="Balanced", random_state=42, verbose=0, eval_metric="AUC"
    )

results = []

print("="*60)
print("ALGORITHM RACE: Evaluating models via 5-Fold CV...")
print("="*60)

for name, model in models.items():
    all_y, all_prob = [], []
    
    for tr, te in sgkf.split(X, y, groups=groups):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        
        # Logistic Regression needs scaling
        if name == "Logistic Regression":
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
        model.fit(X_tr, y_tr)
        all_y.extend(y_te)
        all_prob.extend(model.predict_proba(X_te)[:, 1])
        
    all_y = np.array(all_y)
    all_prob = np.array(all_prob)
    preds = (all_prob >= THRESHOLD).astype(int)
    
    tp = ((preds==1)&(all_y==1)).sum()
    fn = ((preds==0)&(all_y==1)).sum()
    fp = ((preds==1)&(all_y==0)).sum()
    tn = ((preds==0)&(all_y==0)).sum()
    
    auc = roc_auc_score(all_y, all_prob)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f2 = fbeta_score(all_y, preds, beta=2)
    
    results.append({
        "Model": name,
        "AUC": round(auc, 4),
        "Recall": round(recall * 100, 1),
        "FPR": round(fpr * 100, 1),
        "F2": round(f2, 3),
        "Caught": f"{tp}/{tp+fn}"
    })

# Run Logistic Benchmark
all_y, all_prob = [], []
for tr, te in sgkf.split(X, y, groups=groups):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_te = scaler.transform(X[te])
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_tr, y[tr])
    all_y.extend(y[te])
    all_prob.extend(lr.predict_proba(X_te)[:, 1])

all_y = np.array(all_y); all_prob = np.array(all_prob)
preds = (all_prob >= THRESHOLD).astype(int)
tp = ((preds==1)&(all_y==1)).sum(); fn = ((preds==0)&(all_y==1)).sum()
fp = ((preds==1)&(all_y==0)).sum(); tn = ((preds==0)&(all_y==0)).sum()
auc = roc_auc_score(all_y, all_prob)
recall = tp / (tp + fn); fpr = fp / (fp + tn)
f2 = fbeta_score(all_y, preds, beta=2)

results.append({
    "Model": "Logistic Reg. (Benchmark)", "AUC": round(auc, 4),
    "Recall": round(recall * 100, 1), "FPR": round(fpr * 100, 1),
    "F2": round(f2, 3), "Caught": f"{tp}/{tp+fn}"
})

# ── Print Leaderboard ──
print(f"\n{'Leaderboard (Sorted by F2-Score)':^60}")
print("-"*60)
print(f"{'Model Name':<28} | {'AUC':>6} | {'Recall':>7} | {'FPR':>5} |  {'F2':>4}")
print("-"*60)
for r in sorted(results, key=lambda x: -x["F2"]):
    print(f"{r['Model']:<28} | {r['AUC']:>6.4f} | {r['Recall']:>6.1f}% | {r['FPR']:>4.1f}% | {r['F2']:>5.3f}")
print("="*60)
