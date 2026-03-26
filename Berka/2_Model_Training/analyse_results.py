import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score, precision_recall_curve
import xgboost as xgb

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Output_Artifacts", "behavioral_engine_v2.pkl")
model = joblib.load(model_path)
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "vector_features_v4.csv")
df = pd.read_csv(data_path)
df['salary_drift_v'] = df['salary_drift_v'].fillna(0)

FEATURES = ['income_erosion_v','liquidity_momentum_v','overdraft_v','overdraft_t2',
            'salary_drift_v','tx_freq_v','avg_balance_t2','min_balance_t2','total_out_t2']

X, y, g = df[FEATURES].values, df['default'].values, df['account_id'].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
all_y, all_prob = [], []

for tr, te in sgkf.split(X, y, groups=g):
    scale_w = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    m = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        scale_pos_weight=scale_w, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, random_state=42, eval_metric='auc', verbosity=0
    )
    m.fit(X[tr], y[tr])
    prob = m.predict_proba(X[te])[:, 1]
    all_y.extend(y[te])
    all_prob.extend(prob)

all_y = np.array(all_y)
all_prob = np.array(all_prob)

print("=" * 55)
print("STRATEGY COMPARISON — Handling 1:8 Class Imbalance")
print("=" * 55)

for label, thresh in [("Default (0.50 threshold)", 0.50),
                       ("Tuned  (0.35 threshold)", 0.35),
                       ("Tuned  (0.25 threshold)", 0.25)]:
    preds = (all_prob >= thresh).astype(int)
    cm = np.array([[((preds==0)&(all_y==0)).sum(), ((preds==1)&(all_y==0)).sum()],
                   [((preds==0)&(all_y==1)).sum(), ((preds==1)&(all_y==1)).sum()]])
    tn,fp,fn,tp = cm.ravel()
    recall   = tp / (tp + fn)
    fpr      = fp / (fp + tn)
    prec     = tp / (tp + fp) if (tp+fp)>0 else 0
    f2       = fbeta_score(all_y, preds, beta=2)
    print(f"\n  [{label}]")
    print(f"    Default Recall (Catch Rate):  {recall*100:.1f}%  (catching {tp} of {tp+fn} defaults)")
    print(f"    False Positive Rate:          {fpr*100:.1f}%   ({fp} unnecessary alerts)")
    print(f"    Precision on Defaults:        {prec*100:.1f}%")
    print(f"    F2-Score (recall-weighted):   {f2:.3f}")

# Find the optimal threshold for F2
precs, recs, thresholds = precision_recall_curve(all_y, all_prob)
f2_scores = (5 * precs * recs) / (4 * precs + recs + 1e-9)
best_idx = np.argmax(f2_scores)
best_thresh = thresholds[best_idx]
best_preds  = (all_prob >= best_thresh).astype(int)
tp = ((best_preds==1)&(all_y==1)).sum()
fn = ((best_preds==0)&(all_y==1)).sum()
fp = ((best_preds==1)&(all_y==0)).sum()
tn = ((best_preds==0)&(all_y==0)).sum()

print(f"\n  [F2-Optimal (threshold={best_thresh:.2f})]")
print(f"    Default Recall:         {tp/(tp+fn)*100:.1f}%  (catching {tp} of {tp+fn} defaults)")
print(f"    False Positive Rate:    {fp/(fp+tn)*100:.1f}%")
print(f"    Precision on Defaults:  {tp/(tp+fp)*100:.1f}%")
print(f"    F2-Score:               {f2_scores[best_idx]:.3f}")
print(f"\n  Overall ROC-AUC: {roc_auc_score(all_y, all_prob):.4f}")
print("\n  KEY INSIGHT: threshold tuning is free — same model, better recall.")
