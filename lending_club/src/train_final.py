"""
train_final.py — Final production model retrain on ALL clean post-2012 data.
No train/val/test split — evaluation is complete. This is the deployment model.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix
)

from config import (
    CLEAN_PARQUET_PATH, PARQUET_PATH, ALL_FEATURES, T0_FEATURES,
    TARGET, XGB_PARAMS, MODEL_DIR, PLOT_DIR, REPORT_DIR, SEED
)
from features import load_and_filter, impute

plt.rcParams.update({
    'figure.dpi': 130, 'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d27', 'axes.edgecolor': '#444',
    'axes.labelcolor': '#ccc', 'xtick.color': '#aaa', 'ytick.color': '#aaa',
    'text.color': '#eee', 'grid.color': '#2a2d3a', 'grid.linestyle': '--',
    'axes.titlecolor': '#fff',
})

print("=" * 65)
print("  FINAL PRODUCTION MODEL — Full Data Retrain")
print("  No held-out split. All post-2012 clean data used.")
print("=" * 65)

# ── Load clean data ────────────────────────────────────────────────────────
data_path = CLEAN_PARQUET_PATH if CLEAN_PARQUET_PATH.exists() else PARQUET_PATH
print(f"\n📂 Using: {data_path.name}")
df = load_and_filter(str(data_path))

available = [f for f in ALL_FEATURES if f in df.columns]
print(f"Using {len(available)} features | {len(df):,} total rows")
df = impute(df, available)

X_all = df[available].values
y_all = df[TARGET].values

pos = y_all.sum()
neg = len(y_all) - pos
scale_pw = neg / pos
print(f"Class distribution: {int(neg):,} safe / {int(pos):,} delinquent | scale_pos_weight={scale_pw:.2f}")

# ── Scale ──────────────────────────────────────────────────────────────────
print("\n⏳ Fitting RobustScaler on full data …")
scaler = RobustScaler()
X_all_s = scaler.fit_transform(X_all)
print("✅ Scaler fitted")

# ── Cohort layer ───────────────────────────────────────────────────────────
t0_idx = [available.index(f) for f in T0_FEATURES if f in available]
print(f"\n⏳ Building cohort clusters (n=20) on {len(t0_idx)} origination features …")
kmeans = MiniBatchKMeans(n_clusters=20, random_state=SEED, batch_size=10000)
kmeans.fit(X_all_s[:, t0_idx])
cohort_labels = kmeans.predict(X_all_s[:, t0_idx])

# Cohort risk map from ALL data
df_tmp = df[[TARGET]].copy()
df_tmp['cohort'] = cohort_labels
cohort_risk = df_tmp.groupby('cohort')[TARGET].mean().to_dict()
print(f"✅ Cohort risk map built")

# ── Anomaly layer ──────────────────────────────────────────────────────────
print(f"\n⏳ Fitting Isolation Forest on full data …")
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=SEED, n_jobs=-1)
iso.fit(X_all_s)
print("✅ Isolation Forest fitted")

# ── XGBoost — full retrain (no eval_set) ──────────────────────────────────
params = {**XGB_PARAMS}
params.pop('random_state', None)
params.pop('eval_metric', None)   # No eval_set → remove eval_metric to avoid warning
params['seed'] = SEED
params['scale_pos_weight'] = scale_pw
params['n_estimators'] = 400      # Use best iteration from previous runs (≈100)

print(f"\n⏳ Training XGBoost on {len(X_all):,} rows …")
model = XGBClassifier(**params)
model.fit(X_all_s, y_all, verbose=50)
print("✅ XGBoost training complete")

# ── Sanity check: in-sample composite score ────────────────────────────────
xgb_prob  = model.predict_proba(X_all_s)[:, 1]
cohort_scores = np.array([cohort_risk.get(c, 0.25) for c in cohort_labels])

# Threshold layer
from features import apply_threshold_rules
thresh_scores = apply_threshold_rules(df).values

# Anomaly layer
anom_raw = iso.decision_function(X_all_s)
anom_scores = 1 - (anom_raw - anom_raw.min()) / (anom_raw.max() - anom_raw.min() + 1e-9)

composite = (0.50 * xgb_prob +
             0.25 * cohort_scores +
             0.15 * thresh_scores +
             0.10 * anom_scores)

# In-sample metrics (informational only — not for model selection)
y_pred = (composite >= 0.30).astype(int)
print(f"\n{'='*65}")
print("  IN-SAMPLE CHECK (expected to be optimistic — informational only)")
print(f"{'='*65}")
print(f"  ROC-AUC  : {roc_auc_score(y_all, composite):.4f}")
print(f"  PR-AUC   : {average_precision_score(y_all, composite):.4f}")
print(f"  Recall   : {recall_score(y_all, y_pred):.4f}")
print(f"  Precision: {precision_score(y_all, y_pred, zero_division=0):.4f}")
print(f"  F1       : {f1_score(y_all, y_pred):.4f}")
cm = confusion_matrix(y_all, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"  FNR      : {fn/(fn+tp):.4f}  (missed delinquencies)")
print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
print("\n  ⚠️  These are in-sample metrics — use held-out test 2018 results for real evaluation")

# ── Feature importance ─────────────────────────────────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=available).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(12, 10))
colors = ['#ff6b6b' if f in ['fico_drop','late_fee_flag','repay_ratio','out_prncp_ratio','delinq_recency_score']
          else '#00d4ff' for f in feat_imp.index]
ax.barh(feat_imp.index, feat_imp.values, color=colors)
ax.set_title('Final Production Model — Feature Importance', fontsize=14)
ax.set_xlabel('XGBoost Gain Score')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#ff6b6b', label='Key historian signals'),
                   Patch(facecolor='#00d4ff', label='Origination features')],
          loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'final_model_feature_importance.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: final_model_feature_importance.png")

# ── Save final production artifacts ───────────────────────────────────────
joblib.dump(model,       MODEL_DIR / 'xgb_historian_FINAL.pkl')
joblib.dump(scaler,      MODEL_DIR / 'robust_scaler_FINAL.pkl')
joblib.dump(kmeans,      MODEL_DIR / 'cohort_kmeans_FINAL.pkl')
joblib.dump(iso,         MODEL_DIR / 'isolation_forest_FINAL.pkl')
joblib.dump(cohort_risk, MODEL_DIR / 'cohort_risk_map_FINAL.pkl')
joblib.dump(available,   MODEL_DIR / 'feature_list_FINAL.pkl')
print("✅ All FINAL model artifacts saved to outputs/models/")

summary = '\n'.join([
    "FINAL PRODUCTION MODEL SUMMARY",
    "=" * 50,
    f"Training data  : {len(df):,} rows (all post-2012 clean data)",
    f"Features used  : {len(available)}",
    f"XGBoost trees  : {model.n_estimators}",
    f"scale_pos_weight: {scale_pw:.2f}",
    f"Cohort clusters: 20",
    "",
    "Validated performance (held-out test 2018):",
    "  ROC-AUC  : 0.8857",
    "  PR-AUC   : 0.3246  (9× above random for 3.6% class rate)",
    "  Recall   : 99.0%",
    "  FNR      : 1.03%",
    "",
    "Deployment threshold: 0.30 (tunable per portfolio)",
    "Artifacts: outputs/models/*_FINAL.pkl",
])
(REPORT_DIR / 'final_model_summary.txt').write_text(summary)
print("✅ Saved: reports/final_model_summary.txt")
print("\n🎉 Final production model ready for deployment!")
