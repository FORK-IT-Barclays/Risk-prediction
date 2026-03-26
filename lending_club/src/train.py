"""
train.py — Financial Historian Model: full training + evaluation pipeline
Run from: /home/hitesh-m-r/Downloads/archive/src/
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score
)
from sklearn.ensemble import IsolationForest

from config import (
    CLEAN_PARQUET_PATH, PARQUET_PATH, CSV_PATH, ALL_FEATURES, T0_FEATURES,
    TARGET, XGB_PARAMS, DECISION_THRESHOLD, MODEL_DIR, PLOT_DIR, REPORT_DIR, SEED
)
from features import (
    load_and_filter, impute, temporal_split,
    scale_features, build_cohorts, compute_cohort_risk, apply_threshold_rules
)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 130, 'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d27', 'axes.edgecolor': '#444',
    'axes.labelcolor': '#ccc', 'xtick.color': '#aaa',
    'ytick.color': '#aaa', 'text.color': '#eee',
    'grid.color': '#2a2d3a', 'grid.linestyle': '--',
    'axes.titlecolor': '#fff', 'axes.titlesize': 13,
})
PAL = ['#00d4ff', '#ff6b6b', '#ffd700', '#7bed9f', '#a29bfe']

# ──────────────────────────────────────────────────────────────────────────────
# ── LOAD DATA — use clean parquet if available ────────────────────────────
data_path = CLEAN_PARQUET_PATH if CLEAN_PARQUET_PATH.exists() else PARQUET_PATH
print(f"📂 Using: {data_path.name}")
df = load_and_filter(str(data_path))

# Validate features exist
available = [f for f in ALL_FEATURES if f in df.columns]
missing_f = [f for f in ALL_FEATURES if f not in df.columns]
if missing_f:
    print(f"⚠️  Features not found (skipping): {missing_f}")
ALL_FEATURES_USE = available
print(f"Using {len(ALL_FEATURES_USE)} features")

# ──────────────────────────────────────────────────────────────────────────────
# 2. IMPUTE + SPLIT
# ──────────────────────────────────────────────────────────────────────────────
df = impute(df, ALL_FEATURES_USE)
train_df, val_df, test_df = temporal_split(df)

X_train = train_df[ALL_FEATURES_USE].values
y_train = train_df[TARGET].values
X_val   = val_df[ALL_FEATURES_USE].values
y_val   = val_df[TARGET].values
X_test  = test_df[ALL_FEATURES_USE].values
y_test  = test_df[TARGET].values

# ──────────────────────────────────────────────────────────────────────────────
# 3. SCALE
# ──────────────────────────────────────────────────────────────────────────────
X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

# ──────────────────────────────────────────────────────────────────────────────
# 4. COHORT LAYER (Layer 1)
# ──────────────────────────────────────────────────────────────────────────────
t0_idx = [ALL_FEATURES_USE.index(f) for f in T0_FEATURES if f in ALL_FEATURES_USE]
kmeans = build_cohorts(X_train_s[:, t0_idx], n_clusters=20)

train_cohorts = kmeans.predict(X_train_s[:, t0_idx])
val_cohorts   = kmeans.predict(X_val_s[:, t0_idx])
test_cohorts  = kmeans.predict(X_test_s[:, t0_idx])

cohort_risk = compute_cohort_risk(train_df, train_cohorts)

def get_cohort_scores(cohort_labels):
    return np.array([cohort_risk.get(c, 0.25) for c in cohort_labels])

cohort_train = get_cohort_scores(train_cohorts)
cohort_val   = get_cohort_scores(val_cohorts)
cohort_test  = get_cohort_scores(test_cohorts)

# ──────────────────────────────────────────────────────────────────────────────
# 5. THRESHOLD LAYER (Layer 2)
# ──────────────────────────────────────────────────────────────────────────────
thresh_train = apply_threshold_rules(train_df).values
thresh_val   = apply_threshold_rules(val_df).values
thresh_test  = apply_threshold_rules(test_df).values

# ──────────────────────────────────────────────────────────────────────────────
# 6. ANOMALY LAYER (Layer 3 — Isolation Forest)
# ──────────────────────────────────────────────────────────────────────────────
print("⏳ Fitting Isolation Forest (anomaly layer) …")
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=SEED, n_jobs=-1)
iso.fit(X_train_s)

def anomaly_score(X_s):
    # -1 = anomaly, 1 = normal → convert to 0-1 prob
    raw = iso.decision_function(X_s)         # higher = more normal
    return 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

anom_train = anomaly_score(X_train_s)
anom_val   = anomaly_score(X_val_s)
anom_test  = anomaly_score(X_test_s)
print("✅ Anomaly layer ready")

# ──────────────────────────────────────────────────────────────────────────────
# 7. XGBOOST CORE MODEL
# ──────────────────────────────────────────────────────────────────────────────
pos = y_train.sum()
neg = len(y_train) - pos
scale_pw = neg / pos
params = {**XGB_PARAMS, 'scale_pos_weight': scale_pw}
# Remove non-xgb keys
params.pop('random_state', None)
params['seed'] = SEED

print(f"\n⏳ Training XGBoost (scale_pos_weight={scale_pw:.2f}) …")
model = XGBClassifier(**params)
model.fit(
    X_train_s, y_train,
    eval_set=[(X_val_s, y_val)],
    verbose=50
)
print("✅ XGBoost training complete")

# ──────────────────────────────────────────────────────────────────────────────
# 8. COMPOSITE HISTORIAN SCORE
# ──────────────────────────────────────────────────────────────────────────────
def historian_score(X_s, cohort_scores, thresh_scores, anom_scores):
    xgb_prob = model.predict_proba(X_s)[:, 1]
    # Weighted composite: 50% XGBoost + 25% cohort + 15% threshold + 10% anomaly
    composite = (0.50 * xgb_prob +
                 0.25 * cohort_scores +
                 0.15 * thresh_scores +
                 0.10 * anom_scores)
    return composite, xgb_prob

hist_val,   xgb_val   = historian_score(X_val_s,  cohort_val,  thresh_val,  anom_val)
hist_test,  xgb_test  = historian_score(X_test_s, cohort_test, thresh_test, anom_test)

# ──────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(y_true, y_prob, label: str, threshold: float = DECISION_THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)
    roc    = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    f1     = f1_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*60}")
    print(f"  📊 {label} METRICS  (threshold={threshold})")
    print(f"{'='*60}")
    print(f"  ROC-AUC          : {roc:.4f}")
    print(f"  PR-AUC           : {pr_auc:.4f}")
    print(f"  F1 Score         : {f1:.4f}")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall (Sensitivity): {rec:.4f}")
    print(f"  Specificity      : {tn/(tn+fp):.4f}  (non-delinquent catch rate)")
    print(f"  False Pos Rate   : {fp/(fp+tn):.4f}")
    print(f"  False Neg Rate   : {fn/(fn+tp):.4f}  ← missed delinquencies")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:>6,}  FP={fp:>6,}")
    print(f"    FN={fn:>6,}  TP={tp:>6,}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Non-Delinquent','Delinquent'])}")
    return dict(roc=roc, pr_auc=pr_auc, f1=f1, precision=prec, recall=rec,
                tn=tn, fp=fp, fn=fn, tp=tp, cm=cm)

val_metrics  = evaluate(y_val,  hist_val,  "VALIDATION (2017) — Historian")
test_metrics = evaluate(y_test, hist_test, "TEST (2018) — Historian")

# ──────────────────────────────────────────────────────────────────────────────
# 10. PLOTS
# ──────────────────────────────────────────────────────────────────────────────

# A. ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Financial Historian Model — ROC & PR Curves (Test 2018)', fontsize=14)

fpr, tpr, _ = roc_curve(y_test, hist_test)
axes[0].plot(fpr, tpr, color='#00d4ff', lw=2.5,
             label=f"Historian (AUC={test_metrics['roc']:.4f})")
fpr_x, tpr_x, _ = roc_curve(y_test, xgb_test)
axes[0].plot(fpr_x, tpr_x, color='#a29bfe', lw=1.5, linestyle='--',
             label=f"XGBoost only (AUC={roc_auc_score(y_test,xgb_test):.4f})")
axes[0].plot([0,1],[0,1],'--', color='#555', lw=1)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve'); axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)

# B. PR Curve
prec_c, rec_c, _ = precision_recall_curve(y_test, hist_test)
axes[1].plot(rec_c, prec_c, color='#ff6b6b', lw=2.5,
             label=f"Historian (PR-AUC={test_metrics['pr_auc']:.4f})")
prec_x, rec_x, _ = precision_recall_curve(y_test, xgb_test)
axes[1].plot(rec_x, prec_x, color='#ffd700', lw=1.5, linestyle='--',
             label=f"XGBoost only (PR-AUC={average_precision_score(y_test,xgb_test):.4f})")
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve'); axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'model_roc_pr.png', bbox_inches='tight')
plt.close()
print("✅ Saved: model_roc_pr.png")

# C. Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrices — Historian Score (threshold=0.30)', fontsize=13)
for ax, metrics, split in zip(axes, [val_metrics, test_metrics], ['Val 2017', 'Test 2018']):
    cm = metrics['cm']
    sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax,
                xticklabels=['Non-Delinq','Delinq'],
                yticklabels=['Non-Delinq','Delinq'],
                annot_kws={'size': 13, 'weight': 'bold'},
                linewidths=1.5, linecolor='#222')
    ax.set_title(f'{split}', fontsize=12)
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'model_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("✅ Saved: model_confusion_matrix.png")

# D. Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=ALL_FEATURES_USE).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(12, 10))
colors = ['#ff6b6b' if f in ['fico_drop','late_fee_flag','repay_ratio','out_prncp_ratio','delinq_recency_score']
          else '#00d4ff' for f in feat_imp.index]
ax.barh(feat_imp.index, feat_imp.values, color=colors)
ax.set_title('XGBoost Feature Importance (gain)', fontsize=14)
ax.set_xlabel('Importance Score')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#ff6b6b', label='Historian signals'),
                   Patch(facecolor='#00d4ff', label='Origination features')],
          loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'model_feature_importance.png', bbox_inches='tight')
plt.close()
print("✅ Saved: model_feature_importance.png")

# E. Threshold sensitivity
thresholds = np.arange(0.10, 0.70, 0.02)
f1s, precs, recs = [], [], []
for t in thresholds:
    y_pred_t = (hist_test >= t).astype(int)
    f1s.append(f1_score(y_test, y_pred_t, zero_division=0))
    precs.append(precision_score(y_test, y_pred_t, zero_division=0))
    recs.append(recall_score(y_test, y_pred_t, zero_division=0))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(thresholds, f1s,   color='#ffd700', lw=2.5, label='F1')
ax.plot(thresholds, precs, color='#00d4ff', lw=2,   label='Precision', linestyle='--')
ax.plot(thresholds, recs,  color='#ff6b6b', lw=2,   label='Recall',    linestyle='--')
ax.axvline(DECISION_THRESHOLD, color='#7bed9f', linestyle=':', lw=2,
           label=f'Chosen threshold ({DECISION_THRESHOLD})')
ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Score')
ax.set_title('Threshold Sensitivity — Historian Score (Test 2018)', fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'model_threshold_sensitivity.png', bbox_inches='tight')
plt.close()
print("✅ Saved: model_threshold_sensitivity.png")

# ──────────────────────────────────────────────────────────────────────────────
# 11. SAVE ARTIFACTS
# ──────────────────────────────────────────────────────────────────────────────
joblib.dump(model,  MODEL_DIR / 'xgb_historian.pkl')
joblib.dump(scaler, MODEL_DIR / 'robust_scaler.pkl')
joblib.dump(kmeans, MODEL_DIR / 'cohort_kmeans.pkl')
joblib.dump(iso,    MODEL_DIR / 'isolation_forest.pkl')
joblib.dump(cohort_risk, MODEL_DIR / 'cohort_risk_map.pkl')
print("\n✅ All model artifacts saved to outputs/models/")

# ──────────────────────────────────────────────────────────────────────────────
# 12. SAVE METRICS REPORT
# ──────────────────────────────────────────────────────────────────────────────
report_lines = [
    "# Financial Historian Model — Performance Report",
    f"\nDataset: post-2012 Lending Club sample | Split: 2012-2016 train / 2017 val / 2018 test",
    f"Feature count: {len(ALL_FEATURES_USE)} | Decision threshold: {DECISION_THRESHOLD}",
    "\n## Validation Set (2017)",
    f"  ROC-AUC   : {val_metrics['roc']:.4f}",
    f"  PR-AUC    : {val_metrics['pr_auc']:.4f}",
    f"  F1 Score  : {val_metrics['f1']:.4f}",
    f"  Precision : {val_metrics['precision']:.4f}",
    f"  Recall    : {val_metrics['recall']:.4f}",
    "\n## Test Set (2018)",
    f"  ROC-AUC   : {test_metrics['roc']:.4f}",
    f"  PR-AUC    : {test_metrics['pr_auc']:.4f}",
    f"  F1 Score  : {test_metrics['f1']:.4f}",
    f"  Precision : {test_metrics['precision']:.4f}",
    f"  Recall    : {test_metrics['recall']:.4f}",
    f"\n  TN={test_metrics['tn']:,}  FP={test_metrics['fp']:,}",
    f"  FN={test_metrics['fn']:,}  TP={test_metrics['tp']:,}",
]
(REPORT_DIR / 'performance_report.txt').write_text('\n'.join(report_lines))
print(f"✅ Report saved: reports/performance_report.txt")
print("\n🎉 Financial Historian Model pipeline complete!")
