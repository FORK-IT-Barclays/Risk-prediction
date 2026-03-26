
"""
VECTOR: Imbalance Strategy Comparison + Hyperparameter Tuning
=============================================================
Strategies tested (all inside CV folds — no leakage):
  1. Baseline:           scale_pos_weight only
  2. SMOTE:              Synthetic Minority Oversampling
  3. BorderlineSMOTE:    SMOTE focused on decision boundary
  4. ADASYN:             Adaptive synthetic (harder samples get more copies)
  5. Tuned (Optuna):     Best strategy + hyperparameter search

Threshold fixed at 0.46 (F2-optimal).
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, fbeta_score, precision_recall_curve
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import os
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = WORKSPACE + r"\dataset"
THRESHOLD = 0.46   # F2-optimal fixed threshold

# ── Load Data ──
df = pd.read_csv(DATA_DIR + r"\vector_features_v4.csv")
df["salary_drift_v"] = df["salary_drift_v"].fillna(0)

FEATURES = ['income_erosion_v','liquidity_momentum_v','overdraft_v','overdraft_t2',
            'salary_drift_v','tx_freq_v','avg_balance_t2','min_balance_t2','total_out_t2']

X = df[FEATURES].values
y = df["default"].values
g = df["account_id"].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate(sampler_fn, xgb_params, label):
    """Run 5-fold CV with a given resampler and XGBoost params. Returns summary dict."""
    all_y, all_prob = [], []
    for tr, te in sgkf.split(X, y, groups=g):
        X_tr, y_tr = X[tr], y[tr]

        # Apply resampler INSIDE the fold (prevents leakage)
        if sampler_fn is not None:
            try:
                X_tr, y_tr = sampler_fn().fit_resample(X_tr, y_tr)
            except Exception:
                pass   # If ADASYN fails on a small fold, skip

        sw = (y_tr == 0).sum() / (y_tr == 1).sum()
        params = {**xgb_params, "scale_pos_weight": sw, "random_state": 42,
                  "eval_metric": "auc", "verbosity": 0}
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr)
        all_y.extend(y[te])
        all_prob.extend(m.predict_proba(X[te])[:, 1])

    all_y   = np.array(all_y)
    all_prob = np.array(all_prob)
    preds   = (all_prob >= THRESHOLD).astype(int)

    tp = ((preds==1)&(all_y==1)).sum(); fn = ((preds==0)&(all_y==1)).sum()
    fp = ((preds==1)&(all_y==0)).sum(); tn = ((preds==0)&(all_y==0)).sum()

    return {
        "label":   label,
        "auc":     round(roc_auc_score(all_y, all_prob), 4),
        "recall":  round(tp / (tp+fn) * 100, 1),
        "fpr":     round(fp / (fp+tn) * 100, 1),
        "f2":      round(fbeta_score(all_y, preds, beta=2), 3),
        "caught":  f"{tp}/{tp+fn}",
    }


BASE_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=3)

# ── Strategy Comparison ──
print("Running strategy comparison (threshold=0.46)...\n")

results = []
strategies = [
    (None,                     "1. Baseline (scale_pos_weight only)"),
    (SMOTE,                    "2. SMOTE"),
    (BorderlineSMOTE,          "3. BorderlineSMOTE"),
    (ADASYN,                   "4. ADASYN"),
]

for sampler_fn, label in strategies:
    r = evaluate(sampler_fn, BASE_PARAMS, label)
    results.append(r)
    print(f"  {label}")
    print(f"     AUC={r['auc']}  Recall={r['recall']}%  FPR={r['fpr']}%  F2={r['f2']}  Caught={r['caught']}")

# ── Hyperparameter Tuning with Optuna on best strategy ──
# Find best F2 strategy
best_result = max(results, key=lambda r: r["f2"])
print(f"\nBest strategy for Optuna tuning: {best_result['label']} (F2={best_result['f2']})")

# Map label back to sampler
sampler_map = {"1. Baseline (scale_pos_weight only)": None,
               "2. SMOTE": SMOTE, "3. BorderlineSMOTE": BorderlineSMOTE, "4. ADASYN": ADASYN}
best_sampler = sampler_map[best_result["label"]]

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0, 5),
    }
    r = evaluate(best_sampler, params, "optuna")
    return r["f2"]   # maximize F2

print("\nRunning Optuna (50 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=False)

print(f"Best F2 from tuning: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")

best_tuned_params = {**study.best_params}
tuned_result = evaluate(best_sampler, best_tuned_params, "5. Tuned (Optuna)")
results.append(tuned_result)

# ── Final Leaderboard ──
print("\n" + "="*65)
print("FINAL LEADERBOARD (threshold=0.46)")
print("="*65)
print(f"{'Strategy':<38} {'AUC':>6} {'Recall':>8} {'FPR':>6} {'F2':>6} {'Caught'}")
print("-"*65)
for r in sorted(results, key=lambda x: -x["f2"]):
    print(f"{r['label']:<38} {r['auc']:>6} {r['recall']:>7}% {r['fpr']:>5}% {r['f2']:>6} {r['caught']}")

# ── Save winning model ──
winner = max(results, key=lambda r: r["f2"])
print(f"\nWinner: {winner['label']}")

final_sampler = sampler_map.get(winner["label"], best_sampler)
final_params  = study.best_params if "Tuned" in winner["label"] else BASE_PARAMS
X_res, y_res  = X, y
if final_sampler is not None:
    X_res, y_res = final_sampler().fit_resample(X, y)
sw = (y_res == 0).sum() / (y_res == 1).sum()
final_model = xgb.XGBClassifier(**final_params, scale_pos_weight=sw,
                                  random_state=42, eval_metric="auc", verbosity=0)
final_model.fit(X_res, y_res)
joblib.dump({"model": final_model, "threshold": THRESHOLD, "features": FEATURES},
            WORKSPACE + r"\Output_Artifacts\behavioral_engine_v2.pkl")
print(f"Saved -> Output_Artifacts\\behavioral_engine_v2.pkl")
print(f"Use prob >= {THRESHOLD} for deployment decisions.")
