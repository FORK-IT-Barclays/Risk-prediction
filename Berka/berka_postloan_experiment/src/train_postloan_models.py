from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
FEATURE_PATH = OUTPUT_DIR / "postloan_vector_features.csv"

FEATURES = [
    "income_erosion_v",
    "liquidity_momentum_v",
    "overdraft_v",
    "overdraft_t2",
    "salary_drift_v",
    "tx_freq_v",
    "avg_balance_t2",
    "min_balance_t2",
    "total_out_t2",
]

SEED = 42
CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)


def build_candidates() -> list[tuple[str, str, dict]]:
    candidates: list[tuple[str, str, dict]] = []

    for params in ParameterGrid(
        {
            "model__C": [0.1, 1.0, 5.0, 10.0],
            "model__solver": ["liblinear"],
        }
    ):
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        candidates.append(("LogisticRegression", "plain", {"estimator": estimator, "params": params}))

    for params in ParameterGrid(
        {
            "n_estimators": [300],
            "max_depth": [None, 12],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.7],
        }
    ):
        estimator = RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=SEED,
            **params,
        )
        candidates.append(("RandomForest", "plain", {"estimator": estimator, "params": params}))

    for params in ParameterGrid(
        {
            "n_estimators": [400],
            "max_depth": [None, 12],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.7],
        }
    ):
        estimator = ExtraTreesClassifier(
            class_weight="balanced",
            n_jobs=1,
            random_state=SEED,
            **params,
        )
        candidates.append(("ExtraTrees", "plain", {"estimator": estimator, "params": params}))

    return candidates


def fit_estimator(estimator, fit_mode: str, x_train: np.ndarray, y_train: np.ndarray):
    if fit_mode == "weighted":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        estimator.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        estimator.fit(x_train, y_train)
    return estimator


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    f2_scores = (5 * precision[:-1] * recall[:-1]) / (4 * precision[:-1] + recall[:-1] + 1e-9)
    best_idx = int(np.nanargmax(f2_scores))
    return float(thresholds[best_idx])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def cross_validated_oof(
    estimator,
    fit_mode: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    oof_prob = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in CV.split(x, y, groups):
        fold_model = clone(estimator)
        fit_estimator(fold_model, fit_mode, x[train_idx], y[train_idx])
        oof_prob[test_idx] = fold_model.predict_proba(x[test_idx])[:, 1]

    return oof_prob


def summarize_params(params: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature matrix at {FEATURE_PATH}. Run generate_postloan_features.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_PATH)
    df["salary_drift_v"] = df["salary_drift_v"].fillna(0.0)

    x = df[FEATURES].values
    y = df["default"].values.astype(int)
    groups = df["account_id"].values

    print(f"Training rows: {len(df):,}")
    print(f"Accounts: {df['account_id'].nunique():,}")
    print(f"Default rate: {y.mean() * 100:.2f}%")

    candidates = build_candidates()
    results: list[dict[str, float | str]] = []
    best_result: dict | None = None
    best_estimator = None
    best_fit_mode = "plain"
    best_oof_prob = None

    for idx, (model_name, fit_mode, payload) in enumerate(candidates, start=1):
        estimator = payload["estimator"]
        params = payload["params"]

        print(f"[{idx}/{len(candidates)}] Evaluating {model_name} | {summarize_params(params)}")
        oof_prob = cross_validated_oof(estimator, fit_mode, x, y, groups)
        threshold = tune_threshold(y, oof_prob)
        metrics = compute_metrics(y, oof_prob, threshold)

        result = {
            "model": model_name,
            "params": json.dumps(params, sort_keys=True),
            **metrics,
        }
        results.append(result)

        is_better = (
            best_result is None
            or metrics["f2"] > best_result["f2"]
            or (
                np.isclose(metrics["f2"], best_result["f2"])
                and metrics["roc_auc"] > best_result["roc_auc"]
            )
        )
        if is_better:
            best_result = result
            best_estimator = estimator
            best_fit_mode = fit_mode
            best_oof_prob = oof_prob.copy()

    results_df = pd.DataFrame(results).sort_values(
        by=["f2", "roc_auc", "pr_auc"], ascending=[False, False, False]
    )
    results_path = OUTPUT_DIR / "leaderboard.csv"
    results_df.to_csv(results_path, index=False)

    if best_result is None or best_estimator is None or best_oof_prob is None:
        raise RuntimeError("No model candidate produced a valid result.")

    print("\nTop models:")
    print(results_df[["model", "roc_auc", "pr_auc", "accuracy", "recall", "precision", "f2", "fpr", "threshold"]].head(10).to_string(index=False))

    final_model = clone(best_estimator)
    fit_estimator(final_model, best_fit_mode, x, y)

    artifact = {
        "model": final_model,
        "features": FEATURES,
        "threshold": float(best_result["threshold"]),
        "model_name": best_result["model"],
        "params": json.loads(best_result["params"]),
        "metrics": {
            key: value
            for key, value in best_result.items()
            if key not in {"model", "params"}
        },
    }
    model_path = MODEL_DIR / "postloan_best_model.joblib"
    joblib.dump(artifact, model_path)

    report_lines = [
        "BERKA POST-LOAN EXPERIMENT",
        "=" * 40,
        f"Training rows      : {len(df):,}",
        f"Accounts           : {df['account_id'].nunique():,}",
        f"Default rate       : {y.mean() * 100:.2f}%",
        f"Best model         : {best_result['model']}",
        f"Best params        : {best_result['params']}",
        f"Threshold          : {best_result['threshold']:.4f}",
        f"ROC-AUC            : {best_result['roc_auc']:.4f}",
        f"PR-AUC             : {best_result['pr_auc']:.4f}",
        f"Accuracy           : {best_result['accuracy']:.4f}",
        f"Precision          : {best_result['precision']:.4f}",
        f"Recall             : {best_result['recall']:.4f}",
        f"F2                 : {best_result['f2']:.4f}",
        f"FPR                : {best_result['fpr']:.4f}",
        f"TP / FP / TN / FN  : {best_result['tp']} / {best_result['fp']} / {best_result['tn']} / {best_result['fn']}",
        f"Leaderboard        : {results_path}",
        f"Model artifact     : {model_path}",
    ]
    report_path = OUTPUT_DIR / "training_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    roc_fpr, roc_tpr, _ = roc_curve(y, best_oof_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(roc_fpr, roc_tpr, label=f"{best_result['model']} (AUC={best_result['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Post-Loan OOF ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_model_roc.png", dpi=150)
    plt.close()

    threshold_grid = np.linspace(0.05, 0.95, 91)
    recalls = []
    precisions = []
    f2_scores = []
    for threshold in threshold_grid:
        metrics = compute_metrics(y, best_oof_prob, float(threshold))
        recalls.append(metrics["recall"])
        precisions.append(metrics["precision"])
        f2_scores.append(metrics["f2"])

    plt.figure(figsize=(9, 5))
    plt.plot(threshold_grid, recalls, label="Recall")
    plt.plot(threshold_grid, precisions, label="Precision")
    plt.plot(threshold_grid, f2_scores, label="F2")
    plt.axvline(best_result["threshold"], linestyle="--", color="red", label="Chosen threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sensitivity (OOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_sensitivity.png", dpi=150)
    plt.close()

    if hasattr(final_model, "feature_importances_"):
        importance = np.asarray(final_model.feature_importances_)
    elif hasattr(final_model, "named_steps") and hasattr(final_model.named_steps["model"], "coef_"):
        importance = np.abs(final_model.named_steps["model"].coef_[0])
    else:
        importance = None

    if importance is not None:
        feat_df = pd.DataFrame({"feature": FEATURES, "importance": importance}).sort_values(
            by="importance", ascending=True
        )
        plt.figure(figsize=(8, 5))
        plt.barh(feat_df["feature"], feat_df["importance"])
        plt.xlabel("Importance")
        plt.title("Best Model Feature Importance")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
        plt.close()

    print("\nBest model summary:")
    for line in report_lines[3:14]:
        print(line)


if __name__ == "__main__":
    main()
