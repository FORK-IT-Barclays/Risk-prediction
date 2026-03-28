import os
from datetime import datetime, timezone

import joblib
import pandas as pd
import xgboost as xgb

from .config import BEHAVIORAL_MODEL_PATH, F2_OPTIMAL_THRESHOLD
from .feature_engine import RealTimeFeatureEngine
from .fusion import fuse_scores
from .historian import UniversalHistorian
from .stress_type import classify_stress_from_shap
from .transformer import MoneyVisTransformer


class RiskEngine:
    """
    Dual-model real-time risk engine.

    It exposes historian-only, behavioral-only, and fused risk inference paths.
    """

    def __init__(self):
        self.transformer = MoneyVisTransformer()
        self.feature_engine = RealTimeFeatureEngine()
        self.historian = UniversalHistorian()

        if not os.path.exists(BEHAVIORAL_MODEL_PATH):
            raise FileNotFoundError(
                f"VECTOR behavioral model not found at {BEHAVIORAL_MODEL_PATH}"
            )

        artifact = joblib.load(BEHAVIORAL_MODEL_PATH)
        self.behavioral_model = artifact["model"]
        self.behavioral_threshold = float(
            artifact.get("threshold", F2_OPTIMAL_THRESHOLD)
        )
        self.behavioral_feature_names = list(artifact["features"])
        self.behavioral_model_version = os.path.basename(BEHAVIORAL_MODEL_PATH)

    def score_profile(self, profile: dict):
        """Run the structural baseline model on a raw profile payload."""
        return self.historian.score_profile(profile)

    def score_behavioral(
        self,
        raw_tx_df: pd.DataFrame,
        ref_date: str = None,
        account_id: str = "SIM_USER_001",
    ):
        """Run the behavioral model on a batch of UK-style transactions."""
        ledger = self.transformer.transform_batch(raw_tx_df)
        target_date = pd.Timestamp(ref_date) if ref_date else ledger["date"].max()
        signals = self.feature_engine.compute_signals(ledger, target_date)

        if not signals:
            return {
                "account_id": account_id,
                "behavioral_score": None,
                "behavioral_scored_at": None,
                "behavioral_model_version": self.behavioral_model_version,
                "status": "INCOMPLETE_HISTORY",
                "error": "Insufficient history (180 days needed)",
            }

        X = pd.DataFrame([signals])[self.behavioral_feature_names]
        prob = float(self.behavioral_model.predict_proba(X)[0, 1])
        dmatrix = xgb.DMatrix(X, feature_names=self.behavioral_feature_names)
        contribs = self.behavioral_model.get_booster().predict(
            dmatrix, pred_contribs=True
        )
        shap_row = contribs[0]
        shap_values = {
            feature: float(shap_row[idx])
            for idx, feature in enumerate(self.behavioral_feature_names)
        }
        shap_bias = float(shap_row[len(self.behavioral_feature_names)])

        return {
            "account_id": account_id,
            "behavioral_score": round(prob, 4),
            "behavioral_scored_at": datetime.now(timezone.utc).isoformat(),
            "behavioral_model_version": self.behavioral_model_version,
            "is_distressed": bool(prob >= self.behavioral_threshold),
            "signals": signals,
            "behavioral_shap": shap_values,
            "behavioral_shap_bias": shap_bias,
            "status": "OK",
        }

    def predict_risk(
        self,
        raw_tx_df: pd.DataFrame = None,
        profile: dict = None,
        ref_date: str = None,
        account_id: str = "SIM_USER_001",
    ):
        """
        Unified entry point returning structural, behavioral, and fused scores.
        """
        historian_result = None
        behavioral_result = None

        if profile is not None:
            historian_result = self.score_profile(profile)
        if raw_tx_df is not None:
            behavioral_result = self.score_behavioral(
                raw_tx_df, ref_date=ref_date, account_id=account_id
            )

        historian_score = (
            None if historian_result is None else historian_result["historian_score"]
        )
        behavioral_score = None
        if behavioral_result is not None:
            behavioral_score = behavioral_result.get("behavioral_score")

        final_score = fuse_scores(historian_score, behavioral_score)
        status = "OK" if final_score is not None else "INSUFFICIENT_DATA"
        stress_profile = classify_stress_from_shap(
            None if historian_result is None else historian_result.get("historian_shap"),
            None if behavioral_result is None else behavioral_result.get("behavioral_shap"),
        )

        return {
            "account_id": account_id,
            "historian": historian_result,
            "behavioral": behavioral_result,
            "stress_profile": stress_profile,
            "final_risk_score": None if final_score is None else round(final_score, 4),
            "status": status,
        }


class VectorPredictor(RiskEngine):
    """Backward-compatible alias for earlier behavioral-only integrations."""

    pass
