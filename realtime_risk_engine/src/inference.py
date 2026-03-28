import os
from datetime import datetime, timezone

import joblib
import pandas as pd
import xgboost as xgb
from physics_engine.meta_physics import MetaPhysicsEngine

from .config import BEHAVIORAL_MODEL_PATH, F2_OPTIMAL_THRESHOLD
from .feature_engine import RealTimeFeatureEngine
from .fusion import fuse_scores
from .historian import UniversalHistorian
from .stress_type import classify_stress_from_shap
from .transformer import UniversalTransformer


def _decision_matrix_from_trajectory(trajectory: dict | None) -> dict:
    zone = None if not trajectory else trajectory.get("zone")
    confidence = None if not trajectory else trajectory.get("confidence")

    if not isinstance(zone, str):
        return {
            "zone": zone,
            "confidence": confidence,
            "risk_band": "unknown",
            "intervention_required": False,
            "stress_classification_required": False,
            "reason": "trajectory_unavailable",
        }

    if zone.startswith("SAFE"):
        return {
            "zone": zone,
            "confidence": confidence,
            "risk_band": "low_risk",
            "intervention_required": False,
            "stress_classification_required": False,
            "reason": "low_risk_zone",
        }

    confidence_text = str(confidence or "").upper()
    if "WAIT" in confidence_text or "INSUFFICIENT" in confidence_text:
        return {
            "zone": zone,
            "confidence": confidence,
            "risk_band": "elevated_risk",
            "intervention_required": False,
            "stress_classification_required": False,
            "reason": "trajectory_not_actionable",
        }

    is_high_risk = zone.startswith("CRITICAL")
    is_worsening = zone.endswith("WORSENING") or zone.endswith("SPIRAL")

    if is_high_risk or is_worsening:
        return {
            "zone": zone,
            "confidence": confidence,
            "risk_band": "high_risk" if is_high_risk else "elevated_risk",
            "intervention_required": True,
            "stress_classification_required": True,
            "reason": "high_or_worsening_zone",
        }

    return {
        "zone": zone,
        "confidence": confidence,
        "risk_band": "elevated_risk",
        "intervention_required": False,
        "stress_classification_required": False,
        "reason": "non_worsening_non_high_zone",
    }


def _trajectory_is_actionable(trajectory: dict | None) -> bool:
    return _decision_matrix_from_trajectory(trajectory)[
        "stress_classification_required"
    ]


class RiskEngine:
    """
    Dual-model real-time risk engine.

    It exposes historian-only, behavioral-only, and fused risk inference paths.
    """

    def __init__(self):
        self.transformer = UniversalTransformer()
        self.feature_engine = RealTimeFeatureEngine()
        self.historian = UniversalHistorian()
        self.meta_physics = MetaPhysicsEngine()

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
        """Run the behavioral model on a batch of transactions."""
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
        risk_history: list = None,
    ):
        """
        Unified pipeline entry point.

        Layer 1: Historian (structural baseline)
        Layer 2: Behavioral (velocity signals)
        Fusion:  Combined score
        Layer 3: Meta-Physics (trajectory analysis from risk_history)
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
        calculated_at = datetime.now(timezone.utc).isoformat()
        status = "OK" if final_score is not None else "INSUFFICIENT_DATA"
        trajectory = None
        decision_matrix = None
        stress_profile = None

        if final_score is not None:
            history_for_trajectory = list(risk_history or [])
            history_for_trajectory.append(
                {
                    "calculated_at": calculated_at,
                    "final_risk_score": round(final_score, 4),
                }
            )
            trajectory = self.meta_physics.analyze(history_for_trajectory)
            decision_matrix = _decision_matrix_from_trajectory(trajectory)
            if decision_matrix["stress_classification_required"]:
                stress_profile = classify_stress_from_shap(
                    None
                    if historian_result is None
                    else historian_result.get("historian_shap"),
                    None
                    if behavioral_result is None
                    else behavioral_result.get("behavioral_shap"),
                )

        return {
            "account_id": account_id,
            "historian": historian_result,
            "behavioral": behavioral_result,
            "trajectory": trajectory,
            "decision_matrix": decision_matrix,
            "stress_profile": stress_profile,
            "calculated_at": calculated_at,
            "final_risk_score": None if final_score is None else round(final_score, 4),
            "status": status,
        }


class VectorPredictor(RiskEngine):
    """Backward-compatible alias for earlier behavioral-only integrations."""

    pass
