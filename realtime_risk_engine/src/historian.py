from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict

import joblib
import pandas as pd
import xgboost as xgb

from .config import (
    HISTORIAN_FEATURE_MAP_PATH,
    HISTORIAN_MODEL_PATH,
    HISTORIAN_SENTINEL_ANNUAL_INC,
    HISTORIAN_SENTINEL_DTI,
)


class UniversalHistorian:
    """
    Runtime wrapper for the Universal Historian structural risk model.

    The model expects 13 ordered features. Runtime callers provide raw
    structural profile fields and the engine derives the two ratio features.
    """

    RAW_PROFILE_FIELDS = [
        "annual_inc",
        "loan_amnt",
        "dti",
        "term_months",
        "open_acc",
        "total_acc",
        "revol_bal",
        "revol_util",
        "delinq_2yrs",
        "pub_rec",
        "inq_last_6mths",
        "installment",
    ]

    def __init__(self):
        if not os.path.exists(HISTORIAN_MODEL_PATH):
            raise FileNotFoundError(
                f"Universal Historian model not found at {HISTORIAN_MODEL_PATH}"
            )
        if not os.path.exists(HISTORIAN_FEATURE_MAP_PATH):
            raise FileNotFoundError(
                f"Universal Historian feature map not found at {HISTORIAN_FEATURE_MAP_PATH}"
            )

        self.model = joblib.load(HISTORIAN_MODEL_PATH)
        self.feature_names = list(joblib.load(HISTORIAN_FEATURE_MAP_PATH))
        self.model_version = os.path.basename(HISTORIAN_MODEL_PATH)

    def _as_numeric_frame(self, profile: Dict[str, float]) -> pd.DataFrame:
        frame = pd.DataFrame([profile], columns=self.RAW_PROFILE_FIELDS)
        for col in self.RAW_PROFILE_FIELDS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    def _apply_sentinel_rules(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()

        frame["annual_inc"] = frame["annual_inc"].fillna(HISTORIAN_SENTINEL_ANNUAL_INC)
        frame.loc[frame["annual_inc"] < 100, "annual_inc"] = HISTORIAN_SENTINEL_ANNUAL_INC

        frame["dti"] = frame["dti"].fillna(HISTORIAN_SENTINEL_DTI)
        frame.loc[frame["dti"] > 900, "dti"] = HISTORIAN_SENTINEL_DTI

        frame["revol_util"] = frame["revol_util"].fillna(0.0).clip(lower=0.0, upper=100.0)
        frame["delinq_2yrs"] = frame["delinq_2yrs"].fillna(0.0).clip(lower=0.0)
        frame["pub_rec"] = frame["pub_rec"].fillna(0.0).clip(lower=0.0)

        for col in [
            "loan_amnt",
            "term_months",
            "open_acc",
            "total_acc",
            "revol_bal",
            "inq_last_6mths",
            "installment",
        ]:
            frame[col] = frame[col].fillna(0.0).clip(lower=0.0)

        return frame

    def _derive_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        annual_inc = frame["annual_inc"].replace(0.0, HISTORIAN_SENTINEL_ANNUAL_INC)

        frame["loan_to_income_ratio"] = frame["loan_amnt"] / annual_inc
        frame["installment_burden"] = frame["installment"] / (annual_inc / 12.0)
        frame = frame.replace([float("inf"), float("-inf")], pd.NA)

        frame["loan_to_income_ratio"] = frame["loan_to_income_ratio"].fillna(0.0)
        frame["installment_burden"] = frame["installment_burden"].fillna(0.0)
        return frame

    def build_feature_frame(self, profile: Dict[str, float]) -> pd.DataFrame:
        missing = [field for field in self.RAW_PROFILE_FIELDS if field not in profile]
        if missing:
            raise KeyError(f"Missing required historian profile fields: {missing}")

        frame = self._as_numeric_frame(profile)
        frame = self._apply_sentinel_rules(frame)
        frame = self._derive_features(frame)
        return frame[self.feature_names]

    def score_profile(self, profile: Dict[str, float]) -> Dict[str, object]:
        X = self.build_feature_frame(profile)
        probability = float(self.model.predict_proba(X)[0, 1])
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        contribs = self.model.get_booster().predict(dmatrix, pred_contribs=True)
        shap_row = contribs[0]
        shap_values = {
            feature: float(shap_row[idx])
            for idx, feature in enumerate(self.feature_names)
        }
        shap_bias = float(shap_row[len(self.feature_names)])
        return {
            "historian_score": round(probability, 4),
            "historian_scored_at": datetime.now(timezone.utc).isoformat(),
            "historian_model_version": self.model_version,
            "historian_features": {
                feature: float(X.iloc[0][feature]) for feature in self.feature_names
            },
            "historian_shap": shap_values,
            "historian_shap_bias": shap_bias,
        }
