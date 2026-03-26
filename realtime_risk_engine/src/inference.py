import os
import joblib
import pandas as pd
from .config import MODEL_PATH, F2_OPTIMAL_THRESHOLD
from .feature_engine import RealTimeFeatureEngine
from .transformer import MoneyVisTransformer

class VectorPredictor:
    """
    Main entry point for scoring a batch of UK transactions through the VECTOR brain.
    """
    
    def __init__(self):
        self.transformer = MoneyVisTransformer()
        self.feature_engine = RealTimeFeatureEngine()
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"VECTOR Model not found at {MODEL_PATH}")
            
        # Load the XGBoost artifact
        artifact = joblib.load(MODEL_PATH)
        self.model = artifact["model"]
        self.feature_names = artifact["features"]

    def predict_risk(self, raw_tx_df: pd.DataFrame, ref_date: str = None):
        """
        Takes raw UK DataFrame, transforms, windows, and scores.
        """
        # 1. Standardize
        ledger = self.transformer.transform_batch(raw_tx_df)
        
        # 2. Get reference date (default to latest tx)
        target_date = pd.Timestamp(ref_date) if ref_date else ledger["date"].max()
        
        # 3. Compute Vector signals
        signals = self.feature_engine.compute_signals(ledger, target_date)
        
        if not signals:
            return {"error": "Insufficient history (180 days needed)", "status": "INCOMPLETE_HISTORY"}
            
        # 4. Score through Model
        X = pd.DataFrame([signals])[self.feature_names].values
        prob = self.model.predict_proba(X)[0, 1]
        
        return {
            "account_id": "SIM_USER_001",
            "probability": round(float(prob), 4),
            "is_distressed": bool(prob >= F2_OPTIMAL_THRESHOLD),
            "signals": signals
        }
