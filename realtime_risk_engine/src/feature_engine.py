import pandas as pd
import numpy as np
from typing import Optional, Dict
from .config import ECONOMIC_PPP_SCALER, EPS, VELOCITY_WINDOW_DAYS

class RealTimeFeatureEngine:
    """
    Computes VECTOR Velocity signals for a single account's transaction history.
    """
    
    def calculate_velocity(self, t1: float, t2: float) -> float:
        """Dimensionless rate of change between two periods."""
        return float(((t2 - t1) / (abs(t1) + EPS)).clip(-5, 5))

    def compute_signals(self, ledger_df: pd.DataFrame, ref_date: pd.Timestamp) -> Optional[Dict[str, float]]:
        """
        Extracts 9 behavioral signals for a 180-day window ending at ref_date.
        Divided into T1 (first 90 days) and T2 (most recent 90 days).
        """
        t2_end = ref_date
        t2_start = t2_end - pd.Timedelta(days=VELOCITY_WINDOW_DAYS)
        t1_start = t2_start - pd.Timedelta(days=VELOCITY_WINDOW_DAYS)
        
        df_t2 = ledger_df[(ledger_df["date"] >= t2_start) & (ledger_df["date"] < t2_end)]
        df_t1 = ledger_df[(ledger_df["date"] >= t1_start) & (ledger_df["date"] < t2_start)]
        
        # Quality Gate: Ensure minimum transaction density in both periods
        if len(df_t1) < 3 or len(df_t2) < 3:
            return None
            
        # 1. Fundamental Aggregations
        in_t1, in_t2 = df_t1["cash_in"].sum(), df_t2["cash_in"].sum()
        bal_t1, bal_t2 = df_t1["balance"].mean(), df_t2["balance"].mean()
        od_t1, od_t2 = (df_t1["balance"] < 0).sum(), (df_t2["balance"] < 0).sum()
        
        # 2. Category Specifics
        sal_t1 = df_t1[df_t1["tag"] == "SALARY"]["cash_in"].sum()
        sal_t2 = df_t2[df_t2["tag"] == "SALARY"]["cash_in"].sum()
        
        # 3. Compute Dimensionless Velocities
        signals = {
            "income_erosion_v":     self.calculate_velocity(in_t1, in_t2),
            "liquidity_momentum_v": self.calculate_velocity(bal_t1, bal_t2),
            "overdraft_v":          float(np.clip(od_t2 - od_t1, -5, 5)),
            "overdraft_t2":         float(od_t2),
            "salary_drift_v":       self.calculate_velocity(sal_t1, sal_t2),
            "tx_freq_v":            self.calculate_velocity(len(df_t1), len(df_t2)),
        }
        
        # 4. Scale Absolute Anchors (Applying PPP Scaler for UK Context)
        signals["avg_balance_t2"] = float(bal_t2 * ECONOMIC_PPP_SCALER)
        signals["min_balance_t2"] = float(df_t2["balance"].min() * ECONOMIC_PPP_SCALER)
        signals["total_out_t2"]   = float(df_t2["cash_out"].sum() * ECONOMIC_PPP_SCALER)
        
        return signals
