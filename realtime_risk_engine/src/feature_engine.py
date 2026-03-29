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
        return float(np.clip((t2 - t1) / (abs(t1) + EPS), -5, 5))

    def compute_signals(self, ledger_df: pd.DataFrame, ref_date: pd.Timestamp) -> Optional[Dict[str, float]]:
        """
        Extracts 9 behavioral signals for a 180-day window ending at ref_date.
        Divided into T1 (first 90 days) and T2 (most recent 90 days).
        """
        # Treat ref_date as the latest included calendar day so newly appended
        # same-day demo transactions actually affect the live score.
        t2_end = ref_date + pd.Timedelta(days=1)
        t2_start = t2_end - pd.Timedelta(days=VELOCITY_WINDOW_DAYS)
        t1_start = t2_start - pd.Timedelta(days=VELOCITY_WINDOW_DAYS)
        
        df_t2 = ledger_df[(ledger_df["date"] >= t2_start) & (ledger_df["date"] < t2_end)]
        df_t1 = ledger_df[(ledger_df["date"] >= t1_start) & (ledger_df["date"] < t2_start)]
        
        # Match Berka training semantics more closely by excluding leakage tags
        # and allowing lightly populated windows to still score.
        df_t1 = df_t1[df_t1["tag"] != "LOAN_PAYMENT"]
        df_t2 = df_t2[df_t2["tag"] != "LOAN_PAYMENT"]
        if len(df_t1) < 2 or len(df_t2) < 2:
            return None
            
        # 1. Fundamental Aggregations
        in_t1, in_t2 = df_t1["cash_in"].sum(), df_t2["cash_in"].sum()
        bal_t1, bal_t2 = df_t1["balance"].mean(), df_t2["balance"].mean()
        od_t1, od_t2 = (df_t1["balance"] < 0).sum(), (df_t2["balance"] < 0).sum()
        
        # 2. Category Specifics
        sal_day_t1 = (
            df_t1[df_t1["tag"] == "SALARY"]["date"].dt.day.median()
            if (df_t1["tag"] == "SALARY").any()
            else np.nan
        )
        sal_day_t2 = (
            df_t2[df_t2["tag"] == "SALARY"]["date"].dt.day.median()
            if (df_t2["tag"] == "SALARY").any()
            else np.nan
        )
        
        # 3. Compute Dimensionless Velocities
        signals = {
            "income_erosion_v":     self.calculate_velocity(in_t1, in_t2),
            "liquidity_momentum_v": self.calculate_velocity(bal_t1, bal_t2),
            "overdraft_v":          float(np.clip(od_t2 - od_t1, -5, 5)),
            "overdraft_t2":         float(od_t2),
            "salary_drift_v":       float(
                (0.0 if np.isnan(sal_day_t2) else sal_day_t2) -
                (0.0 if np.isnan(sal_day_t1) else sal_day_t1)
            ),
            "tx_freq_v":            self.calculate_velocity(len(df_t1), len(df_t2)),
        }
        
        # 4. Scale Absolute Anchors (Applying PPP Scaler for UK Context)
        signals["avg_balance_t2"] = float(bal_t2 * ECONOMIC_PPP_SCALER)
        signals["min_balance_t2"] = float(df_t2["balance"].min() * ECONOMIC_PPP_SCALER)
        signals["total_out_t2"]   = float(df_t2["cash_out"].sum() * ECONOMIC_PPP_SCALER)
        
        return signals
