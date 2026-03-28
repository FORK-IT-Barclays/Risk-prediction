import math
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from realtime_risk_engine.src.config import (
    META_ACCEL_WINDOW,
    META_DEFAULT_THRESHOLD,
    META_MIN_SPAN_HOURS,
    META_R2_HIGH,
    META_R2_LOW,
    META_RECENCY_WEIGHT,
    META_VELOCITY_CAP,
    META_VELOCITY_EPSILON,
    META_WINDOW_DAYS,
    META_ZONE_GRID,
)


class MetaPhysicsEngine:
    """
    Trajectory Analysis Engine (Layer 3).
    Fits weighted regression to historical risk scores to project default dates.
    """

    def analyze(self, risk_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for trajectory analysis.
        Returns a dictionary containing velocity, acceleration, and projection.
        """
        # 1. Initialize result with safe defaults
        result = self._get_empty_result()

        # 2. Parse and validate history
        pairs = self._parse_history(risk_history)
        if len(pairs) < 1:
            return result
            
        if len(pairs) == 1:
            score = pairs[0][1]
            result.update({
                "current_risk": round(score, 4),
                "zone": self._classify_zone(score, 0.0),
                "trajectory_window": 1,
            })
            return result

        # 3. Filter to the regression window (last META_WINDOW_DAYS days)
        latest_time = pairs[-1][0]
        cutoff = latest_time - timedelta(days=META_WINDOW_DAYS)
        filtered = [p for p in pairs if p[0] >= cutoff]
        
        # Current risk is the most recent score
        current_risk = pairs[-1][1]
        
        if len(filtered) < 2:
            # Not enough data in the local window, but can still return current status
            result.update({
                "current_risk": round(current_risk, 4),
                "zone": self._classify_zone(current_risk, 0.0, 0.0),
                "trajectory_window": len(filtered),
            })
            return result

        # 4. Perform Weighted Least Squares (WLS) Regression
        regression_data = self._prepare_regression(filtered)
        if regression_data is None:
            # Handle clustered data (span < 1hr)
            result.update({
                "confidence": "CLUSTER_WAIT (Min 1hr needed)",
                "current_risk": round(current_risk, 4),
                "zone": self._classify_zone(current_risk, 0.0),
                "trajectory_window": len(filtered),
            })
            return result

        beta, r_squared = self._fit_wls(regression_data)
        
        # 5. Compute Acceleration (beta')
        acceleration = self._compute_acceleration(filtered)

        # 6. Classify Dynamics
        trend = self._classify_trend(beta)
        # Pass window count to confidence for Density Guard
        confidence = self._classify_confidence(r_squared, len(filtered)) 
        zone = self._classify_zone(current_risk, beta)

        # 7. Parabolic Default-Date Projection
        days_to_default = self._solve_days_to_default(current_risk, beta, acceleration, r_squared)

        return {
            "velocity": round(beta, 6),
            "acceleration": round(acceleration, 6) if acceleration is not None else 0.0,
            "r_squared": round(r_squared, 4),
            "confidence": confidence,
            "trend": trend,
            "days_to_default": days_to_default,
            "current_risk": round(current_risk, 4),
            "zone": zone,
            "trajectory_window": len(filtered),
            "analysed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _get_empty_result(self) -> Dict[str, Any]:
        """Default structure for insufficient or invalid data."""
        return {
            "velocity": 0.0,
            "acceleration": 0.0,
            "r_squared": 0.0,
            "confidence": "INSUFFICIENT_DATA",
            "trend": "STABLE",
            "days_to_default": None,
            "current_risk": None,
            "zone": "INSUFFICIENT_DATA",
            "trajectory_window": 0,
            "analysed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _parse_history(self, history: List[Dict]) -> List[Tuple[datetime, float]]:
        """Parses and validates time-series data."""
        pairs = []
        for entry in history:
            ts_str = entry.get("calculated_at")
            score = entry.get("final_risk_score")
            if ts_str and score is not None:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    pairs.append((ts, float(score)))
                except (ValueError, TypeError):
                    continue
        return sorted(pairs, key=lambda x: x[0])

    def _prepare_regression(self, pairs: List[Tuple[datetime, float]]) -> Optional[Dict]:
        """Prepares x, y, and weights for WLS."""
        times = [p[0] for p in pairs]
        start_time = times[0]
        end_time = times[-1]
        
        hours_span = (end_time - start_time).total_seconds() / 3600.0
        if hours_span < META_MIN_SPAN_HOURS:
            return None

        # Normalized time in days
        x = [(t - start_time).total_seconds() / 86400.0 for t in times]
        y = [p[1] for p in pairs]
        
        # Exponential-style linspace weights
        weights = np.linspace(1.0, META_RECENCY_WEIGHT, len(x))
        return {"x": x, "y": y, "weights": weights}

    def _fit_wls(self, data: Dict) -> Tuple[float, float]:
        """Fits WLS and returns (slope, r_squared2)."""
        x = sm.add_constant(data["x"])
        y = data["y"]
        w = data["weights"]
        
        if x.shape[1] < 2:
            return 0.0, 0.0

        model = sm.WLS(y, x, weights=w)
        results = model.fit()
        
        beta = float(results.params[1]) if len(results.params) > 1 else 0.0
        r_squared = float(results.rsquared) if not np.isnan(results.rsquared) else 0.0
        
        # Cap velocity
        beta = max(-META_VELOCITY_CAP, min(META_VELOCITY_CAP, beta))
        return beta, r_squared

    def _compute_acceleration(self, pairs: List[Tuple[datetime, float]]) -> float:
        """Computes change in slope over split windows."""
        if len(pairs) < 4:
            return 0.0
            
        latest_time = pairs[-1][0]
        first_time = pairs[0][0]
        total_days = (latest_time - first_time).total_seconds() / 86400.0
        
        # Split point is either T-7 days or the midpoint of the available data
        split_days = min(META_ACCEL_WINDOW, total_days / 2.0)
        mid_point = latest_time - timedelta(days=split_days)
        
        early = [p for p in pairs if p[0] < mid_point]
        recent = [p for p in pairs if p[0] >= mid_point]
        
        if len(early) < 2 or len(recent) < 2:
            return 0.0
            
        def get_slope(p_list):
            dt = (p_list[-1][0] - p_list[0][0]).total_seconds() / 86400.0
            if dt == 0: return 0.0
            return (p_list[-1][1] - p_list[0][1]) / dt
            
        s1 = get_slope(early)
        s2 = get_slope(recent)
        
        # Acceleration = Δv / Δt
        # Δt is the time between the midpoints of the two windows
        t1 = (early[0][0] + (early[-1][0] - early[0][0]) / 2)
        t2 = (recent[0][0] + (recent[-1][0] - recent[0][0]) / 2)
        dt = (t2 - t1).total_seconds() / 86400.0
        
        if dt == 0: return 0.0
        return (s2 - s1) / dt

    def _solve_days_to_default(self, score: float, v: float, a: float, r2: float) -> Optional[int]:
        """Solves quadratic trajectory for default projection."""
        if score >= META_DEFAULT_THRESHOLD:
            return 0
        if r2 < META_R2_LOW:
            return "Inconclusive (Low Confidence)"
        if v <= 0:
            return "Stable / Improving"
            
        gap = META_DEFAULT_THRESHOLD - score
        a_quad = 0.5 * a
        b_quad = v
        c_quad = -gap
        
        t = None
        if abs(a_quad) > 1e-7:
            disc = b_quad**2 - (4 * a_quad * c_quad)
            if disc >= 0:
                roots = [
                    (-b_quad + math.sqrt(disc)) / (2 * a_quad),
                    (-b_quad - math.sqrt(disc)) / (2 * a_quad)
                ]
                valid = [r for r in roots if r > 0]
                if valid:
                    t = min(valid)
        
        if t is None:
            t = gap / v
            
        result = int(math.ceil(t))
        return result if result <= 365 else "No Default Projected (Stable Trajectory)"

    @staticmethod
    def _classify_level(score: float) -> str:
        if score < 0.46: return "LOW"
        if score <= META_DEFAULT_THRESHOLD: return "MID"
        return "HIGH"

    @staticmethod
    def _classify_trend(v: float) -> str:
        """Original 3-state trend logic."""
        if abs(v) <= META_VELOCITY_EPSILON:
            return "STABLE"
        return "WORSENING" if v > 0 else "IMPROVING"

    @classmethod
    def _classify_zone(cls, score: float, v: float) -> str:
        """Reverted 9-zone grid logic."""
        level = cls._classify_level(score)
        trend = cls._classify_trend(v)
        return META_ZONE_GRID.get((level, trend), "UNKNOWN")

    @staticmethod
    def _classify_confidence(r2: float, n_points: int) -> str:
        """
        Maps R2 fit to confidence tiers with a Density Guard.
        A perfect R2 with only 2 points is low-confidence (overfit risk).
        """
        if n_points < 5:
            return "LOW (Sparse Data)"
            
        if r2 >= META_R2_HIGH: return "HIGH"
        if r2 >= META_R2_LOW: return "MODERATE"
        return "LOW"
