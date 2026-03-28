import sys
import os
import math
import numpy as np
from datetime import datetime, timedelta, timezone

# Ensure imports work from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physics_engine.meta_physics import MetaPhysicsEngine

def audit_archetype(name, scores, days_range=14):
    engine = MetaPhysicsEngine()
    history = []
    now = datetime.now(timezone.utc)
    
    # Generate history
    for i, s in enumerate(scores):
        ts = now - timedelta(days=days_range - (i * (days_range / (len(scores)-1))))
        history.append({
            "calculated_at": ts.isoformat(),
            "final_risk_score": s
        })
    
    res = engine.analyze(history)
    
    print(f"\nAUDIT: {name}")
    print(f"  Input Scores: {scores}")
    print(f"  Velocity:     {res['velocity']:.6f} (Expected units: risk/day)")
    print(f"  Accel:        {res['acceleration']:.6f} (Expected units: risk/day²)")
    print(f"  R-Squared:    {res['r_squared']:.4f}")
    print(f"  Confidence:   {res['confidence']}")
    print(f"  Zone:         {res['zone']}")
    print(f"  DTT:          {res['days_to_default']} days")
    
    return res

def run_full_audit():
    print("="*60)
    print("META-PHYSICS ENGINE RELIABILITY AUDIT (v2.2)")
    print("="*60)

    # 1. Linear Rise (v = 0.02, a = 0)
    # 0.4 -> 0.68 over 14 days
    scores_linear = np.linspace(0.4, 0.68, 15).tolist()
    audit_archetype("LINEAR RISE (14-Day Window)", scores_linear)

    # 2. Perfect Parabolic Spiral (a = 0.004, v = 0)
    # y = 0.5 * 0.004 * t^2 + 0.4
    scores_parabolic = [0.5 * 0.004 * (i**2) + 0.4 for i in range(15)]
    audit_archetype("PURE SPIRAL (a=0.004)", scores_parabolic)

    # 3. Sudden Recovery (Velocity Flip)
    # Day 0-7 rising, Day 8-14 falling
    scores_pivot = [0.4, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.48, 0.44, 0.40, 0.36, 0.32, 0.30, 0.28, 0.26]
    audit_archetype("SUDDEN PIVOT (Recovery Detection)", scores_pivot)

    # 4. Sparse Data Trap (Only 3 points)
    # Should trigger the Density Guard
    scores_sparse = [0.4, 0.5, 0.6]
    audit_archetype("SPARSE DATA (N=3)", scores_sparse)

    # 5. Volatile Noise (R2 should be low)
    scores_noise = [0.4, 0.5, 0.45, 0.55, 0.4, 0.6, 0.4, 0.55, 0.45, 0.5]
    audit_archetype("NOISE FLOOR (High Volatility)", scores_noise)

    print("\n" + "="*60)
    print("AUDIT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_full_audit()
