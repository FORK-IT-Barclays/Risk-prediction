import sys
import os
import random
import math
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

# Ensure imports work from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physics_engine.meta_physics import MetaPhysicsEngine

def generate_random_history(num_tx: int, days_range: int, archetype: str) -> List[Dict[str, Any]]:
    """Generates synthetic risk history for an archetype."""
    history = []
    now = datetime.now(timezone.utc)
    
    # Base risk settings
    if archetype == "SPIRALER":
        base_risk = 0.4
        vel = 0.01
        accel = 0.005
    elif archetype == "SAVER":
        base_risk = 0.6
        vel = -0.01
        accel = -0.002
    elif archetype == "VOLATILE":
        base_risk = 0.5
        vel = 0.0
        accel = 0.0
    else: # STABLE
        base_risk = 0.3
        vel = 0.0
        accel = 0.0

    for i in range(num_tx):
        # Time distribution: uniform over days_range
        days_ago = (days_range / num_tx) * (num_tx - i)
        ts = now - timedelta(days=days_ago)
        
        # Physics-based risk score + some noise
        # 0.5 * a * t^2 + v * t + base
        t = days_range - days_ago
        noise = random.uniform(-0.05, 0.05)
        score = base_risk + (vel * t) + (0.5 * accel * t * t) + noise
        
        # Clamp between 0 and 1
        score = max(0.0, min(1.0, score))
        
        history.append({
            "calculated_at": ts.isoformat(),
            "final_risk_score": score
        })
        
    return history

def run_stress_test(num_users: int = 50, tx_per_user: int = 200):
    engine = MetaPhysicsEngine()
    archetypes = ["SPIRALER", "SAVER", "VOLATILE", "STABLE"]
    results = []

    print(f"Starting Stress Test: {num_users} users, {tx_per_user} tx each...")
    
    start_time = datetime.now()

    for uid in range(num_users):
        arch = random.choice(archetypes)
        history = generate_random_history(tx_per_user, 180, arch)
        
        analysis = engine.analyze(history)
        
        results.append({
            "uid": f"USER_{uid:03d}",
            "archetype": arch,
            "velocity": analysis["velocity"],
            "acceleration": analysis["acceleration"],
            "zone": analysis["zone"],
            "dtt": analysis["days_to_default"],
            "confidence": analysis["confidence"]
        })

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"Stress Test Complete in {duration:.2f}s")
    
    # Save results to CSV for analysis
    df = pd.DataFrame(results)
    df.to_csv("stress_test_results.csv", index=False)
    
    # Summary Statistics
    print("\n" + "="*40)
    print("PHYSICS ENGINE EFFECTIVENESS REPORT")
    print("="*40)
    print(f"Total Users Simulated: {len(df)}")
    print(f"Mean Velocity: {df['velocity'].mean():.6f}")
    print(f"Mean Accel:    {df['acceleration'].mean():.6f}")
    print("\nZone Distribution:")
    print(df['zone'].value_counts())
    print("\nDTT Projections found for:")
    print(df[df['dtt'].notnull()]['archetype'].value_counts())
    
    # Check Spiralers specifically
    spiralers = df[df['archetype'] == "SPIRALER"]
    caught = len(spiralers[spiralers['zone'] == "CRITICAL_SPIRAL"])
    total_spiralers = len(spiralers)
    if total_spiralers > 0:
        print(f"\nSpiral Alert Accuracy (CRITICAL_SPIRAL): {caught/total_spiralers:.1%}")
    
    print("="*40)

if __name__ == "__main__":
    run_stress_test()
