"""
VECTOR Full Pipeline Simulation
================================
Creates 10 diverse customer archetypes with realistic risk histories,
runs them through the complete Meta-Physics pipeline, and generates
a deep performance report.

Each customer has:
  - A unique profile (structural data for the Historian)
  - A synthetic risk_history (simulating multiple scoring rounds over 14 days)
  - A distinct risk trajectory archetype (spiral, recovery, stable, volatile, etc.)
"""

import os
import sys
import math
# Ensure imports work from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from physics_engine.meta_physics import MetaPhysicsEngine
from realtime_risk_engine.src.config import META_ZONE_GRID as ZONE_GRID

# ============================================================
#  10 CUSTOMER ARCHETYPES
# ============================================================
# Each customer has a name, profile, and a risk_history trajectory.

engine = MetaPhysicsEngine()
now = datetime.now(timezone.utc)


def build_history(scores, days_back=14):
    """Generate risk_history entries spread over N days."""
    history = []
    interval = days_back / max(len(scores) - 1, 1)
    start = now - timedelta(days=days_back)
    for i, score in enumerate(scores):
        ts = start + timedelta(days=i * interval)
        history.append({
            "calculated_at": ts.isoformat(),
            "final_risk_score": round(score, 4),
            "status": "OK",
            "historian_score": round(score * 0.7, 4),
            "behavioral_score": round(score * 1.2, 4),
        })
    return history


CUSTOMERS = [
    {
        "id": "CUST_001",
        "name": "Sarah Mitchell (Stable Low-Risk)",
        "archetype": "SAFE_STABLE",
        "profile": {
            "annual_inc": 72000, "loan_amnt": 10000, "dti": 12.5,
            "term_months": 36, "open_acc": 4, "total_acc": 11,
            "revol_bal": 3200, "revol_util": 22.0, "delinq_2yrs": 0,
            "pub_rec": 0, "inq_last_6mths": 1, "installment": 350,
        },
        "scores": [0.18, 0.17, 0.19, 0.18, 0.17, 0.18, 0.19, 0.18, 0.17, 0.18, 0.19, 0.18, 0.17, 0.18, 0.18],
    },
    {
        "id": "CUST_002",
        "name": "James Rodriguez (Steady Spiral)",
        "archetype": "WATCH_WORSENING",
        "profile": {
            "annual_inc": 45000, "loan_amnt": 18000, "dti": 28.3,
            "term_months": 60, "open_acc": 8, "total_acc": 20,
            "revol_bal": 12500, "revol_util": 68.0, "delinq_2yrs": 1,
            "pub_rec": 0, "inq_last_6mths": 3, "installment": 580,
        },
        "scores": [0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.55, 0.57, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68],
    },
    {
        "id": "CUST_003",
        "name": "Priya Sharma (Active Recovery)",
        "archetype": "WATCH_RECOVERING",
        "profile": {
            "annual_inc": 58000, "loan_amnt": 14000, "dti": 18.7,
            "term_months": 48, "open_acc": 5, "total_acc": 14,
            "revol_bal": 7800, "revol_util": 41.0, "delinq_2yrs": 1,
            "pub_rec": 0, "inq_last_6mths": 2, "installment": 430,
        },
        "scores": [0.72, 0.70, 0.67, 0.65, 0.62, 0.60, 0.57, 0.55, 0.53, 0.52, 0.50, 0.49, 0.47, 0.46, 0.45],
    },
    {
        "id": "CUST_004",
        "name": "Michael O'Brien (Critical Spiral - Exponential)",
        "archetype": "CRITICAL_SPIRAL",
        "profile": {
            "annual_inc": 32000, "loan_amnt": 22000, "dti": 35.6,
            "term_months": 60, "open_acc": 11, "total_acc": 25,
            "revol_bal": 19500, "revol_util": 82.0, "delinq_2yrs": 3,
            "pub_rec": 1, "inq_last_6mths": 5, "installment": 780,
        },
        "scores": [0.55, 0.57, 0.59, 0.62, 0.65, 0.68, 0.71, 0.73, 0.76, 0.78, 0.80, 0.82, 0.85, 0.87, 0.90],
    },
    {
        "id": "CUST_005",
        "name": "Emily Chen (Volatile - No Clear Trend)",
        "archetype": "VOLATILE",
        "profile": {
            "annual_inc": 65000, "loan_amnt": 12000, "dti": 15.2,
            "term_months": 36, "open_acc": 6, "total_acc": 16,
            "revol_bal": 5400, "revol_util": 30.0, "delinq_2yrs": 0,
            "pub_rec": 0, "inq_last_6mths": 1, "installment": 380,
        },
        "scores": [0.35, 0.55, 0.28, 0.60, 0.31, 0.52, 0.38, 0.48, 0.30, 0.56, 0.34, 0.50, 0.36, 0.45, 0.40],
    },
    {
        "id": "CUST_006",
        "name": "David Kowalski (Critical but Recovering)",
        "archetype": "CRITICAL_RECOVERING",
        "profile": {
            "annual_inc": 48000, "loan_amnt": 20000, "dti": 30.1,
            "term_months": 60, "open_acc": 9, "total_acc": 21,
            "revol_bal": 15200, "revol_util": 72.0, "delinq_2yrs": 2,
            "pub_rec": 0, "inq_last_6mths": 3, "installment": 650,
        },
        "scores": [0.92, 0.90, 0.88, 0.87, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77, 0.76, 0.76],
    },
    {
        "id": "CUST_007",
        "name": "Aisha Patel (Early Warning - Low but Rising)",
        "archetype": "SAFE_WORSENING",
        "profile": {
            "annual_inc": 55000, "loan_amnt": 8000, "dti": 14.8,
            "term_months": 36, "open_acc": 3, "total_acc": 10,
            "revol_bal": 4100, "revol_util": 28.0, "delinq_2yrs": 0,
            "pub_rec": 0, "inq_last_6mths": 1, "installment": 300,
        },
        "scores": [0.15, 0.17, 0.18, 0.20, 0.22, 0.23, 0.25, 0.27, 0.28, 0.30, 0.31, 0.33, 0.35, 0.37, 0.39],
    },
    {
        "id": "CUST_008",
        "name": "Robert Williams (Watch Stable - Plateau)",
        "archetype": "WATCH_STABLE",
        "profile": {
            "annual_inc": 52000, "loan_amnt": 16000, "dti": 22.4,
            "term_months": 48, "open_acc": 7, "total_acc": 17,
            "revol_bal": 9800, "revol_util": 52.0, "delinq_2yrs": 1,
            "pub_rec": 0, "inq_last_6mths": 2, "installment": 510,
        },
        "scores": [0.58, 0.57, 0.59, 0.58, 0.57, 0.58, 0.59, 0.58, 0.57, 0.58, 0.59, 0.58, 0.57, 0.58, 0.58],
    },
    {
        "id": "CUST_009",
        "name": "Lisa Nakamura (Rapid Recovery from Crisis)",
        "archetype": "SAFE_RECOVERING",
        "profile": {
            "annual_inc": 82000, "loan_amnt": 15000, "dti": 13.0,
            "term_months": 36, "open_acc": 5, "total_acc": 13,
            "revol_bal": 6500, "revol_util": 26.0, "delinq_2yrs": 0,
            "pub_rec": 0, "inq_last_6mths": 1, "installment": 480,
        },
        "scores": [0.68, 0.62, 0.56, 0.50, 0.45, 0.41, 0.38, 0.35, 0.32, 0.29, 0.26, 0.24, 0.22, 0.20, 0.18],
    },
    {
        "id": "CUST_010",
        "name": "Carlos Mendez (Accelerating into Default)",
        "archetype": "CRITICAL_SPIRAL",
        "profile": {
            "annual_inc": 38000, "loan_amnt": 25000, "dti": 38.2,
            "term_months": 60, "open_acc": 10, "total_acc": 23,
            "revol_bal": 21000, "revol_util": 88.0, "delinq_2yrs": 2,
            "pub_rec": 1, "inq_last_6mths": 4, "installment": 820,
        },
        # Slow rise then exponential acceleration
        "scores": [0.50, 0.51, 0.52, 0.53, 0.55, 0.57, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92],
    },
]


# ============================================================
#  RUN PIPELINE
# ============================================================
print("=" * 70)
print("  VECTOR FULL PIPELINE SIMULATION - 10 CUSTOMER ARCHETYPES")
print("=" * 70)

results = []
for cust in CUSTOMERS:
    history = build_history(cust["scores"])
    trajectory = engine.analyze(history)
    results.append({
        "customer": cust,
        "trajectory": trajectory,
        "history": history,
    })

# ============================================================
#  DETAILED OUTPUT
# ============================================================
separator = "-" * 70

for r in results:
    cust = r["customer"]
    traj = r["trajectory"]
    scores = cust["scores"]

    print(f"\n{separator}")
    print(f"  {cust['id']}: {cust['name']}")
    print(f"  Expected Archetype: {cust['archetype']}")
    print(f"{separator}")

    # Profile summary
    p = cust["profile"]
    print(f"  Income: ${p['annual_inc']:,.0f}  |  Loan: ${p['loan_amnt']:,.0f}  |  DTI: {p['dti']}%")
    print(f"  Revol Util: {p['revol_util']}%  |  Delinq: {p['delinq_2yrs']}  |  Inquiries: {p['inq_last_6mths']}")

    # Score trajectory visualization
    print(f"\n  Risk Score Trajectory (14 days):")
    bar_width = 40
    for i, s in enumerate(scores):
        bar = "#" * int(s * bar_width)
        marker = " "
        if s >= 0.75:
            marker = "!"
        elif s >= 0.46:
            marker = "~"
        print(f"    Day {i+1:2d}: [{bar:<{bar_width}}] {s:.4f} {marker}")

    # Meta-Physics results
    print(f"\n  META-PHYSICS ENGINE OUTPUT:")
    print(f"    Velocity (B):      {traj['velocity']:+.6f} risk/day")
    if traj['acceleration'] is not None:
        print(f"    Acceleration (B'): {traj['acceleration']:+.6f} risk/day2")
    else:
        print(f"    Acceleration (B'): N/A")
    print(f"    R-Squared:         {traj['r_squared']:.4f}")
    print(f"    Confidence:        {traj['confidence']}")
    print(f"    Trend:             {traj['trend']}")
    print(f"    Current Risk:      {traj['current_risk']:.4f}")

    # DTT
    if traj['days_to_default'] is not None:
        if traj['days_to_default'] == 0:
            print(f"    Days to Default:   0 (ALREADY IN DEFAULT ZONE)")
        else:
            print(f"    Days to Default:   {traj['days_to_default']} days")
    else:
        print(f"    Days to Default:   N/A (safe or unreliable)")

    # Zone
    print(f"    ZONE:              {traj['zone']}")
    print(f"    Window Size:       {traj['trajectory_window']} data points")

    # Match check
    expected = cust['archetype']
    actual = traj['zone']
    match = "MATCH" if actual == expected else "MISMATCH"
    if expected == "VOLATILE" and traj['confidence'] == "LOW":
        match = "MATCH (LOW CONFIDENCE = EXPECTED)"
    print(f"\n    Expected: {expected}")
    print(f"    Actual:   {actual}")
    print(f"    Result:   [{match}]")


# ============================================================
#  PORTFOLIO SUMMARY TABLE
# ============================================================
print(f"\n\n{'=' * 70}")
print("  PORTFOLIO RISK SUMMARY")
print(f"{'=' * 70}")
print(f"  {'Customer':<12} {'Name':<35} {'Score':>6} {'Vel':>8} {'DTT':>5} {'Zone':<22}")
print(f"  {'-'*12} {'-'*35} {'-'*6} {'-'*8} {'-'*5} {'-'*22}")

for r in results:
    cust = r["customer"]
    traj = r["trajectory"]
    dtt = str(traj['days_to_default']) if traj['days_to_default'] is not None else "N/A"
    name = cust['name'].split('(')[0].strip()
    print(f"  {cust['id']:<12} {name:<35} {traj['current_risk']:>6.4f} {traj['velocity']:>+8.4f} {dtt:>5} {traj['zone']:<22}")


# ============================================================
#  ZONE DISTRIBUTION
# ============================================================
print(f"\n\n{'=' * 70}")
print("  9-ZONE DISTRIBUTION ANALYSIS")
print(f"{'=' * 70}")

zone_counts = {}
for r in results:
    zone = r["trajectory"]["zone"]
    zone_counts[zone] = zone_counts.get(zone, 0) + 1

zone_order = [
    "SAFE_STABLE", "SAFE_RECOVERING", "SAFE_WORSENING",
    "WATCH_STABLE", "WATCH_RECOVERING", "WATCH_WORSENING",
    "CRITICAL_STABLE", "CRITICAL_RECOVERING", "CRITICAL_SPIRAL",
]

for zone in zone_order:
    count = zone_counts.get(zone, 0)
    bar = "X" * (count * 5) if count > 0 else ""
    urgency = "LOW" if "SAFE" in zone else ("MEDIUM" if "WATCH" in zone else "HIGH")
    print(f"  {zone:<25} [{bar:<15}] {count} customer(s) -- Urgency: {urgency}")


# ============================================================
#  RISK LEVEL BREAKDOWN
# ============================================================
print(f"\n\n{'=' * 70}")
print("  RISK INTELLIGENCE REPORT")
print(f"{'=' * 70}")

# Categorize
critical = [r for r in results if "CRITICAL" in r["trajectory"]["zone"]]
watch = [r for r in results if "WATCH" in r["trajectory"]["zone"]]
safe = [r for r in results if "SAFE" in r["trajectory"]["zone"]]

print(f"\n  CRITICAL ALERTS ({len(critical)} customers):")
for r in critical:
    c = r["customer"]
    t = r["trajectory"]
    dtt_msg = f"DTT={t['days_to_default']} days" if t['days_to_default'] is not None else "No DTT"
    print(f"    [{c['id']}] {c['name'].split('(')[0].strip()} -- {t['zone']} | {dtt_msg} | R2={t['r_squared']:.2f}")

print(f"\n  WATCH LIST ({len(watch)} customers):")
for r in watch:
    c = r["customer"]
    t = r["trajectory"]
    dtt_msg = f"DTT={t['days_to_default']} days" if t['days_to_default'] is not None else "No DTT"
    print(f"    [{c['id']}] {c['name'].split('(')[0].strip()} -- {t['zone']} | {dtt_msg} | R2={t['r_squared']:.2f}")

print(f"\n  SAFE ({len(safe)} customers):")
for r in safe:
    c = r["customer"]
    t = r["trajectory"]
    print(f"    [{c['id']}] {c['name'].split('(')[0].strip()} -- {t['zone']} | Velocity={t['velocity']:+.4f}")


# ============================================================
#  PIPELINE ACCURACY
# ============================================================
print(f"\n\n{'=' * 70}")
print("  PIPELINE CLASSIFICATION ACCURACY")
print(f"{'=' * 70}")

correct = 0
total_checks = 0
for r in results:
    expected = r["customer"]["archetype"]
    actual = r["trajectory"]["zone"]
    total_checks += 1
    if actual == expected:
        correct += 1
    elif expected == "VOLATILE" and r["trajectory"]["confidence"] == "LOW":
        correct += 1  # Volatile customers should have low confidence

accuracy = (correct / total_checks) * 100 if total_checks > 0 else 0
print(f"\n  Correctly Classified: {correct}/{total_checks}")
print(f"  Pipeline Accuracy:   {accuracy:.1f}%")
print(f"\n{'=' * 70}")
print("  SIMULATION COMPLETE")
print(f"{'=' * 70}")
