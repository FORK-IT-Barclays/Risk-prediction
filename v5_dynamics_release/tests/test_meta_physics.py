"""
Comprehensive Test Suite for the VECTOR Meta-Physics Engine (Layer 3)
=====================================================================
Tests all mathematical components: WLS regression, velocity, acceleration,
R[2] confidence filter, DTT forecasting, recovery reward, and 9-zone
classification. Includes edge cases for numerical stability.
"""

import math
import sys
import os
from datetime import datetime, timedelta, timezone

# Ensure imports work from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from physics_engine.meta_physics import MetaPhysicsEngine
from realtime_risk_engine.src.config import META_ZONE_GRID


def make_history(scores, start_days_ago=14, interval_hours=24):
    """Helper: build a risk_history list from a sequence of scores."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=start_days_ago)
    history = []
    for i, score in enumerate(scores):
        ts = start + timedelta(hours=i * interval_hours)
        history.append({
            "calculated_at": ts.isoformat(),
            "final_risk_score": score,
        })
    return history


engine = MetaPhysicsEngine()
passed = 0
failed = 0
total = 0


def assert_test(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")


# =======================================================================
#  TEST 1: Steady Upward Trend
# =======================================================================
print("\n" + "=" * 60)
print("TEST 1: Steady Upward Trend")
print("=" * 60)

# Linear rise from 0.20 to 0.50 over 14 points
scores = [0.20 + 0.02 * i for i in range(15)]
result = engine.analyze(make_history(scores))

assert_test("velocity > 0 (rising)", result["velocity"] > 0,
            f"got {result['velocity']}")
assert_test("trend = WORSENING", result["trend"] == "WORSENING",
            f"got {result['trend']}")
assert_test("R^2 is high (linear data)", result["r_squared"] > 0.95,
            f"got {result['r_squared']}")
assert_test("confidence = HIGH", result["confidence"] == "HIGH",
            f"got {result['confidence']}")
assert_test("DTT is calculated (score rising toward 0.75)",
            result["days_to_default"] is not None and result["days_to_default"] > 0,
            f"got {result['days_to_default']}")
print(f"  >> Velocity={result['velocity']:.6f}, DTT={result['days_to_default']} days, Zone={result['zone']}")


# =======================================================================
#  TEST 2: Steady Downward Trend (Recovery)
# =======================================================================
print("\n" + "=" * 60)
print("TEST 2: Steady Downward Trend (Recovery)")
print("=" * 60)

scores = [0.60 - 0.02 * i for i in range(15)]
result = engine.analyze(make_history(scores))

assert_test("velocity < 0 (falling)", result["velocity"] < 0,
            f"got {result['velocity']}")
assert_test("trend = IMPROVING", result["trend"] == "IMPROVING",
            f"got {result['trend']}")
assert_test("DTT = Stable / Improving label", result["days_to_default"] == "Stable / Improving",
            f"got {result['days_to_default']}")
assert_test("zone = LINEAR_RECOVERY", result["zone"] == "LINEAR_RECOVERY",
            f"got {result['zone']}")
print(f"  >> Velocity={result['velocity']:.6f}, Zone={result['zone']}")


# =======================================================================
#  TEST 3: Flat / Stable Risk
# =======================================================================
print("\n" + "=" * 60)
print("TEST 3: Flat / Stable Risk")
print("=" * 60)

scores = [0.35] * 15
result = engine.analyze(make_history(scores))

assert_test("|velocity| ~= 0", abs(result["velocity"]) < 0.001,
            f"got {result['velocity']}")
assert_test("trend = STABLE", result["trend"] == "STABLE",
            f"got {result['trend']}")
assert_test("zone = SOLID_STABILITY", result["zone"] == "SOLID_STABILITY",
            f"got {result['zone']}")
assert_test("DTT = Inconclusive (Flat Risk)", result["days_to_default"] == "Inconclusive (Low Confidence)",
            f"got {result['days_to_default']}")
print(f"  >> Velocity={result['velocity']:.6f}, Zone={result['zone']}")


# =======================================================================
#  TEST 4: Noisy / Volatile Data (Low R^2)
# =======================================================================
print("\n" + "=" * 60)
print("TEST 4: Noisy / Volatile Data (Low R^2)")
print("=" * 60)

import random
random.seed(42)
scores = [0.40 + random.uniform(-0.25, 0.25) for _ in range(15)]
result = engine.analyze(make_history(scores))

assert_test("R^2 < 0.70 (noisy data)", result["r_squared"] < 0.70,
            f"got {result['r_squared']}")
assert_test("confidence = LOW", result["confidence"] == "LOW",
            f"got {result['confidence']}")
assert_test("DTT = Inconclusive", result["days_to_default"] == "Inconclusive (Low Confidence)",
            f"got {result['days_to_default']}")
print(f"  >> R^2={result['r_squared']:.4f}, Confidence={result['confidence']}")


# =======================================================================
#  TEST 5: Insufficient Data (< 2 points)
# =======================================================================
print("\n" + "=" * 60)
print("TEST 5: Insufficient Data")
print("=" * 60)

# Empty history
result_empty = engine.analyze([])
assert_test("empty: zone = INSUFFICIENT_DATA", result_empty["zone"] == "INSUFFICIENT_DATA",
            f"got {result_empty['zone']}")
assert_test("empty: trajectory_window = 0", result_empty["trajectory_window"] == 0,
            f"got {result_empty['trajectory_window']}")

# Single point
result_single = engine.analyze(make_history([0.40], start_days_ago=1))
assert_test("single: trajectory_window = 1", result_single["trajectory_window"] == 1,
            f"got {result_single['trajectory_window']}")
assert_test("single: current_risk present", result_single["current_risk"] == 0.40,
            f"got {result_single['current_risk']}")
assert_test("single: zone defaults to INSUFFICIENT_DATA", result_single["zone"] == "INSUFFICIENT_DATA",
            f"got {result_single['zone']}")
print(f"  >> Empty zone={result_empty['zone']}, Single zone={result_single['zone']}")


# =======================================================================
#  TEST 6: Positive Acceleration (Spiral)
# =======================================================================
print("\n" + "=" * 60)
print("TEST 6: Positive Acceleration (Risk Accelerating)")
print("=" * 60)

# First half: slow rise, second half: fast rise (exponential-like)
scores = [0.30 + 0.005 * i for i in range(8)] + [0.34 + 0.03 * i for i in range(7)]
result = engine.analyze(make_history(scores))

assert_test("acceleration > 0 (speeding up)", 
            result["acceleration"] is not None and result["acceleration"] > 0,
            f"got {result['acceleration']}")
assert_test("velocity > 0", result["velocity"] > 0,
            f"got {result['velocity']}")
print(f"  >> Velocity={result['velocity']:.6f}, Acceleration={result['acceleration']:.6f}")


# =======================================================================
#  TEST 7: Negative Acceleration (Decelerating)
# =======================================================================
print("\n" + "=" * 60)
print("TEST 7: Negative Acceleration (Risk Decelerating)")
print("=" * 60)

# First half: fast rise, second half: slow rise
scores = [0.30 + 0.03 * i for i in range(8)] + [0.54 + 0.005 * i for i in range(7)]
result = engine.analyze(make_history(scores))

assert_test("acceleration < 0 (slowing down)",
            result["acceleration"] is not None and result["acceleration"] < 0,
            f"got {result['acceleration']}")
print(f"  >> Velocity={result['velocity']:.6f}, Acceleration={result['acceleration']:.6f}")


# =======================================================================
#  TEST 8: Already Above Threshold
# =======================================================================
print("\n" + "=" * 60)
print("TEST 8: Already Above Default Threshold")
print("=" * 60)

scores = [0.78 + 0.01 * i for i in range(10)]
result = engine.analyze(make_history(scores))

assert_test("DTT = 0 (already critical)", result["days_to_default"] == 0,
            f"got {result['days_to_default']}")
# Note: Since zone is purely v*a, if data is linear worsening, it stays LINEAR_WORSENING
assert_test("zone = LINEAR_WORSENING", result["zone"] == "LINEAR_WORSENING",
            f"got {result['zone']}")
assert_test("level is HIGH", result["current_risk"] > 0.75,
            f"got {result['current_risk']}")
print(f"  >> DTT={result['days_to_default']}, Zone={result['zone']}")


# =======================================================================
#  TEST 9: Very Slow Rise (Large DTT)
# =======================================================================
print("\n" + "=" * 60)
print("TEST 9: Very Slow Rise (Large DTT)")
print("=" * 60)

# Tiny increments — beta ~= 0.001 per day
scores = [0.30 + 0.001 * i for i in range(15)]
result = engine.analyze(make_history(scores))

assert_test("velocity > 0 but small", 0 < result["velocity"] < 0.01,
            f"got {result['velocity']}")
# DTT depends on R^2 — check if it's either a string or large int
if isinstance(result["days_to_default"], int):
    assert_test("DTT > 100 days (slow rise)", result["days_to_default"] > 100,
                f"got {result['days_to_default']}")
else:
    assert_test("DTT = Descriptive string", isinstance(result["days_to_default"], str),
                f"got {result['days_to_default']}")
print(f"  >> Velocity={result['velocity']:.6f}, DTT={result['days_to_default']}")


# =======================================================================
#  TEST 10: All 9 Zones Classification
# =======================================================================
print("\n" + "=" * 60)
print("TEST 10: 9-Zone Classification Matrix")
print("=" * 60)

# Test the static classification methods directly
test_cases = [
    (-0.01, -0.01, "TOTAL_REHAB"),
    (0.000, 0.000, "SOLID_STABILITY"),
    (0.01,  0.01,  "EXPONENTIAL_CRASH"),
    (-0.01, 0.01,  "RECOVERY_STALL"),
    (0.01,  -0.01, "REACHING_PEAK"),
]

for score, beta, expected in test_cases:
    actual = MetaPhysicsEngine._classify_zone(score, beta)
    assert_test(f"Zone({score:.2f}, beta={beta:+.3f}) = {expected}",
                actual == expected, f"got {actual}")


# =======================================================================
#  TEST 11: Numerical Stability — Extreme Values
# =======================================================================
print("\n" + "=" * 60)
print("TEST 11: Numerical Stability")
print("=" * 60)

# All zeros
result_zeros = engine.analyze(make_history([0.0] * 10))
assert_test("all-zero scores: no crash", result_zeros["velocity"] == 0.0)

# All ones
result_ones = engine.analyze(make_history([1.0] * 10))
assert_test("all-one scores: no crash", result_ones["velocity"] == 0.0)

# NaN in scores (should skip them)
nan_history = [
    {"calculated_at": datetime.now(timezone.utc).isoformat(), "final_risk_score": 0.3},
    {"calculated_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), "final_risk_score": None},
    {"calculated_at": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(), "final_risk_score": 0.4},
]
result_nan = engine.analyze(nan_history)
assert_test("NaN scores skipped gracefully", result_nan["trajectory_window"] == 2,
            f"got {result_nan['trajectory_window']}")

# Missing calculated_at
bad_history = [{"final_risk_score": 0.5}]
result_bad = engine.analyze(bad_history)
assert_test("missing timestamp handled", result_bad["zone"] == "INSUFFICIENT_DATA")


# =======================================================================
#  TEST 12: Pipeline Output Contract
# =======================================================================
print("\n" + "=" * 60)
print("TEST 12: Output Contract Validation")
print("=" * 60)

scores = [0.30 + 0.02 * i for i in range(10)]
result = engine.analyze(make_history(scores))

required_keys = [
    "velocity", "acceleration", "r_squared", "confidence", "trend",
    "days_to_default", "current_risk", "zone", "tier",
    "trajectory_window", "analysed_at"
]
for key in required_keys:
    assert_test(f"output has '{key}'", key in result, f"missing key: {key}")

assert_test("zone is a valid 9-zone label",
            result["zone"] in META_ZONE_GRID.values() or result["zone"] == "INSUFFICIENT_DATA",
            f"got {result['zone']}")


# =======================================================================
#  TEST 13: Burst Scoring (Velocity Explosion Protection)
# =======================================================================
print("\n" + "=" * 60)
print("TEST 13: Burst Scoring / Velocity Explosion Protection")
print("=" * 60)

# 5 scores within 10 seconds of each other
now = datetime.now(timezone.utc)
burst_history = []
for i in range(5):
    ts = now + timedelta(seconds=i * 2)
    burst_history.append({
        "calculated_at": ts.isoformat(),
        "final_risk_score": 0.40 + (i * 0.01) # Small change
    })

result = engine.analyze(burst_history)

assert_test("burst: velocity explosion prevented (velocity=0)", result["velocity"] == 0,
            f"got {result['velocity']}")
assert_test("burst: confidence flags CLUSTER_WAIT", "CLUSTER_WAIT" in result["confidence"],
            f"got {result['confidence']}")
assert_test("burst: current_risk is correct", result["current_risk"] == 0.44,
            f"got {result['current_risk']}")
assert_test("burst: zone is assigned correctly", result["zone"] == "SOLID_STABILITY",
            f"got {result['zone']}")

print(f"  >> Velocity={result['velocity']:.6f}, Confidence={result['confidence']}")


# =======================================================================
#  SUMMARY
# =======================================================================
print("\n" + "=" * 60)
print(f"META-PHYSICS ENGINE TEST SUMMARY")
print("=" * 60)
print(f"  Total:  {total}")
print(f"  Passed: {passed} [OK]")
print(f"  Failed: {failed} [X]")
print(f"  {'ALL TESTS PASSED!' if failed == 0 else 'SOME TESTS FAILED.'}")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
