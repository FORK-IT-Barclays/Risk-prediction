# VECTOR 9-Zone Risk Classification System

## Overview

The 9-Zone Classification is the human-readable output of the Meta-Physics Engine. It maps every customer into one of 9 zones based on two axes:
- **Risk Level:** Where the customer's score is right now (LOW / MID / HIGH)
- **Trend:** Which direction the score is moving (IMPROVING / STABLE / WORSENING)

---

## The 9-Zone Grid

| | LOW (<0.46) | MID (0.46-0.75) | HIGH (>0.75) |
|---|---|---|---|
| **IMPROVING** (B < -0.005) | SAFE_RECOVERING | WATCH_RECOVERING | CRITICAL_RECOVERING |
| **STABLE** (\|B\| <= 0.005) | SAFE_STABLE | WATCH_STABLE | CRITICAL_STABLE |
| **WORSENING** (B > 0.005) | SAFE_WORSENING | WATCH_WORSENING | CRITICAL_SPIRAL |

---

## Zone Definitions & Bank Actions

### GREEN Zones (Low Urgency)

**SAFE_STABLE** — Score < 0.46, no trend
- Low risk, no intervention needed
- Action: Standard monitoring

**SAFE_RECOVERING** — Score < 0.46, improving
- Previously stressed customer returning to normal
- Action: None. Recovery reward in effect.

**SAFE_WORSENING** — Score < 0.46, rising
- Early warning. Score still safe but trajectory is concerning.
- Action: Watchlist. Schedule proactive check-in.

### YELLOW Zones (Medium Urgency)

**WATCH_STABLE** — Score 0.46-0.75, no trend
- Elevated risk, not moving
- Action: Enhanced monitoring. Review monthly.

**WATCH_RECOVERING** — Score 0.46-0.75, improving
- Was in distress but actively recovering
- Action: Supportive. Consider flexible restructuring.

**WATCH_WORSENING** — Score 0.46-0.75, rising
- Active slide toward default. DTT projection available.
- Action: Intervene via Nudge. Contact customer with options.

### RED Zones (High Urgency)

**CRITICAL_STABLE** — Score > 0.75, not moving
- At maximum risk, but not accelerating
- Action: Immediate review. Consider debt restructuring.

**CRITICAL_RECOVERING** — Score > 0.75, improving
- Was in crisis, now recovering
- Action: Supportive restructuring. Extend payment plan.

**CRITICAL_SPIRAL** — Score > 0.75, accelerating
- The most dangerous zone. Active exponential decay.
- Action: URGENT intervention. Pay plan, loan restructuring, or collections.
- DTT projection: "N days to default" available with high confidence.

---

## Threshold Configuration

All thresholds are configurable in `config.py`:

| Constant | Value | Purpose |
|---|---|---|
| `META_WINDOW_DAYS` | 14 | WLS regression window |
| `META_VELOCITY_EPSILON` | 0.005 | STABLE zone boundary |
| `META_R2_LOW` | 0.70 | Below this, suppress DTT |
| `META_R2_HIGH` | 0.85 | Above this, high-confidence spiral |
| `META_DEFAULT_THRESHOLD` | 0.75 | Risk score for default projection |
| `META_RECENCY_WEIGHT` | 10.0 | Today = 10x weight of oldest |
| `F2_OPTIMAL_THRESHOLD` | 0.46 | LOW/MID boundary |

---

## Output Format

Each `/risk-score` API response now includes a `trajectory` object:

```json
{
  "account_id": "CUST_001",
  "final_risk_score": 0.5832,
  "trajectory": {
    "velocity": 0.018,
    "acceleration": 0.003,
    "r_squared": 0.92,
    "confidence": "HIGH",
    "trend": "WORSENING",
    "days_to_default": 9,
    "current_risk": 0.5832,
    "zone": "WATCH_WORSENING",
    "trajectory_window": 7
  }
}
```

---

*VECTOR 9-Zone Classification System v1.0*
*Integrated with Meta-Physics Engine - 2026-03-27*
