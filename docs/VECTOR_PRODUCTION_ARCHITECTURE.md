# VECTOR: The Production-Grade Risk Engine Architecture (v3.0)

This document provides the full technical overview of the **VECTOR Pre-Delinquency Shield**. It is a **"Multi-Expert Waterfall"** that identifies financial death spirals across three distinct temporal layers.

---

## 🏛️ Layer 1: Structural Anchor (The Historian)
*   **Model:** XGBoost (Proprietary-Free / Static)
*   **Goal:** Determine the "Floor" of the house.
*   **Key Inputs:** DTI, Absolute Income, Total Debt Burden, Credit Utilization.
*   **Output:** `Historian_Prob` (The 0-1 baseline risk). 
*   *Role: Detects if a customer has a structural lack of "Financial Buffer".*

---

## 🌀 Layer 2: Behavioral Physics (The Analyst)
*   **Model:** CatBoost / V2 Feature Engine (Real-Time)
*   **Goal:** Determine the "Velocity" and "Acceleration" of decay.
*   **Key Signals:**
    *   **Macro-V (30-day):** Is the savings rate sliding month-over-month?
    *   **Micro-V (14-day):** Is there a sudden spending spike in the current window?
    *   **Instant Events:** Zero-latency triggers for **Failed Auto-Debits** or **Overdrafts**.
*   **Output:** `Behavioral_Prob` (Shift relative to the **0.46** threshold).

---

## 📈 Layer 3: Meta-Physics (The Trajectory Engine)
*   **Model:** Weighted Least-Squares (WLS) Regression
*   **Goal:** Predict the **Default Day** and calculate **Risk Acceleration**.
*   **The Logic:**
    1.  Collect risk scores over the **Fixed 14nd-Day Window**.
    2.  Fit a **Best-Fit Line** weighted by recency.
    3.  **The $R^2$ Guard:** If the line fit is poor, ignore the trend (Prevents "Wedding Spikes"). 
    4.  **Forecasting:** Project the date when the line will hit the **0.75 (Critical)** limit.
*   **Outputs:** `Risk_Velocity` ($\beta$), `Risk_Acceleration` ($\beta'$), `Days_to_Potential_Default`.

---

## 🎯 Unified Consensus Formula
The final score visible to the bank is a synthesis of all layers:

$$ UnifiedScore = HistorianProb + (BehavioralProb - 0.46) \times W_{Behav} + InstantShock $$

### Output Classifications:
*   **STABLE:** No trend detected; Score < 0.46.
*   **⚠️ BEHAVIORAL_SLIDE:** Negative Velocity detected; Score 0.46–0.75. (Intervene via Nudge).
*   **🚨 CRITICAL_SHOCK:** Immediate Event or Unified_Score > 0.75. (Intervene via Pay Plan).

---
**Status: Blueprint Validated & Production-Ready.**
*Drafted by Antigravity AI - 2026-03-27*
