# Technical Deep-Dive: The Physics of Risk (V & A)

In the **VECTOR** architecture, we move beyond static balances to analyze the **Trajectory** of a customer's financial health using Velocity and Acceleration.

---

## 1. Risk Velocity (V)
**Definition:** The speed at which a behavioral signal is changing over a specific time window.

**Formula:**
`V = (Signal_Current - Signal_Prior) / Time_Delta`

*   **Example (Liquidity Erosion):** If a customer's savings drop from $5,000 to $3,500 in 15 days:
    *   `V = ($3,500 - $5,000) / 15 = -100 $/day`
*   **Threshold:** High negative velocity in savings, combined with high positive velocity in credit utilization, is the strongest possible leading indicator of default.

---

## 2. Risk Acceleration (A)
**Definition:** The rate at which the Velocity of a signal is increasing. This identifies the "Crisis Spiral."

**Formula:**
`A = (V_Current - V_Prior) / Time_Delta`

*   **Positive Acceleration (Crisis):** If the rate of spending is increasing (e.g., they were losing $50/day last week, but are losing $150/day this week), `A` is positive. This indicates an **Exponential Collapse.**
*   **The Signal:** Acceleration allows us to distinguish between a "One-time Expense" (Velocity spikes then drops) and a "Systemic Failure" (Velocity keeps increasing).

---

## 3. The 7 Behavioral Vectors
We apply V and A to the following 7 high-frequency signals:

1.  **Salary Timing Drift:** Calculated as `V = Date_Actual - Date_Median`. Deviations > 3 days trigger a warning.
2.  **Income Erosion:** Negative MoM (Month-over-Month) income velocity.
3.  **Liquidity Momentum:** Week-over-week depletion speed of the main current account.
4.  **Payment Integrity:** Drift acceleration in utility/bill payment dates.
5.  **Failed Auto-Debits:** Frequency acceleration of technical or insufficient fund bounces.
6.  **Credit Exhaustion:** The acceleration of utilization towards the hard limit.
7.  **Digital Silence:** Decay velocity in mobile app logins (Behavioral avoidance).

---

## ⚖️ The 9-Zone Risk Matrix
Based on the intersection of **FH (Structural Stability)** and **V/A (Behavioral Velocity)**, we map every customer into one of 9 zones:

*   **Zone 1: STABLE** (High Buffer, Low Velocity)
*   **Zone 5: SLIDING** (Medium Buffer, Accelerating Risk)
*   **Zone 9: SPIRALING** (Low Buffer, Max Velocity) -> **Target for Immediate GenAI Empathy Outreach.**
