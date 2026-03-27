# Deep-Dive: The VECTOR Meta-Physics Engine (v1.0)

While the **Behavioral Analyst (V&A)** identifies the raw physics of transactions, the **Meta-Physics Engine** analyzes the *Trajectory of the Risk Score itself.* 

It answers the ultimate question: **"If the customer continues exactly as they are today, on what date will they default?"**

---

## 🔬 1. The Core Philosophy
Traditional risk models are **Reactive** (they see a problem after it happened). 
Meta-Physics is **Predictive** (it sees the *momentum* of the problem).

### The "Fever vs. Hot Coffee" Principle (Outlier Detection)
*   **The Problem:** A single high-risk event (a massive one-time withdrawal) can look like a crisis.
*   **The Logic:** A fever is a steady rise in temperature (Trend). Spilling hot coffee on a thermometer is a sudden, random spike (Noise). 
*   **The Solution:** Meta-Physics uses **$R^2$ Confidence Fit** to tell the difference. If the data doesn't fit a straight line, it's just "Hot Coffee."

---

## 📈 2. The Mathematical Engine
The engine uses **Weighted Least Squares (WLS) Regression** to fit the line:  
$$ Risk(t) = \beta t + \alpha $$

### I. The Risk Velocity ($\beta$)
*   **Fixed Window:** Based on our **Phase 3 Benchmarks**, the regression window is now fixed at **14nd days** (The "Goldilocks Zone"). 
*   **The Slope:** If $\beta = 0.02$, it means risk is rising by 2% per day.

### II. The Risk Acceleration ($\beta'$)
*   **The Concept:** This looks at the **Change in the Slope** over the last 7nd days.
*   **The Logic:** If the slope was 0.01 and is now 0.03, the **Acceleration is Positive**.
*   **Alert:** A positive acceleration means the risk is not just rising—it is entering an **Exponential Spiral.**

### III. Time-Decay Weighting
*   We use a **Recency Multiplier**. 
*   Data from today is given **10x the weight** of data from 14nd days ago.
*   *Why?* This allows the model to prioritize "Recent Recovery" or "Recent Shock" without being anchored to "Ancient History."

### III. The $R^2$ Confidence Filter
*   We calculate how well the scatter-plot fits the best-fit line.
*   **High $R^2$ (> 0.85):** A "Crisis Spiral" is in progress. The default projection is high-confidence.
*   **Low $R^2$ (< 0.70):** Volatile behavior. The person is spending randomly. We ignore the projection to prevent "False Alarms."

---

## 🗓️ 3. Forecasting: The "Default Day" Projection
Once we have a high-confidence Slope ($\beta$), we calculate the **Days-to-Threshold (DTT)**:

$$ DTT = \frac{Threshold(0.75) - CurrentRisk}{\beta} $$

*   **Result:** The system outputs: *"User 005 at critical risk in **12 days**."* 
*   **Bank Action:** The bank can offer a loan restructuring on Day 0, preventing the default on Day 12.

---

## 🛡️ 4. The "Recovery Reward" Guard
One of the most powerful features of Meta-Physics is detecting **Positive Change.**
*   If a customer had a bad month but suddenly starts behaving well, the **Slope ($\beta$) becomes Negative.**
*   The engine recognizes the **Negative Momentum** and "forgives" the user’s score faster than a traditional model would.

---
**Status: Blueprint Validated & Production-Ready.**
*Drafted by Antigravity AI - 2026-03-27*
