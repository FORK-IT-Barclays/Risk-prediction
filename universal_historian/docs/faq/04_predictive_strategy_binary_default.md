# FAQ 04: How is a "Default" officially flagged?

### ❓ The Question:
How did you decide who is a "Defaulter" and who is "Safe" in the training data?

### 🔬 The Technical Answer:
We map dozens of messy statuses into a **Binary Target (0 or 1).**

1.  **Group 0 (The Safe Baseline):** 
    *   **Statuses:** "Fully Paid" and "Current."
    *   **Signal:** These are loans that have either finished successfully or are performing exactly as the contract demands.

2.  **Group 1 (The Default Flag):** 
    *   **Statuses:** "Charged Off," "Default," and "Late (31-120 days)."
    *   **Signal:** These represent cases where the lender has lost money or is mathematically likely to.

3.  **The Filtered Gap:** 
    *   We remove statuses like "In Grace Period." If we don't know for sure if they will default, we don't let the AI learn from them. This ensures the model is trained on **Absolute Certainty.**
