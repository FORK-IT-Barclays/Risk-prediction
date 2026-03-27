# FAQ 08: How do "Life Changes" after the loan affect the model?

### ❓ The Question:
If the input data is static (who they are on Day 1), but their finances change *after* they get the loan (job loss, more debt), how does the model handle it?

### 🔬 The Technical Answer:
The Financial Historian is built on the concept of **Structural Resiliency.**

1.  **The Underwriting Snapshot:** 
    *   This model is designed for **"Day Zero" decision-making.** Its goal is to filter out anyone whose financial foundation is already "Brittle."

2.  **Structural Buffer vs. The Cliff:**
    *   **High-Resiliency Borrower:** If a borrower has a low `installment_burden` and high income, they have a "Structural Buffer." Even if their expenses go up later, they are unlikely to default because their base foundation is wide. 
    *   **Low-Resiliency Borrower:** If a borrower is already maxed out (`revol_util > 90%`) on Day 1, they have **Zero Buffer.** Any tiny change in their life after the loan will cause them to "fall off the cliff." The AI identifies this brittleness immediately and blocks them.

3.  **Underwriting vs. Lifecycle Monitoring:**
    *   To catch changes *during* the active loan, you would implement **Lifecycle Monitoring.** You would run the customer through the model again every 3 months. If their features have drifted into a "Red Zone," the bank can then intervene (e.g., lower their credit limit) before they officially default.

**Summary:** The model doesn't predict "The Future"; it predicts **"Survival Probability."** It determines if the borrower's financial structure is strong enough to survive the normal volatility of human life!
