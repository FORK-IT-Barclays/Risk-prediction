# FAQ 09: How do we avoid mistakes from "Old Data"?

### ❓ The Question:
To avoid mistakes, shouldn't the model fetch the customer's *current* financial situation—the features as they are "right now"?

### 🔬 The Technical Answer:
Yes! Data **"Freshness"** is the single most critical input for any Financial Historian.

1.  **The "Live Snapshot" Principle:** 
    *   In a production environment, the 13 features are not stored in a static database. They are generated via **Live API Calls** at the moment of the request.
    *   When the borrower clicks "Apply," the system fetches their **Current Balance** and **Current Inquiries** (T=0). 

2.  **API Integration (Plaid & Bureaus):**
    *   To keep the "13 Features" accurate, the system connects to modern financial APIs. This ensures the model is never looking at an "old" income or an "old" debt ratio. It sees the **"Right Now" Reality.**

3.  **Ongoing Validity:**
    *   If a bank wants to catch a "mistake" after the loan has started, they simply "re-ping" the APIs and run the model again. If a borrower's `revol_util` has jumped from 40% to 95% in one month, the model will immediately catch the new risk.

**Conclusion:** The model is only as "smart" as its data is "fresh." By using live snapshots, we ensure the AI is evaluating the borrower's current capacity to pay, not their past history.
