# FAQ 03: What are Winsorization and Sentinel Imputation?

### ❓ The Question:
What do these data cleaning terms mean and why are they required?

### 🔬 The Technical Answer:
They are "Sanity Filters" that protect the AI from learning from impossible or extreme data.

1.  **Winsorization (Outlier Capping):** 
    *   **The Problem:** Billionaires or data-typos for high income can ruin a model's math.
    *   **The Fix:** We "crush" extreme values down to the **99.5th percentile.** An applicant claiming $10,000,000 is overwritten to $225,000.
    *   **Result:** The AI sees they are rich, but it doesn't let their extreme numbers skew the logic for everyone else.

2.  **Sentinel Imputation (Flag Fixing):** 
    *   **The Problem:** Databases often use flags like `annual_inc = 0` or typos for missing data. 
    *   **The Fix:** We identify these "impossible" values and replace them with the **Population Median.**
    *   **Result:** The AI never sees "Impossible Physics" (like a borrower with zero income or 900% debt), ensuring all learned patterns are realistic.
