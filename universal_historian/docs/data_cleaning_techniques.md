# 🧹 Advanced Data Cleaning: Winsorization & Sentinel Imputation

These two techniques are the "secret sauce" of the **Universal Financial Historian**. They ensure that Messi-scale wealth or broken data entries don't "hallucinate" the XGBoost model and cause incorrect risk assessments.

---

## 🏗️ 1. Winsorization (Wealth Compression)

### What is it?
Named after the statistician Charles Winsor, this technique "clings" or "crushes" all outlier values to a specific percentile. Instead of **deleting** an extreme row, we **modify** it.

### The "Why":
If one applicant claims an annual income of **$1,000,000,000** (an extreme outlier), a normal model might shift its entire decision-making boundary just to accommodate that one person. This makes the model "stiff" and inaccurate for normal people.

### The Code Implementation:
In our code (`universal_data_pipeline.py`), we use the **99.5th Percentile** as the boundary.
*   **Original:** An applicant claims to make $9.5 Million.
*   **Winsorized:** The system finds that 99.5% of borrowers make below $225k. It overwrites the $9.5M with **exactly $225,000**.
*   **Result:** The AI still sees them as "Very Wealthy," but the number is not so astronomically high that it ruins the mathematical splits for everyone else.

---

## 🛡️ 2. Sentinel Imputation (Flag Correction)

### What is it?
A "Sentinel" is a value that looks like data but is actually a **Flag** (like `999` or `0`). Sentinel Imputation is the process of identifying these "impossible" flags and replacing them with a statistically neutral placeholder.

### The "Why":
Sometimes raw data is simply "broken" or "missing" at the source:
*   **Problem:** Some applicants (typos) might have `annual_inc = 0`.
*   **The Risk:** If we train the AI on this, it might think "People with $0 income are safe."
*   **The Fix:** We identify the sentinel (`annual_inc < 100`) and replace it with the **Population Median** (e.g., $62,000).

### The Implementation in our Pipeline:
*   **Condition:** `annual_inc < 100` $\rightarrow$ `62k (Median)`
*   **Condition:** `dti > 900` (an impossible debt ratio used as a system error flag) $\rightarrow$ `17 (Median)`
*   **Benefit:** The AI never sees "Impossible Physics." It only sees valid, plausible financial profiles, allowing it to learn the *true* correlations between debt and default.

---

## 🎯 Summary Comparison

| Technique | Problem Solved | Method | Behavioral Outcome |
|---|---|---|---|
| **Winsorization** | Billionaires / Outliers | Percetile Capping (99.5%) | Prevents mathematical "hallucination." |
| **Sentinel Imputation** | Typos / Error Flags | Median Replacement | Prevents learning from "Impossible Physics." |
