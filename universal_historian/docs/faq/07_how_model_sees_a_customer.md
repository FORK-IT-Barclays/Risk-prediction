# FAQ 07: How does the Model see my "Snapshot"?

### ❓ The Question:
Explain the dataset columns and rows. How does it see a customer's snapshot?

### 🔬 The Technical Answer:
The dataset is a **High-Density Vector Matrix.**

1.  **The Row (The Applicant):** Each row is a single historical person at the moment of application. It is a "Frozen Snapshot" of their financial status.
2.  **The Columns (The Features):** There are **13 Features** that represent the "Stress Ratios" of that life (DTI, Installment Burden, Income).
3.  **The Chronology:** During training, we use a **Chronological Split.** The AI trains on 2012-2016 data and is then tested on 2017-2018 data. This proves the model is not "guessing"—it is successfully identifying structural risk patterns that survived several years into the future.

**Summary:** The AI sees you as a **Value Vector** in a 13-dimensional space. If your vector lands in a region of space that was previously 90% occupied by defaulters, it flags you as High Risk.
