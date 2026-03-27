# 🚀 High-Level Project Workflow

The **Risk-Prediction** repository is a dual-engine credit underwriting system. It allows a financial institution to predict loan defaults using two distinct mathematical strategies depending on the available data.

---

## 🛠️ Phase 1: Engine Selection
An underwriter or developer determines which "Historian" to use based on the applicant's profile:

1.  **Native Historian (`lending_club/`):** 
    *   **Use when:** You have access to official U.S. Credit Bureau data (Experian, TransUnion).
    *   **Power:** Uses a 4-layer ensemble including FICO scores and platform grades.
    
2.  **Universal Historian (`universal_historian/`):** 
    *   **Use when:** You have zero bureau data (Credit Invisible / Thin-File) or want a globally agnostic assessment.
    *   **Power:** Uses a pure 13-feature structural math engine (XGBoost).

---

## 🏗️ Phase 2: Data & Feature Engineering
Both engines follow a strict "Clean-to-Vector" pipeline to prevent the model from learning from corrupt data:

1.  **Raw Ingestion:** Scripts load the massive 1.6GB Lending Club CSV.
2.  **Sanitization:** **Winsorization** caps extreme wealth, and **Sentinel Imputation** fixes data glitches (like $0 income typos).
3.  **Compounding:** The system calculates new ratios like **DTI** and **Installment Burden** to measure the "Self-Correction" capability of the borrower.
4.  **Parquet Export:** The final dense matrix is saved as a memory-mapped `.parquet` to save RAM during the next step.

---

## 🧠 Phase 3: Model Training & Validation
The Python scripts in the `src/` folders execute the machine learning training:

1.  **Temporal Splitting:** The model trains on 2012-2016 and tests on 2017-2018. This ensures the model is predicting the **future**, not memorizing the past.
2.  **Algorithmic Balancing:** We use `scale_pos_weight` to force the AI to be hyper-sensitive to "Default" cases, treating them as 4.7x more important than "Safe" cases.
3.  **Serialization:** The final "brain" is save as a `.pkl` file in the `models/` directory for production use.

---

## 📊 Phase 4: Diagnostic Audit
Before a model is pushed to production, the developer reviews the `outputs/plots/` dashboard:

*   **Sensitivity Check:** Does the model catch at least 80% of defaults? (**Recall**)
*   **Precision Check:** How many "Safe" people are we accidentally blocking? (**Precision**)
*   **Feature Audit:** What are the top 3 drivers of risk? (e.g., `term_months` or `revol_util`).

---

## 🚀 Phase 5: Production Deployment
The final workflow move to the **Inference stage**:

1.  The `models/*.pkl` is loaded into a microservice or web app.
2.  A new applicant's static profile is passed through the same 13 mathematical filters.
3.  The model outputs a probability score (0.0 to 1.0).
4.  A final decision is made based on the project's chosen threshold (e.g., 0.40).
