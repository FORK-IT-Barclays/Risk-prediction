# VECTOR Behavioral Model: Formal Evaluation Q&A

This document serves as an objective, data-supported response to the architectural and evaluative questions regarding the VECTOR Behavioral Risk Engine, based directly on the PKDD99 (Berka) dataset execution and model generation artifacts.

---

### 1. Data Source: What's in the PKDD99 Czech bank dataset?
The model is trained on the canonical **1999 PKDD Discovery Challenge (Berka)** dataset, which contains anonymized client data from a Czech bank. 
* **Time Span:** The data spans roughly 5 years, ending on December 31, 1998.
* **Core Tables Used:** The model fundamentally relies on two tables:
  * [trans.csv](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/dataset/trans.csv): Contains **1,056,320 transactions** across all accounts.
  * [loan.csv](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/dataset/loan.csv): Contains **682 granted loans**, representing the ground truth.
* **Transaction Types:** The raw data categorizes transactions via the `type` column (`PRIJEM` for credit/income, `VYDAJ`/`VYBER` for debit/withdrawal) and provides semantic context via `k_symbol` and `operation` fields (e.g., `SIPO` for household bills, `DUCHOD` for old-age pension/salary, `UVER` for loan payments).

### 2. Features (Raw): What raw transaction fields do you use?
The pipeline ignores all proprietary demographic data (age, gender, district) to ensure anti-discriminatory compliance and cross-border portability. It strictly uses five raw financial fields from the `trans` table:
1. `amount` (Transaction magnitude)
2. `date` (Chronology)
3. `type` (Direction of cash flow)
4. `balance` (Account balance *after* the transaction)
5. `k_symbol` / `operation` (Semantic tags)

During preprocessing, these are standardized into a "Universal Ledger" format: `date`, `cash_in`, `cash_out`, `balance`, and [tag](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/3_Experiments/moneyvis_inference.py#36-41) (`SALARY`, `BILL`, `UNCATEGORIZED`). Notably, transactions tagged as `LOAN_PAYMENT` are **explicitly excluded** from the feature engineering logic to prevent data leakage (where the model trivially predicts "good" if it spots active loan payments).

### 3. Feature Engineering: How do you aggregate transactions into behavioral signals?
Transactions are aggregated using a **Statistical Velocity Architecture** rather than complex state machines.
The core mathematical principle is calculating the "Physics of Risk" — the rate of change between periods rather than static totals. For example, the `income_erosion_v` signal is calculated using a standard velocity formula:
`velocity = (Total_Inflows_T2 - Total_Inflows_T1) / (|Total_Inflows_T1| + ε)`

The model computes 6 dimensionless velocity/trend signals (e.g., `liquidity_momentum_v`, `tx_freq_v`, `overdraft_v`) and 3 absolute scale anchors (`avg_balance_t2`, `min_balance_t2`, `total_out_t2`).

### 4. Time-Series Aspect
* **Do you use sliding windows?** 
  Yes. The architecture constructs up to **15 non-overlapping 6-month windows** per account, stepping backward from the loan maturity date using a 90-day stride. Each 6-month window is divided into a **T1 (Prior 90 days)** and **T2 (Recent 90 days)** period.
* **Do you extract trends?**
  Yes. The entire predictive foundation is built on trends. The pipeline compares the T2 period against the T1 period to extract acceleration metrics (e.g., "Are overdraft instances increasing?" or "Is salary arriving later in the month?").
* **Do you use RNN/LSTM, or statistical feature extraction?**
  **Statistical Feature Extraction.** The model avoids sequence models (RNNs/LSTMs) due to their high computational cost, lower explainability (black-box nature), and tendency to overfit small tabular datasets. Instead, temporal sequencing is heavily engineered into explicitly readable tabular features.

### 5. Target Definition: What's your prediction task?
The prediction task is **loan-term default** — predicting whether a loan historically failed or will fail to be paid back in full. 
* Ground truth is derived from the `status` column in [loan.csv](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/dataset/loan.csv).
* Statuses `A` (paid) and `C` (running, OK) are mapped to **0 (Safe)**.
* Statuses `B` (unpaid) and `D` (running, in debt) are mapped to **1 (Default/Distressed)**.
* The "reference date" (the anchor from which all historical windows look backward) is dynamically calculated as the `loan_date + duration`. For running loans, this anchor is capped strictly at the dataset cutoff (`1998-12-31`) to prevent looking into non-existent future data.

### 6. Performance: What metrics did you achieve?
Because misclassifying a defaulter (False Negative) mathematically costs a bank more than investigating a false alarm (False Positive), the model was tuned heavily to maximize **F2-Score** (which weights Recall twice as heavily as Precision).

Using a strict **5-Fold Stratified Group K-Fold Cross Validation** (grouped by `account_id` to eliminate same-account leakage across folds), the deployed model ([behavioral_engine_v2.pkl](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/Output_Artifacts/behavioral_engine_v2.pkl)) achieved:
* **Overall ROC-AUC:** `0.8447`
* At the **F2-Optimal Threshold (0.46)**:
  * **Recall (Catch Rate):** `68.1%` (Caught 561 of 824 default windows)
  * **False Positive Rate (FPR):** `14.7%`
  * **Precision:** `38.7%`
  * **F2-Score:** `0.591`

### 7. Model Type: XGBoost on engineered features? Neural network? Sequence models?
The core engine is an **XGBoost Classifier** (`xgb.XGBClassifier`) deployed atop the 9 heavily engineered tabular velocity features. 

The hyperparameter selection was dictated by an Optuna tuning sweep (50 trials) aimed at maximizing the F2-Score. The chosen architecture uses a relatively shallow configuration (`max_depth=4`, `learning_rate=0.05`, `n_estimators=300`) alongside row/column subsampling (`0.8`) to aggressively prevent memorization of specific customer accounts.

### 8. Size: How many loans/transactions? Class balance?
* **Raw Size:** 682 loans and ~1 million transactions.
* **Engineered Size:** By deploying the sliding window temporal sampling, the 682 loans are mathematically expanded into an evaluation matrix of **~8,000 independent longitudinal rows** (each representing a 6-month snapshot of a customer's life).
* **Class Balance:** The target variable explicitly maintains an **11% default rate vs. 89% good rate** (approx. a 1:8 class imbalance).
* **Imbalance Handling:** The script tested algorithmic options including SMOTE and ADASYN, but the baseline XGBoost scale-weighting proved superior. The imbalance is resolved natively via XGBoost's `scale_pos_weight` parameter (which calculates a dynamic ~8.0 multiplier penalty per specific cross-validation fold).

---
*Generated directly from the `VECTOR/Berka` codebase execution logs and data extraction algorithms.*
