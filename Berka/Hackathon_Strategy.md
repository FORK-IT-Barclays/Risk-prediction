# Hackathon Strategy: Predictive Behavioral Modeling

When competing in a hackathon with a relatively small number of labeled target rows (N=682 loans), standard models like a basic Random Forest might struggle to impress judges. To address the 682-loan limitation while leveraging the massive amount of unlabeled transaction data, we can deploy the following advanced techniques.

## Innovative Solutions for Small-N Datasets

### 1. Time-Series Data Augmentation (Window Slicing)
Instead of aggregating a customer's entire 3-year history into a single static row, we can use sliding temporal windows:
* Divide the 3-year transaction history into 12-month rolling windows (e.g., Months 1-12, Months 2-13, Months 3-14).
* A single loan account is now represented as multiple independent training rows capturing evolving behavior over time.
* **Benefit:** This artificially expands our training set from 682 rows to potentially 5,000+ robust temporal samples.

### 2. Semi-Supervised Pretraining (Sequence Autoencoders)
We have 3,818 accounts that *do not* have a loan, but possess over 850,000 transactions. 
* **Execution:** Train an LSTM or Transformer to predict the next transaction in a sequence for *all* 4,500 accounts. 
* **Benefit:** The model learns a deep "financial embedding" (understanding normal vs strange spending patterns) from the entire dataset. We then freeze this encoder and fine-tune only the final classification layer on our 682 labeled loans to predict the default `status`.

### 3. Synthetic Tabular Generation (CTGAN)
* **Execution:** Use a Conditional Tabular Generative Adversarial Network (CTGAN).
* **Benefit:** Generate hundreds or thousands of highly realistic "synthetic" borrowers that mimic the statistical correlations of our actual defaults. This elegantly solves both the small sample size and the class imbalance (11% default rate).

---

## Data Scientist EDA Checklist: Preparing for Classification

Before we apply any sophisticated models, we must perform a rigorous Exploratory Data Analysis (EDA) specifically tailored for a classification pipeline on behavioral data. 

**Here is the step-by-step checklist to ensure our data is ready:**

### 1. Target Variable Definition
- [ ] Group the 4 loan statuses (`A`, `B`, `C`, `D`) into a binary target: **Good Loan** (`0`) vs **Bad/Default Loan** (`1`).
- [ ] Measure the exact class imbalance (e.g., 89% Good vs 11% Bad).

### 2. Temporal & Date Engineering
- [ ] Convert all YYMMDD date integer columns (e.g., `930101`) to proper Pandas `datetime` objects across all tables.
- [ ] Calculate the crucial "Time to Default" or "Account Age at Loan Issuance" by comparing account creation dates to loan creation dates. We must ensure we do not use transactions that occurred *after* the loan defaulted (Data Leakage!).

### 3. Behavioral Feature Extraction (Aggregation)
Create summary statistics per `account_id` based *only* on the 12 to 24 months strictly prior to the loan date:
- [ ] **Volume:** Total number of transactions, avg transactions per month.
- [ ] **Velocity:** Frequency of low balance events (balance dropping below a threshold).
- [ ] **Cash Flow Ratio:** Ratio of total `PRIJEM` (income) vs `VYDAJ` (expenses).
- [ ] **Volatility:** Standard deviation of the monthly account balance.

### 4. Categorical & Demographic Mapping
- [ ] Map the Czech terms in `trans` and `order` tables to English categories for clarity (e.g., `VYBER` = Withdrawal, `DUCHOD` = Old-age Pension).
- [ ] Extract strict demographic features from `client.csv`: Derive the `Age` and `Gender` directly from the `birth_number` format. 
- [ ] Integrate macro-economic indicators (unemployment rate, average salary) from the `district.csv` table via the account's district ID.

### 5. Outlier & Missing Value Handling
- [ ] Check for missing values (`NaN` or spaces) resulting from joins. 
- [ ] Detect extreme outliers in transaction `amount` or `balance` that might aggressively skew the gradient boosting trees.

### 6. Correlation & Collinearity Strategy
- [ ] Generate a correlation matrix (Pearson/Spearman) of all generated features against the `Target` to measure predictive power.
- [ ] Remove highly correlated engineered features (e.g., Total Income vs Average Monthly Income) to prevent multi-collinearity issues.

Completing this checklist will yield a powerful, flat analytical matrix `(X, y)` that is perfectly primed for our classification models!
