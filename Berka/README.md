# PKDD99 Czech Republic Financial Dataset (Berka) Analysis

This repository contains an implementation for loading and analyzing the Berka dataset, a rich financial database originally released for the PKDD'99 Discovery Challenge. It is an excellent dataset for building **behavioral machine learning models for predicting financial default risk**.

## Dataset Structure
The dataset is formatted as relational CSV files using semicolons (`;`) as delimiters. It consists of the following 8 core tables:
* **account.csv** (4,500 records): Static information about the accounts (e.g., creation date, statement frequency).
* **client.csv** (5,369 records): Demographic characteristics of the bank's clients (e.g., age, gender).
* **disp.csv** (5,369 records): The disposition/authorization table that maps `client_id`s to `account_id`s. Identifies whether a client is the primary `OWNER` or a secondary `DISPONENT`.
* **district.csv** (77 records): Macro-economic and demographic data for regions.
* **loan.csv** (682 records): Details of granted loans including amount, duration, payments, and standard default `status`.
* **order.csv** (6,471 records): Standing orders (e.g., standard monthly payments) attached to accounts.
* **card.csv** (892 records): Credit cards issued to clients.
* **trans.csv** (1,056,320 records): Every incoming and outgoing transaction for every account.

## Key Findings & Relationships
Through exploratory data analysis, we verified the following structural rules of the dataset:
1. **One-to-One Loan Mapping**: Each account has a maximum of exactly 1 loan. Out of 4,500 total accounts, only 682 have a loan.
2. **One Primary Owner**: Every single account is linked to exactly 1 primary `OWNER`. Secondary clients (`DISPONENT`) may exist, but the primary risk responsibility falls on the single owner. 
3. **Imbalanced Targets**: Of the 682 loans, roughly 11.1% (76 loans) are classified as problematic or in default (Status `B` or `D`). The remaining ~89% are completed successfully or currently in good standing.

## Why is this Dataset Good for Behavioral Credit Risk Modeling?

At first glance, a dataset with only 682 historical loans might seem insufficient when compared to modern big-data deep learning tasks. However, it is an exceptionally strong candidate for predictive default modeling for the following reasons:

### 1. Depth over Breadth (Time-Series Density)
While the number of individual loan labels (N=682) is relatively small, the behavioral history available for those specific customers is vast. 
* Total dataset transactions: ~1.05 Million
* Transactions belonging strictly to the 682 Loan Accounts: **191,556**

This means there is an average of **~280 temporal transaction records per borrower**. A behavioral model leverages this depth. Instead of relying on static metrics (like age or account creation date), the predictive power will come from engineered features such as trailing 3-month average balances, overdraft frequencies, and variance in monthly cash flow. 

### 2. High Suitability for Tabular Machine Learning
Financial risk models rarely rely on deep learning architectures requiring millions of rows. Specialized tabular algorithms like **XGBoost, LightGBM, Random Forests, and Penalized Logistic Regression** thrive on datasets of this size. With strong feature engineering, 682 distinct labels are more than enough for a tree-based ensemble to capture decision boundaries without aggressively overfitting. 

### 3. Realistic Class Imbalance
An 11% default rate perfectly mirrors the highly imbalanced nature of real-world credit risk. Most people pay their loans back. This dataset offers an excellent playground to demonstrate critical data science competencies handling class imbalance, such as applying SMOTE, altering decision thresholds, or tuning class weights.
