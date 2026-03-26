# VECTOR Behavioral Engine: Feature Definitions & Predictive Rationale

This document details the 9 engineered features that the VECTOR XGBoost model uses to predict loan default. These features are designed to capture the **"Physics of Risk"** — measuring the acceleration, trajectory, and momentum of a customer's financial distress, rather than just static snapshots.

Each feature is computed over a sliding 6-month window, divided into two periods:
* **T1 (Prior):** The first 90 days of the window
* **T2 (Recent):** The last 90 days of the window

---

## 1. `income_erosion_v`
*"Is the money coming in shrinking?"*

### Formula
[(total_cash_in_T2 − total_cash_in_T1) / |total_cash_in_T1|](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/1_Feature_Engineering/feature_engineering.py#81-92)  *(Clipped to [-5, +5])*

### Predictive Rationale
This is the **primary distress signal**. It measures whether the customer's total inflows (salary, transfers, credit) are eroding over the 6-month window. A value of `-0.3` means income dropped 30% quarter-on-quarter. 

A person who defaults almost never does so suddenly. The leading indicator is almost always income deterioration — a lost job, reduced hours, or side income drying up. The model learns that sustained negative `income_erosion_v` across multiple windows is a strong precursor to an eventual inability to service a loan.

---

## 2. `liquidity_momentum_v`
*"Is the account draining?"*

### Formula
[(avg_balance_T2 − avg_balance_T1) / |avg_balance_T1|](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/1_Feature_Engineering/feature_engineering.py#81-92) *(Clipped to [-5, +5])*

### Predictive Rationale
This captures the **trajectory of the financial cushion** — not what the balance *is*, but whether it is *rising or falling*. A deeply negative value means the account is bleeding out even if it still contains money.

Balance level alone is misleading (a high earner can have a low balance, and vice-versa). What matters is the direction. An account where balances are consistently falling — even modestly — belongs to a customer who is spending more than they earn. This is the runway to default.

---

## 3. `overdraft_v`
*"Are overdraft events getting worse?"*

### Formula
[(count_of_negative_balance_days_T2) − (count_of_negative_balance_days_T1)](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/1_Feature_Engineering/feature_engineering.py#81-92) *(Raw delta, not a ratio)*

### Predictive Rationale
This acts as an **accelerometer for financial stress**. It doesn't just measure whether overdrafts exist; it measures whether they are *accelerating*. A value of `+5` means the customer went into overdraft 5 more times this quarter than last.

Overdrafts are expensive (fees, punitive interest). A customer who is increasingly slipping into negative balances isn't just stressed — they are actively losing money to their bank while attempting to stay afloat. The worsening trend often signals an approaching breaking point.

---

## 4. `overdraft_t2`
*"How bad is the distress right now, in absolute terms?"*

### Formula
`count_of_negative_balance_days_T2`

### Predictive Rationale
This measures the **absolute severity of current distress**, complementing `overdraft_v`. While `overdraft_v` indicates the direction of travel, `overdraft_t2` tells you how deep in the hole the customer actually is during the most recent period.

A customer who recently spent 40 out of 90 days in overdraft is a materially different risk from one who spent 2 out of 90 days in overdraft, even if both showed the same *improvement* (delta) in `overdraft_v`. Both absolute level and velocity are necessary for an accurate risk assessment.

---

## 5. `salary_drift_v`
*"Is payday creeping later each month?"*

### Formula
`median_salary_day_of_month_T2 − median_salary_day_of_month_T1`

### Predictive Rationale
This is the **subtlest signal** in the set. It tracks whether the employer is paying later, whether payment frequency is changing, or whether the primary income source is switching. A value of `+8` means salary is arriving roughly 8 days later in the recent quarter than it used to.

This serves as a proxy for employment instability that doesn't immediately show up in the total income volume. A person who switches from formal salaried employment (paid on a fixed date each month) to freelance or cash-in-hand income will show a volatile, drifting salary trajectory long before their total income completely drops. It captures the *fragility* of the income source.

---

## 6. `tx_freq_v`
*"Is the customer going quiet?"*

### Formula
[(transaction_count_T2 − transaction_count_T1) / |transaction_count_T1|](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/1_Feature_Engineering/feature_engineering.py#81-92) *(Clipped to [-5, +5])*

### Predictive Rationale
This measures whether the **volume of financial activity** is changing. A negative value means the customer is making fewer purchases, withdrawals, and transfers than they used to.

Counterintuitively, *reduced* transaction activity is often a potent warning sign. It can indicate a frozen credit card, a switch to cash-only survival mode, or simply that the customer has nothing left to spend. It is the digital equivalent of "going quiet" before a financial crisis. 

---

## 7. `avg_balance_t2`
*"What is the absolute level of the account's buffer?"*

### Formula
Mean average daily balance over T2 (the most recent 90 days).

### Predictive Rationale
This is a **scale anchor**. It tells the model the absolute financial size of the dataset. The velocity signals above are dimensionless ratios; without this anchor, the model would not know if a "30% balance drop" is a drop from 10,000 to 7,000, or a drop from 500 to 350.

An exceptionally low average balance is an independently strong predictor of default because there is less margin for error. A customer with a 500 balance who loses 20% of their income will default far sooner than one with a 50,000 balance who faces the identical percentage shock. The XGBoost tree splits learn these critical absolute thresholds automatically.

---

## 8. `min_balance_t2`
*"What is the worst-case financial moment the customer hit recently?"*

### Formula
The minimum (lowest) balance recorded at any point during T2.

### Predictive Rationale
This is the **floor of financial distress** — the hardest moment the customer experienced in the recent window. It reveals acute crises that the smoothed-out `avg_balance_t2` would conceal.

A customer might maintain an acceptable average balance of 5,000, but if their minimum balance spiked down to -2,000, they experienced at least one severe crisis event. Default risk is heavily driven by these worst-case tail events; it only takes one catastrophic shortfall to trigger a missed payment and initiate a default spiral.

---

## 9. `total_out_t2`
*"How heavy is the financial load being carried?"*

### Formula
Sum total of all cash outflows (spending, bills, transfers) in T2.

### Predictive Rationale
This measures the **scale of financial obligation**. It indicates how much money the customer *requires* to maintain their lifestyle and service their existing commitments.

Default risk is fundamentally a function of the gap between total income and total obligations. A customer with `total_out_t2 = 80,000` and `income = 82,000` is living much closer to the edge than an account with `total_out = 20,000` and `income = 40,000`. When combined with `income_erosion_v`, the model can detect precisely when this buffer gap is critically narrowing toward zero — the mathematical prerequisite for default.

---

### Currency Agnostic vs. Currency Dependent Features
It is important to note that features 1 through 6 are **dimensionless velocity ratios** (percentages or counts) and are entirely **currency-agnostic**. They transfer directly between the Czech Koruna (Berka dataset) and British Sterling (MoneyVis) without modification.

Features 7, 8, and 9 (`avg_balance`, `min_balance`, `total_out`) represent **absolute currency values**. When applying the model to foreign datasets, these three columns require a Purchasing Power Parity (PPP) scaler multiplier to align them with the original training currency scale.
