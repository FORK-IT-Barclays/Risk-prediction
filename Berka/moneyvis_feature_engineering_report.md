# MoneyVis Feature Engineering: The Inference Pipeline

The MoneyVis dataset represents a single UK customer's financial history from 2015 to 2022. Because the VECTOR risk model was trained on 1990s Czech Republic data (the Berka dataset), the MoneyVis inference pipeline must perfectly mirror the training feature extraction while safely crossing the economic and currency boundaries.

Here is the step-by-step breakdown of how the raw MoneyVis transactions are transformed into XGBoost-ready risk features.

---

## 1. Raw Data Standardisation

The raw [MoneyVis.csv](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/dataset/MoneyVis.csv) contains standard UK bank statement columns:
`Transaction Date`, `Transaction Type`, `Transaction Description`, `Debit Amount`, `Credit Amount`, `Balance`

**Parsing & Cleaning:**
1. **Dates** are parsed into standard `datetime` objects.
2. **Cash In / Cash Out** is extracted by isolating `Credit Amount` and `Debit Amount` respectively, filling nulls with `0.0`.
3. **Semantic Tagging:** The pipeline uses a simple keyword matcher to map British transaction descriptions into the universal tags expected by the model. 
   * `BGC` (Bank Giro Credit) or keywords like `"UNIV OF"` → Tagged as `SALARY`
   * `DD` (Direct Debit) or utilities like `"VIRGIN", "O2", "OCTOPUS", "CITY COUNC"` → Tagged as `BILL`
   * Everything else → `UNCATEGORIZED`

This transforms the proprietary UK statement into the exact **Universal Ledger Schema** used during Berka training.

---

## 2. Generating the Customer Trajectory (Sliding Windows)

In training, we looked back from the theoretical end-of-loan date. For real-time inference on a live UK customer, we want to see **how their risk changes over time**. 

Instead of generating one single risk score for 2022, the pipeline generates **50 temporal snapshots**:
1. It starts at the very last transaction in the dataset (July 2022).
2. It looks back **180 days** to form a window.
3. It steps backward by exactly **7 days** (a 1-week stride) and forms another 180-day window.
4. It repeats this 50 times.

This effectively draws a "risk trajectory" across the past 50 weeks of the customer's life. 

---

## 3. The T2 vs T1 Split

For *each* of the 50 windows, the 180-day period is explicitly split in half:
* **T2 (Recent):** The most recent 90 days
* **T1 (Prior):** The 90 days immediately preceding T2

If either T1 or T2 has fewer than 5 transactions, the pipeline ignores the window entirely. The customer must have sufficient activity to meaningfully measure trajectory.

---

## 4. Extracting the "Physics of Risk" Signals

Within each window, the same 9 behavioral features are calculated.

### A. The 6 Dimensionless Velocity Signals (Currency-Agnostic)
Because these are relative rates of change, they require **zero currency conversion**. A 25% drop in GBP works mathematically identically to a 25% drop in CZK. 

1. `income_erosion_v`: [(T2 Inflows - T1 Inflows) / |T1 Inflows|](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/1_Feature_Engineering/feature_engineering.py#81-92)
2. `liquidity_momentum_v`: [(T2 Avg Balance - T1 Avg Balance) / |T1 Avg Balance|](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/1_Feature_Engineering/feature_engineering.py#81-92)
3. `tx_freq_v`: [(T2 Txn Count - T1 Txn Count) / |T1 Txn Count|](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/1_Feature_Engineering/feature_engineering.py#81-92)
4. `salary_drift_v`: The shift in the *day of the month* the salary arrives.
5. `overdraft_v`: The raw increase/decrease in the number of days spent below zero balance.
6. `overdraft_t2`: The absolute number of days spent below zero in T2.

### B. The 3 Absolute Scale Anchors (Currency-Dependent)
These features anchor the model to the absolute size of the customer's wallet. Because 1 GBP is vastly different from 1 CZK, these must be mathematically scaled so the XGBoost tree splits trigger correctly. 

7. `avg_balance_t2`
8. `min_balance_t2`
9. `total_out_t2`

---

## 5. Macroeconomic Purchasing Power Parity (PPP) Scaling

Instead of statistically manipulating the MoneyVis customer (e.g. Z-score scaling, which destroys their actual, relative wealth), the pipeline applies a **constant macroeconomic conversion factor**. 

The code defines:
```python
ECONOMIC_PPP_SCALER = 35.0
```

**The Rationale:** 
If the XGBoost model split a decision tree at `total_out_t2 < 40,000`, that 40,000 meant **1998 Czech Korunas**. Feeding an unscaled UK spend of `2,000 GBP` into the model would falsely trigger the low-spend logic, categorizing a normal British citizen as being in dire poverty.

To fix this, we look at economic reality:
* The raw exchange rate in 1998 was roughly 1 GBP = 55 CZK.
* Accounting for two decades of heavy UK inflation between 1998 and 2021 (the MoneyVis era), 1 modern GBP buys less.
* The effective Purchasing Power Parity (PPP) bridging 2021 British Sterling to 1998 Czech Korunas is roughly **35.0**.

By applying `35.0` as a flat multiplier to `avg_balance`, `min_balance`, and `total_out`, the inference pipeline translates the UK customer's exact economic reality directly into the mathematical scale that the XGBoost risk engine understands natively.
