# Financial Historian Model — End-to-End Pipeline

## Persona: Meet "Raj"
A real borrower. Here are his **raw transaction columns** exactly as they would arrive from Lending Club.

---

## STEP 0 — Raw Data Ingestion (What the DB Receives)

These are the _unprocessed_ columns from the Lending Club dataset:

| Column | Raj's Value | Description |
|---|---|---|
| `loan_amnt` | 18000 | Loan applied for |
| `funded_amnt` | 18000 | Actual funded amount |
| `term` | `" 36 months"` | Loan tenure (string) |
| `int_rate` | `"17.49%"` | Interest rate (string with %) |
| `installment` | 644.06 | Monthly payment due |
| `grade` | `"D"` | LC-assigned credit grade |
| `sub_grade` | `"D2"` | Granular sub-grade |
| `emp_length` | `"3 years"` | Employment duration |
| `home_ownership` | `"RENT"` | Housing status |
| `annual_inc` | 52000 | Self-reported annual income |
| `verification_status` | `"Verified"` | Income verification status |
| `purpose` | `"debt_consolidation"` | Reason for loan |
| `addr_state` | `"TX"` | State of residence |
| `dti` | 24.7 | Debt-to-income ratio |
| `earliest_cr_line` | `"Jul-2011"` | When credit history started |
| `issue_d` | `"Mar-2023"` | Loan issued date |
| `fico_range_low` | 665 | FICO at origination (low) |
| `fico_range_high` | 669 | FICO at origination (high) |
| `last_fico_range_low` | 610 | Latest reported FICO (low) |
| `last_fico_range_high` | 614 | Latest reported FICO (high) |
| `delinq_2yrs` | 1 | Delinquencies in past 2 years |
| `mths_since_last_delinq` | 14 | Months since last delinquency |
| `inq_last_6mths` | 3 | Hard inquiries in past 6 months |
| `open_acc` | 7 | Open credit accounts |
| `pub_rec` | 0 | Public derogatory records |
| `revol_bal` | 11400 | Revolving balance owed |
| `revol_util` | `"74.2%"` | Revolving credit utilisation (string) |
| `total_acc` | 14 | Total credit accounts ever |
| `out_prncp` | 14200 | Outstanding principal remaining |
| `total_rec_prncp` | 3800 | Principal paid so far |
| `total_rec_int` | 1960 | Interest paid so far |
| `total_rec_late_fee` | 22.50 | Total late fees paid |
| `recoveries` | 0 | Post-charge-off recoveries |
| `last_pymnt_amnt` | 644.06 | Last payment amount |
| `loan_status` | `"Late (31-120 days)"` | **Current loan status** |

---

## STEP 1 — Data Cleaning & Type Parsing

Raw data has dirty types. We fix them:

```python
# String → Float
df['int_rate']   = df['int_rate'].str.replace('%','').astype(float)   # 17.49
df['revol_util'] = df['revol_util'].str.replace('%','').astype(float) # 74.2

# String → Numeric months
df['term_months'] = df['term'].str.strip().str.extract(r'(\d+)').astype(float)  # 36.0

# String dates → datetime
df['issue_d']          = pd.to_datetime(df['issue_d'], format='%b-%Y')          # 2023-03-01
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y') # 2011-07-01
```

**Raj after parsing:**
- `int_rate` = 17.49, `revol_util` = 74.2, `term_months` = 36

---

## STEP 2 — Feature Engineering (Historian Signals)

This is where raw columns become **meaningful behavioral dimensions**:

### 2a. Credit Deterioration Features
```python
df['fico_avg']      = (fico_range_low + fico_range_high) / 2     # → 667
df['last_fico_avg'] = (last_fico_range_low + last_fico_range_high) / 2  # → 612
df['fico_drop']     = fico_avg - last_fico_avg                    # → 55 ← KEY SIGNAL
```
> Raj's FICO dropped **55 points** since loan origination. This is a severe deterioration event.

### 2b. Credit History Depth
```python
df['cr_history_months'] = (issue_d - earliest_cr_line).dt.days / 30  # → ~140 months
```
> Raj has 140 months (~12 years) of credit history. A 55-pt drop in a mature profile is alarming.

### 2c. Debt Burden Ratios
```python
df['loan_to_income']  = loan_amnt / annual_inc                    # → 0.346
df['inst_to_income']  = installment / (annual_inc / 12)           # → 0.149  (14.9% of monthly income)
```
> Raj is committing 14.9% of monthly income just to this one installment — high pressure.

### 2d. Repayment Trajectory
```python
df['repay_ratio']     = total_rec_prncp / loan_amnt               # → 3800/18000 = 0.211
df['out_prncp_ratio'] = out_prncp / loan_amnt                     # → 14200/18000 = 0.789
```
> After several months, he's only paid back 21.1%. 78.9% is still outstanding — slow repayment.

### 2e. Delinquency Timeline Score
```python
df['delinq_recency_score'] = 1 / (mths_since_last_delinq + 1)    # → 1/15 = 0.067
```
> A delinquency just 14 months ago. Moderate recency — this isn't ancient history.

### 2f. Hard Inquiry Pressure
```python
df['inq_pressure'] = inq_last_6mths / (total_acc + 1)            # → 3/15 = 0.200
```
> 3 hard pulls against only 14 accounts. He's been desperately seeking new credit.

### 2g. Late Fee Flag
```python
df['late_fee_flag'] = int(total_rec_late_fee > 0)                 # → 1  ← TRIGGER
```
> He has already incurred a late fee. This is a **confirmed event-level trigger**.

### 2h. Target Label
```python
df['risk_label'] = 1  # loan_status = "Late (31-120 days)" → Delinquent
```

---

## STEP 3 — The Three Model Layers

Now the system processes Raj's feature vector through all three layers simultaneously.

---

### LAYER 1: Cohort Matching

The historian searches for past borrowers who **looked like Raj at origination**:

```
Search Criteria (Exact):
  grade_enc       = D
  term_months     = 36
  fico_avg        ∈ [650, 680]
  dti             ∈ [20, 30]
  purpose         = debt_consolidation
  cr_history_months ∈ [100, 160]
```

**Cohort found**: 4,820 historical borrowers

Now the historian asks: **"Of these 4,820 similar borrowers, when `late_fee_flag = 1` appeared, what happened?"**

```
late_fee_flag triggered mid-loan → Defaulted within 6 months: 71%
late_fee_flag triggered mid-loan → Resolved safely: 29%
```

**Layer 1 Output**: `COHORT_RISK_SCORE = 0.71` → ⚠️ High

---

### LAYER 2: Absolute Feature Thresholds

Independent of any cohort, certain feature values are **universally dangerous**:

| Feature | Raj's Value | Danger Threshold | Triggered? |
|---|---|---|---|
| `fico_drop` | 55 | > 40 | ✅ YES |
| `late_fee_flag` | 1 | > 0 | ✅ YES |
| `revol_util` | 74.2% | > 70% | ✅ YES |
| `inq_pressure` | 0.20 | > 0.15 | ✅ YES |
| `dti` | 24.7 | > 22 | ✅ YES |
| `repay_ratio` | 0.211 | < 0.30 | ✅ YES |

**6 out of 6 thresholds breached.**

**Layer 2 Output**: `THRESHOLD_RISK_SCORE = 6/6 = 1.0` → 🚨 Critical

---

### LAYER 3: Anomaly Detection

The system checks if this behavioral pattern has ever been seen before:

```python
# Raj's vector fingerprint
[fico_drop=55, late_fee_flag=1, inq_pressure=0.20,
 revol_util=74.2, repay_ratio=0.21, delinq_recency_score=0.067]

# Distance to nearest cluster centroid in the historical embedding space
cosine_distance = 0.18  # Low distance → Pattern IS known historically
```

Since the distance is low, this is **not a novel anomaly** — it's a well-mapped default trajectory. Anomaly layer stays quiet.

**Layer 3 Output**: `ANOMALY_FLAG = 0` (Pattern is historically familiar — and historically dangerous)

---

## STEP 4 — Final Risk Decision

All three layers feed into the final composite score:

```
COMPOSITE_RISK = 0.4 × COHORT_RISK + 0.4 × THRESHOLD_RISK + 0.2 × ANOMALY_FLAG

             = 0.4 × 0.71 + 0.4 × 1.0 + 0.2 × 0.0
             = 0.284 + 0.40 + 0.0
             = 0.684
```

> **Decision: ALERT — Pre-Delinquency Signal Confirmed**
> Raj's behavioral vector is converging on the historical default trajectory of his peer cohort.
> Recommended action: Trigger early outreach / restructuring offer.

---

## Summary Flow

```
Raw Transaction Columns
        ↓
  Type Parsing & Cleaning
        ↓
  Feature Engineering (9 signals)
        ↓
  ┌─────────────────────────────────┐
  │  Layer 1: Cohort Matching       │ → 71% default rate in peer group
  │  Layer 2: Threshold Triggers    │ → 6/6 thresholds breached
  │  Layer 3: Anomaly Detection     │ → Known pattern (no novel flag)
  └─────────────────────────────────┘
        ↓
  Composite Risk Score: 0.684
        ↓
  🚨 PRE-DELINQUENCY ALERT FIRED
```
