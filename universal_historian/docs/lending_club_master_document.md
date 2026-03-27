# Lending Club — Financial Historian Model
## Comprehensive Research & Engineering Document

> **Context**: Barclays Hack-o-Hire | Pre-Delinquency Early-Warning Vector System  
> **Dataset**: Lending Club Accepted Loans 2007–2018 Q4 | 2.26M records, 150+ features  
> **Preprocessed Sample**: 500,000 rows × 31 engineered dimensions

---

## Part 1 — What is the Financial Historian Model?

The **Financial Historian Model** is an analytical paradigm that shifts focus from static, point-in-time credit snapshots to a **longitudinal, event-driven representation** of a borrower's credit health and repayment behavior.

Instead of asking *"Is this customer risky?"*, it asks:

> *"Does this customer's current behavioral trajectory resemble the historical trajectory of customers who defaulted?"*

### Core Principle

The model is a **pattern-matching engine over financial behavior sequences**. Every incoming signal (a late fee, a FICO drop, a hard inquiry) is checked against the historical memory of millions of past borrowers to determine whether the current trajectory converges toward a known default path.

---

## Part 2 — How Transaction-Level Data Maps to the Historian

The Lending Club dataset natively supports the historian paradigm through four chronological pillars:

### A. The Deterioration Trace (Macro Credit Health)
Tracks the timeline of a borrower's fundamental creditworthiness — not just where they are, but where they've been.

| Feature | Role |
|---|---|
| `fico_avg` | Credit health at origination (baseline) |
| `last_fico_avg` | Credit health today |
| `fico_drop` | **Δ deterioration** — the most critical historian signal |
| `cr_history_months` | Context: how long this borrower's history spans |

### B. The Delinquency Timeline (Event Recency)
Past behavior predicts future delinquency, but *when* those events occurred defines their weight.

| Feature | Role |
|---|---|
| `delinq_2yrs` | Frequency of historical failures |
| `pub_rec` | Systemic events (bankruptcies) anchoring risk baseline |
| `delinq_recency_score` | `1 / (mths_since_last_delinq + 1)` — decays older events, amplifies recent ones |

### C. Credit-Seeking Pressure (Immediate Stress Signals)
A pre-origination window showing whether the borrower is in a sudden cash crunch.

| Feature | Role |
|---|---|
| `inq_last_6mths` | Raw recent hard pulls |
| `inq_pressure` | `inq_last_6mths / (total_acc + 1)` — stress normalized by credit portfolio size |

### D. Ongoing Repayment Trajectory (Micro Behavior)
Month-to-month post-origination behavior — the real-time heartbeat of the loan.

| Feature | Role |
|---|---|
| `repay_ratio` | `total_rec_prncp / loan_amnt` — principal paydown velocity |
| `late_fee_flag` | Binary trigger: any late fee = confirmed distress event |
| `out_prncp_ratio` | How much of the original debt still remains |

---

## Part 3 — Cohort Matching: Why Your History is Not Compared to Everyone

Every borrower is unique in scale and context. The historian solves this by **clustering each borrower into a peer group** — people who looked like them at origination — before comparing trajectories.

### Cohort Construction Criteria

```
Grade D | Loan: $18K | DTI: 22–26 | FICO: 650–680 | Purpose: debt_consolidation
           ↓
  4,820 historically matched borrowers found

Of those — when late_fee_flag = 1 triggered:
  → 71% defaulted within 6 months
  → 29% resolved safely
```

The historian doesn't compare you to *all* borrowers — it finds exactly who you are comparable to and then checks whether your behavior is following their path.

### Cohort Relaxation (When an Exact Match Doesn't Exist)

If no exact cohort is found, the system widens the search radius iteratively:

```
Exact match → Not found
  ↓ Relax grade/FICO range
  ↓ Relax to DTI range + Purpose only
  ↓ Use broader cohort as signal baseline
```

### Absolute Feature Thresholds (Cohort-Independent Signals)

Even without a peer group, individual features carry **universal danger levels**:

| Feature | Danger Threshold | Interpretation |
|---|---|---|
| `fico_drop` | > 40 points | Severe credit erosion |
| `late_fee_flag` | > 0 | Confirmed distress event |
| `revol_util` | > 70% | Near-maxed revolving credit |
| `inq_pressure` | > 0.15 | Desperate credit seeking |
| `repay_ratio` | < 0.30 | Barely paying down principal |

### Anomaly Detection (Novel Patterns)

If a borrower's behavioral signature has **no historical precedent**, the model escalates rather than staying silent:

> *"This pattern has no historical match → Escalate for manual review"*

A scenario with no cohort match is itself an anomaly — and anomalies in credit behavior are almost always a risk signal, not a green light.

---

## Part 4 — End-to-End Pipeline: Worked Example

**Persona**: Raj | $18K loan | Grade D | 36 months | Purpose: debt consolidation

### Step 0 — Raw Data (Exact Columns)

| Column | Value |
|---|---|
| `loan_amnt` | 18000 |
| `term` | `" 36 months"` |
| `int_rate` | `"17.49%"` |
| `installment` | 644.06 |
| `grade` / `sub_grade` | D / D2 |
| `annual_inc` | 52000 |
| `dti` | 24.7 |
| `fico_range_low/high` | 665 / 669 |
| `last_fico_range_low/high` | 610 / 614 |
| `delinq_2yrs` | 1 |
| `mths_since_last_delinq` | 14 |
| `inq_last_6mths` | 3 |
| `revol_util` | `"74.2%"` |
| `total_rec_late_fee` | 22.50 |
| `loan_status` | `"Late (31-120 days)"` |

### Step 1 — Type Parsing

```python
df['int_rate']    = "17.49%"  → 17.49
df['revol_util']  = "74.2%"   → 74.2
df['term_months'] = " 36 months" → 36.0
df['issue_d']     = "Mar-2023" → 2023-03-01
```

### Step 2 — Feature Engineering (9 Historian Signals)

| Engineered Feature | Formula | Raj's Value | Signal |
|---|---|---|---|
| `fico_avg` | (665+669)/2 | **667** | Origination baseline |
| `last_fico_avg` | (610+614)/2 | **612** | Current credit |
| `fico_drop` | 667 − 612 | **55** ⚠️ | Severe deterioration |
| `cr_history_months` | (issue_d − earliest_cr_line) / 30 | **140** | Mature profile → drop is alarming |
| `loan_to_income` | 18000 / 52000 | **0.346** | High debt load |
| `inst_to_income` | 644 / (52000/12) | **0.149** | 14.9% of monthly income |
| `repay_ratio` | 3800 / 18000 | **0.211** ⚠️ | Barely paying down |
| `delinq_recency_score` | 1 / (14+1) | **0.067** | Recent past failure |
| `inq_pressure` | 3 / (14+1) | **0.200** ⚠️ | Desperate credit seeking |
| `late_fee_flag` | (22.50 > 0) | **1** 🚨 | Confirmed trigger |

### Step 3 — Three Model Layers

**Layer 1: Cohort Matching**
- 4,820 historical borrowers matched to Raj's profile
- Of those who also triggered `late_fee_flag = 1` mid-loan → **71% defaulted**
- `COHORT_RISK_SCORE = 0.71`

**Layer 2: Absolute Thresholds**
- 6 out of 6 danger thresholds breached
- `THRESHOLD_RISK_SCORE = 1.0`

**Layer 3: Anomaly Detection**
- Cosine distance to nearest cluster = 0.18 → pattern is historically known (and dangerous)
- `ANOMALY_FLAG = 0`

### Step 4 — Final Decision

```
COMPOSITE_RISK = 0.4 × 0.71  +  0.4 × 1.0  +  0.2 × 0.0  =  0.684

→ 🚨 PRE-DELINQUENCY ALERT — Raj's trajectory converges on known default path
```

---

## Part 5 — EDA Analysis Results

### 1 — Class Imbalance
![Class Imbalance](01_class_imbalance.png)

### 2 — Null Value Analysis
![Null Analysis](02_null_analysis.png)

### 3 — Feature Distributions by Risk Label
````carousel
![Feature Distributions — Page 1](03_feature_distributions_page01.png)
<!-- slide -->
![Feature Distributions — Page 2](03_feature_distributions_page02.png)
<!-- slide -->
![Feature Distributions — Page 3](03_feature_distributions_page03.png)
<!-- slide -->
![Feature Distributions — Page 4](03_feature_distributions_page04.png)
<!-- slide -->
![Feature Distributions — Page 5](03_feature_distributions_page05.png)
<!-- slide -->
![Feature Distributions — Page 6](03_feature_distributions_page06.png)
<!-- slide -->
![Feature Distributions — Page 7](03_feature_distributions_page07.png)
````

### 4 — Loan Status Distribution
![Loan Status](04_loan_status_distribution.png)

### 5 — Full Correlation Matrix
![Correlation Matrix](05_correlation_matrix.png)

### 6 — Feature Correlations with risk_label
![Risk Label Correlations](06_risk_label_correlations.png)

### 7 — Box Plots by Risk Label
![Box Plots](07_boxplots_by_risk.png)

### 8 — Outlier Summary
![Outlier Summary](08_outlier_summary.png)

### 9 — Geographic Delinquency by State
![Geographic Delinquency](09_geographic_delinquency.png)

### 10 — Temporal Trend
![Temporal Trend](10_temporal_trend.png)

---

## Part 6 — Why Lending Club Beats Other Datasets

| Criterion | German Credit | UCI Taiwan | Home Credit | **Lending Club** |
|---|---|---|---|---|
| Records | 1,000 | 30,000 | ~300K | **2.26M** |
| Years covered | 1 snapshot | 6 months | Limited | **11 years** |
| Full loan lifecycle | ❌ | Partial | Partial | ✅ |
| Multi-stage outcome labels | ❌ | ❌ | Partial | ✅ |
| Behavioral + origination data | ❌ | Partial | Partial | ✅ |
| Internal risk grading | ❌ | ❌ | ❌ | ✅ (Grade A–G) |
| Multi-economic-cycle coverage | ❌ | ❌ | ❌ | ✅ (GFC + recovery) |

### Key Differentiators

1. **Full lifecycle** — origination → monthly payments → terminal outcome (not just a snapshot)
2. **Multi-stage labels** — `Late (16-30)`, `Late (31-120)`, `Charged Off`, `Default` — enables pre-delinquency detection, not just end-state classification
3. **Delta features possible** — both origination FICO and current FICO are present, enabling `fico_drop` (impossible with single-snapshot datasets)
4. **Real data** — actual US consumer marketplace, not synthetic or heavily anonymized; historical patterns are genuine
5. **11-year span** — covers GFC (2008–09), recovery (2011–14), and expansion (2015–18), giving the historian model exposure to multiple macro regimes

---

## Part 7 — Recommended Next Steps

```
Current State: Preprocessed 31-feature vector (500K rows) ✅

Next Pipeline Steps:
┌──────────────────────────────────────────────────────┐
│  1. Feature Scaling  — RobustScaler (outlier-safe)   │
│  2. Class Balancing  — SMOTE or class_weight         │
│  3. Cohort Builder   — KMeans clustering on          │
│                         origination features         │
│  4. Model Training   — XGBoost / LightGBM baseline  │
│  5. Vector Encoder   — Deep Autoencoder for          │
│                         embedding space              │
│  6. Similarity Layer — Cosine distance for           │
│                         peer trajectory matching     │
│  7. Alert Engine     — Threshold-based drift         │
│                         detector on vector space     │
└──────────────────────────────────────────────────────┘
```
