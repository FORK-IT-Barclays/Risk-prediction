## 🏦 Project Overview

This is a **credit risk / loan default prediction** system built for Barclays, using the Lending Club public dataset (2007–2018). It has two distinct ML tracks:

---

## 📁 Directory Structure

| Path | Purpose |
|---|---|
| [accepted_2007_to_2018Q4.csv](accepted_2007_to_2018Q4.csv) | Raw accepted loan data (~1.6 GB) |
| [rejected_2007_to_2018Q4.csv](rejected_2007_to_2018Q4.csv) | Raw rejected loan data (~1.7 GB) |
| [data.csv](data.csv) | Raw open-banking ledger (for MoneyViz) |
| `data/processed/` | Cleaned/preprocessed parquet & CSV files |
| `src/` | All Python source scripts |
| `models/` | Open Banking surrogate model artifacts |
| `outputs/models/` | Main Financial Historian model artifacts |
| `outputs/plots/` | All generated charts (27 PNGs) |
| `reports/` | Text reports + HTML risk report |
| `notebooks/` | EDA notebook + helper scripts |

---

## 🧠 Two ML Tracks

### Track 1 — "Financial Historian" (Main Model)
A **4-layer composite risk scoring system** for predicting loan defaults:

```
Composite Score = 50% XGBoost + 25% Cohort Risk + 15% Rule Layer + 10% Anomaly
```

| Layer | Component | Script |
|---|---|---|
| Core | XGBoost classifier | `src/train.py` |
| Layer 1 | KMeans cohort clustering (20 clusters) | `src/features.py` |
| Layer 2 | Hard threshold rules (FICO drop, DTI, etc.) | `src/features.py` |
| Layer 3 | Isolation Forest (anomaly detection) | `src/train.py` |

**Feature split** (from `config.py`):
- **T=0 features** (20): Known at origination — `loan_amnt`, `int_rate`, `fico_avg`, `dti`, `grade_enc`, etc.
- **Behavioral features** (5): Mid-loan signals — `fico_drop`, `repay_ratio`, `late_fee_flag`, etc.

**Temporal train/val/test split** — no data leakage:
- Train: 2012–2016 | Val: 2017 | Test: 2018

**Final model performance** (held-out 2018 test):
- ROC-AUC: **0.8857** | PR-AUC: **0.3246** | Recall: **99.0%** | FNR: **1.03%**
- Decision threshold: `0.30` (tuned for high recall / low missed defaults)

### Track 2 — "Open Banking Surrogate" (MoneyViz Pipeline)
A lighter model that scores raw **bank transaction ledgers** (no credit bureau data):

```
data.csv (bank ledger) → moneyviz_transformer.py → data_transformed.csv
                                                  → train_surrogate.py (SMOTE + XGBoost)
                                                  → score_accounts.py → Risk_Report.html
```

It extracts 4 proxy features from transactions: `annual_inc`, `dti`, `loan_to_income`, `delinq_2yrs`

---

## 🔄 Execution Order (Main Pipeline)

```
1.  src/clean.py          → clean raw parquet → lending_club_clean.parquet
2.  src/train.py          → train + evaluate model → outputs/models/*.pkl
3.  src/train_final.py    → retrain on ALL data → outputs/models/*_FINAL.pkl
```

**Open Banking path:**
```
4.  src/moneyviz_transformer.py  → transform ledger → data_transformed.csv
5.  src/train_surrogate.py       → train surrogate → models/open_banking_surrogate.pkl
6.  src/score_accounts.py        → score + generate → reports/Risk_Report.html
```

---

## ⚙️ Key Config (`src/config.py`)
- All paths, feature lists, XGBoost hyperparams, and temporal split years are centralized here — this is the **single source of truth** to tweak.
- Decision threshold `0.30` is intentionally below 0.5 to **maximize recall** (catch more defaulters at the cost of some precision).

---

## 📊 Outputs Already Generated
- **27 plots** in `outputs/plots/` (EDA, confusion matrix, ROC/PR curves, feature importance, threshold sensitivity, etc.)
- **Trained `.pkl` files** for both the main and final production models
- **`reports/Risk_Report.html`** — browser-viewable risk scoring report for open-banking accounts