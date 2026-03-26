# 🏦 Universal Financial Historian

This is a **structural credit risk prediction engine** built for Barclays. Initially trained on the massive Lending Club public framework (2007–2018), it acts as a Generalized Structural Underwriter. It achieves this by assessing the foundational financial structure of an applicant rather than monitoring live behavioral transactions.

---

## 📁 Directory Structure

| Path | Purpose |
|---|---|
| `accepted_2007_to_2018Q4.csv` | Raw accepted loan data (Training input) |
| `rejected_2007_to_2018Q4.csv` | Raw rejected loan data (Not processed) |
| `data/processed/` | Cleaned/preprocessed parquet & CSV files |
| `src/` | Core Python inference and training scripts (`train.py`, `features.py`, `clean.py`) |
| `outputs/models/` | Production serialized `.pkl` models |
| `outputs/plots/` | Visualizations of threshold sensitivity and ROC-AUC curves |
| `reports/` | Summaries of model performance |
| `notebooks/` | Interactive dynamic 19-step EDA generation scripts |
| `moneyviz_surrogate/` | *[QUARANTINED]* Deprecated behavioral scripts |

---

## 🧠 The Architecture (4-Layer Composite)

The system avoids relying on a single "black-box" decision tree. To ensure interpretability and robust edge-case handling, it routes the final score through a **4-layer ensemble composite**:
```text
Composite Score = 50% XGBoost + 25% Cohort KMeans + 15% Expert Rules + 10% Isolation Forest
```

| Layer | Component | Execution Script |
|---|---|---|
| **Core** | **XGBoost Classifier** (Trained specifically via `scale_pos_weight` rather than SMOTE) | `src/train.py` |
| **Layer 1** | **KMeans Cohort Clustering** (Identifies historical similarity to Defaulters) | `src/features.py` |
| **Layer 2** | **Hard Threshold Rules** (Manual DTI & Payment penalty overlays) | `src/features.py` |
| **Layer 3** | **Isolation Forest** (Identifies statistically impossible data combinations as fraud proxy) | `src/train.py` |

---

## ⚙️ The Transformation Pipeline
Before scoring, all inputs run through a strict 7-step mathematical normalization system inside `src/clean.py` & `src/features.py`:
1. **Sentinel Imputation:** Removing algorithm-breaking flags like `999` Constraints and overwriting impossible median values.
2. **Winsorization:** All extreme monetary amounts (Incomes, Balances) are mathematically compressed to the `0.5th -> 99.5th percentiles` to prevent hyper-wealthy outliers from corrupting the continuous XGBoost thresholds.
3. **Structured Derivation:** Constructs massive new burden ratios like `loan_to_income` and `installment_burden` directly from the cleaned base values, presenting a ratio-driven reality to the AI engine.

---

## 🔄 Execution Order

To run the Financial Historian pipeline from scratch:
```bash
1. python src/clean.py          # Cleans raw parquet to `lending_club_clean.parquet`
2. python src/train.py          # Trains the 4-layer architecture & evaluates temporal performance
3. python src/train_final.py    # Retrains the validated architecture on 100% of available data
```

**Final Model Performance (2018 Held-Out Test Set):**
- **Decision Threshold:** `0.30` (Calibrated hyper-aggressively to maximize recall over precision)
- **Recall:** `99.0%` (Successfully captures almost every true default prior to issuance to protect portfolio capital)
- **ROC-AUC:** `0.8857` (Exceptionally predictive ranking probability)