# 🪐 Universal Financial Historian (Agnostic Underwriting)

This standalone module assesses macroeconomic credit risk using **Structural Leverage Math** rather than proprietary bureau scores (like FICO). It is designed to be universally deployable across any financial dataset that contains basic transactional or snapshot structural data.

---

## 🏗️ Architecture & Features

The model is powered by an aggressively optimized **XGBoost Engine** trained on 2.2 million rows from the Lending Club dataset. It focuses on **13 Agnostic Features** stripped of all U.S.-specific or platform-specific biases:

*   **Financial Velocity/Ratio Features:** `loan_to_income_ratio`, `installment_burden`, `dti`.
*   **Structural Liquidity:** `revol_util`, `revol_bal`, `open_acc`.
*   **Macro Foundation:** `annual_inc`, `loan_amnt`, `term_months`.
*   **Historical Failure Flags:** `delinq_2yrs`, `pub_rec`, `inq_last_6mths`.

---

## 📁 Environment Structure

| Folder | Contents |
|---|---|
| `src/` | **`universal_data_pipeline.py`**: Extracts and cleans the 13 features.<br>**`train_universal_model.py`**: Trains the XGBoost engine. |
| `data/` | **`universal_historian_data.parquet`**: [IGNORED] The processed 2.2M row matrix. |
| `models/` | **`universal_historian_v1.pkl`**: [IGNORED] The serialized model artifact. |

---

## ⚡ Quick Start

1. **Extraction**: Run the pipeline to build the structural matrix from the raw dataset.
   ```bash
   python src/universal_data_pipeline.py
   ```
2. **Training**: Execute the training script to generate the production model and evaluate performance metrics.
   ```bash
   python src/train_universal_model.py
   ```

---

## 📊 Performance at Scale (FICO-Free)
*   **Test Sample Size**: ~930,000 Unseen Loans (2017-2018)
*   **Recall (Default Catch Rate)**: **80.5%**
*   **Precision (Flag Success Rate)**: **9.4%**
*   **Decision Threshold**: `0.40`
