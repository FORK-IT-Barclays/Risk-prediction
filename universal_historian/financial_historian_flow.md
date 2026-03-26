# 🌊 Universal Financial Historian: Operational Flow

This document visualizes the end-to-end mathematical journey of a customer's static profile as it travels through the **Universal Financial Historian** pipeline.

---

## 🗺️ The Architecture Flow

```mermaid
graph TD
    A[<b>RAW INPUT:</b><br/>Lending Club Accepted Loans<br/><i>(2.2 Million CSV Rows)</i>] --> B[<b>SELECTIVE INGESTION</b><br/>Only 12 Structural Columns Loaded<br/><i>RAM Optimization</i>]
    
    subgraph "The 5-Step Structural Transformation Pipeline"
    B --> C[<b>1. TEMPORAL FILTERING</b><br/>Post-2012 Only<br/><i>Modern Economy Standard</i>]
    C --> D[<b>2. SENTINEL IMPUTATION</b><br/>Invalid Values -> Medians<br/><i>Annual Inc < $100 Fixed</i>]
    D --> E[<b>3. WINSORIZATION</b><br/>Outliers -> 99.5th Percentile<br/><i>Wealth Compression</i>]
    E --> F[<b>4. RATIO CONSTRUCTION</b><br/>DTI + Loan-to-Income + Burden<br/><i>Self-Normalizing Math</i>]
    F --> G[<b>5. DENSE MATRIX VALIDATION</b><br/>Final Null Sweep<br/><i>Guaranteed Zero-Null Rowset</i>]
    end

    G --> H[<b>VECTORIZED PARQUET</b><br/>13 Agnostic Features<br/><i>(Memory Mapped Archive)</i>]
    
    subgraph "XGBoost Analytical Engine"
    H --> I[<b>TEMPORAL WALK-FORWARD SPLIT</b><br/>Train: 2012-2016<br/>Test: 2017-2018]
    I --> J[<b>SCALE POSITIVE WEIGHT</b><br/>Mathematical Imbalance Compensation<br/><i>Ratio ~4.7 to 1</i>]
    J --> K[<b>XGBOOST CORE</b><br/>PR-AUC Optimization<br/><i>(Learning Rate 0.08, Depth 6)</i>]
    end

    K --> L[<b>PREDICTIVE OUTPUT</b><br/>Static Risk Score<br/><i>(Recall: 80.5%)</i>]

    L --> M[<b>DECISION THRESHOLD</b><br/>Probability Cutoff @ 0.40]
    M --> N{<b>FINAL RATING</b>}
    N --> N1[<b>SAFE:</b> Proceed]
    N --> N2[<b>HIGH RISK:</b> Flag Default]
```

---

## ⚡ Stage-by-Stage Breakdown

### 1. Ingestion & Filtering
The pipeline ignores all behavioral noise and focuses strictly on **12 foundational structural variables** (Income, Inquiries, Debt, Term). It filters out pre-2012 data to ensure the model isn't learning from the 2008 financial crisis outliers which no longer match modern banking reality.

### 2. Mathematical Sanitization (T=0)
Instead of relying on human graders or credit bureaus (like FICO), the system uses **Winsorization** and **Sentinel Imputation** to mathematically "fix" impossible errors and compress extreme wealth into a range the AI can understand reliably.

### 3. Feature Compounding
The system doesn't just look at salary; it looks at **Salary relative to Debt** (`DTI`) and **Salary relative to the New Loan** (`installment_burden`). This creates a "Self-Calibrating" view of the applicant's risk level.

### 4. XGBoost Scaling
We use `scale_pos_weight` to handle the fact that most people pay their loans. This forces the model to be **Hyper-Averse to Defaults**, aggressively catching "Hidden Crashes" in the 13-feature matrix.

### 5. Final Thresholding
The Raw Score is passed through a **Decision Threshold (0.40)** optimized for maximum **Recall** (catching defaults), providing a final Go/No-Go rating for the financial institution.
