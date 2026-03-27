# VECTOR Pre-Delinquency Shield — Master Project Architecture

A comprehensive guide to the full project covering every folder, every file, every model, and every design decision.

---

## 🏗️ 1. Project Overview

**VECTOR** (Velocity-Enhanced Credit Trajectory for Operational Risk) is a multi-expert risk engine that detects loan defaults **before they happen** by analyzing both the structural stability and the behavioral trajectory of a customer.

### Why This Exists
Traditional banks detect default **after** a payment is missed. VECTOR detects it **weeks before** by measuring the *physics* of money flow — how fast savings are dropping and whether that speed is accelerating.

### The Two Experts
| Expert | Model | Trained On | Detects |
|---|---|---|---|
| **Financial Historian** | XGBoost | Lending Club (2.2M US loans) | Structural weakness (low income, high debt) |
| **Behavioral Analyst** | CatBoost | Berka / PKDD99 (Czech bank txns) | Behavioral decay (spending spikes, salary drift) |

### The Unified Formula
```
Final_Risk = Historian_Prob + (Behavioral_Prob - 0.46) * Weight
```
- If behavior is **better** than baseline → score drops (Reward)
- If behavior is **worse** than baseline → score rises (Penalty)
- **0.46** = F2-Optimal threshold tuned via Optuna

---

## 📂 2. Full Directory Tree

```
E:\Risk-prediction\
│
├── .gitignore                          # Master manifest (blocks CSVs, whitelists .pkl/.png/.md)
├── requirements.txt                    # Python dependencies
├── barclaysppt.pdf                     # Original Barclays hackathon brief
│
├── lending_club/                       # 🏦 Native Lending Club module
│   ├── accepted_2007_to_2018Q4.csv     # RAW dataset (1.67 GB, 2.2M loans)
│   ├── rejected_2007_to_2018Q4.csv     # Rejected loans dataset (1.78 GB)
│   ├── data.csv                        # Legacy small subset (420 KB)
│   ├── README.md                       # Module documentation
│   ├── data/                           # Intermediate data artifacts
│   ├── notebooks/                      # Jupyter exploration notebooks
│   ├── src/                            # Native model training scripts
│   ├── outputs/                        # Native model output plots
│   ├── reports/                        # Native evaluation reports
│   └── moneyviz_surrogate/             # MoneyVis surrogate data (blocked from Git)
│
├── Berka/                              # 🌀 Behavioral Analyst module
│   ├── 1_Feature_Engineering/
│   │   └── feature_engineering.py      # 9-signal extraction from Berka time-series
│   ├── 2_Model_Training/
│   │   ├── model_algorithm_race.py     # Benchmarks XGBoost vs CatBoost vs LightGBM
│   │   ├── model_behavioral_engine.py  # CatBoost V1 training script
│   │   └── model_v2_tuned.py           # CatBoost V2 (Optuna-tuned, F2-optimized)
│   ├── 3_Experiments/                  # Experimental scripts and logs
│   ├── Output_Artifacts/
│   │   ├── behavioral_engine_v2.pkl    # Trained CatBoost model artifact
│   │   ├── feature_importance.png      # Top feature weights
│   │   ├── roc_curve.png               # ROC performance curve
│   │   └── moneyvis_predictions.csv    # MoneyVis test predictions
│   ├── implementation plan/
│   │   └── real_time_risk_plan.md      # Real-time deployment architecture
│   ├── EDA_workbook.py                 # Exploratory data analysis
│   ├── dataset_loader.py              # Berka PKDD99 data ingestion
│   ├── feature_definitions.md          # 9-signal mathematical definitions
│   └── behavioral_model_eval.md        # Model performance evaluation
│
├── universal_historian/                # 🏛️ Universal Financial Historian
│   ├── data/
│   │   └── universal_historian_data.parquet  # Cleaned dataset (50 MB, 13 features)
│   ├── models/
│   │   ├── universal_historian_v1.pkl        # Trained XGBoost model (~1 MB)
│   │   └── universal_features_map.pkl        # Feature alignment map
│   ├── src/
│   │   ├── universal_data_pipeline.py        # 5-step data cleaning pipeline
│   │   ├── train_universal_model.py          # XGBoost training with scale_pos_weight
│   │   └── generate_visuals.py               # ROC, PR, confusion matrix generator
│   ├── outputs/plots/                        # Generated diagnostic plots
│   ├── docs/faq/                             # Technical FAQ documents
│   └── README.md                             # Module documentation
│
├── realtime_risk_engine/               # ⚡ Real-Time Physics Engine
│   ├── src/
│   │   ├── config.py                   # All constants (thresholds, paths, scalers)
│   │   ├── transformer.py              # UK MoneyVis → Universal tag mapper
│   │   ├── feature_engine.py           # 9-signal V&A calculator (original)
│   │   ├── physics_core.py             # Hybrid Physics (Macro/Micro/Event)
│   │   └── inference.py                # Main VectorPredictor API entry point
│   ├── models/
│   │   └── behavioral_engine_v2.pkl    # Deployed CatBoost model copy
│   ├── tests/                          # Smoke tests and validation scripts
│   └── reports/                        # Engine-specific reports
│
├── financial_historian_dataset/        # 📥 Dataset Setup Guide
│   ├── raw/                            # Place raw CSVs here
│   ├── processed/                      # Pipeline outputs
│   └── README.md                       # Setup instructions
│
└── docs/                               # 📚 Master Documentation
    ├── VECTOR_PRODUCTION_ARCHITECTURE.md  # 3-Layer system overview
    └── META_PHYSICS_DEEP_DIVE.md         # Trajectory regression deep-dive
```

---

## 🧠 3. The Models — What, How, and Why

### Model 1: Financial Historian (XGBoost)

**What it does:** Scores a customer's **structural capacity** to absorb financial shocks using 13 static features measured at loan origination (T=0).

**How it works:**
1. Ingests raw Lending Club data (2.2M loans, 2007–2018).
2. Cleans via `universal_data_pipeline.py`: filters post-2012, interpolates sentinel values, winsorizes tails, derives ratios.
3. Trains XGBoost with `scale_pos_weight=5.3` to aggressively penalize missed defaults.
4. Outputs a probability (0–1) representing structural risk.

**Why XGBoost:**
- Tree-based models handle outliers naturally (unlike Neural Networks).
- Built-in `scale_pos_weight` handles class imbalance without SMOTE (no fake borrowers).
- Fully interpretable via Feature Importance maps.

**Key Features (13):**
`annual_inc`, `loan_amnt`, `dti`, `term`, `installment`, `open_acc`, `total_acc`, `revol_bal`, `revol_util`, `delinq_2yrs`, `pub_rec`, `inq_last_6mths`, `loan_to_income_ratio`

**Performance:** AUC-ROC ~0.72, F2-Score optimized for recall.

---

### Model 2: Behavioral Analyst (CatBoost V2)

**What it does:** Scores a customer's **real-time behavioral trajectory** by analyzing the velocity and acceleration of 9 high-frequency transaction signals.

**How it works:**
1. Ingests Czech banking data (Berka PKDD99 dataset).
2. Engineers 9 behavioral signals via `feature_engineering.py`:
   - Salary timing drift, income erosion, liquidity momentum
   - Payment integrity, failed auto-debits, credit exhaustion
   - Transaction frequency decay, overdraft velocity
3. Trains CatBoost V2 with Optuna hyperparameter sweep.
4. Uses F2-Optimal threshold of **0.46** (maximizes recall for distress detection).

**Why CatBoost:**
- Superior handling of categorical features (transaction types).
- Native ordered boosting prevents target leakage in time-series.
- Won the algorithm race against XGBoost and LightGBM in `model_algorithm_race.py`.

**Performance:** AUC-ROC ~0.89, F2-Score optimized at threshold 0.46.

---

## ⚡ 4. The Physics Engine — How Risk is Calculated

### The 3-Layer Waterfall

```
Transaction arrives
        ↓
┌─────────────────────────────┐
│  Layer 1: Financial Historian│  → Structural baseline (DTI, Income)
│  Output: Historian_Prob      │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Layer 2: Behavioral Analyst │  → V&A trajectory (Spending, Salary)
│  Output: Behavioral_Prob     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  Layer 3: Meta-Physics       │  → Regression on risk-over-time
│  Output: Days_to_Default     │
└──────────────┬──────────────┘
               ↓
        Unified Verdict
```

### The MoneyVis Transformer (`transformer.py`)
- Maps UK bank statement descriptions to universal tags: `SALARY`, `BILL`, `GROCERY`, etc.
- Uses `ECONOMIC_PPP_SCALER = 35.0` to convert Czech-trained model outputs to UK currency contexts.
- Why 35.0: Based on the PPP ratio between Czech Koruna and British Pound.

### The Physics Core (`physics_core.py`)
- **Macro-Velocity (30d):** Compares 3 consecutive 30-day windows to calculate the rate of balance decay.
- **Micro-Velocity (14d):** Detects sudden 7–14 day spending spikes that indicate behavioral shock.
- **Instant Events:** Zero-latency triggers for failed payments and overdrafts.

### The Meta-Physics Layer (Trajectory Regression)
- Collects `(timestamp, risk_score)` pairs over a **14-day fixed window**.
- Fits a **Weighted Least-Squares** line where today's data has 10x the weight of old data.
- Calculates:
  - **Risk Velocity ($\beta$):** How fast risk is rising per day.
  - **Risk Acceleration ($\beta'$):** Whether the rise is speeding up.
  - **Days-to-Default:** Projected date when risk crosses 0.75.
- **$R^2$ Confidence Guard:** If the trend fit is poor (< 0.70), the projection is ignored to prevent false alarms from one-time "Wedding Spikes."

---

## 🔧 5. Key Source Files — What Each One Does

### `realtime_risk_engine/src/`
| File | Purpose |
|---|---|
| `config.py` | All constants: PPP scaler (35.0), F2 threshold (0.46), window sizes, model paths |
| `transformer.py` | Converts raw UK bank statements into universal tagged format |
| `feature_engine.py` | Original 9-signal V&A calculator (180-day lookback) |
| `physics_core.py` | Hybrid Physics: Macro (30d) + Micro (14d) + Instant Events |
| `inference.py` | Main API: `VectorPredictor.predict_risk()` — orchestrates all layers |

### `universal_historian/src/`
| File | Purpose |
|---|---|
| `universal_data_pipeline.py` | Cleans 2.2M loans → 13 features → `.parquet` |
| `train_universal_model.py` | Trains XGBoost with imbalance correction |
| `generate_visuals.py` | Creates ROC, PR, confusion matrix, feature importance plots |

### `Berka/`
| File | Purpose |
|---|---|
| `1_Feature_Engineering/feature_engineering.py` | Extracts 9 behavioral signals from raw Berka transactions |
| `2_Model_Training/model_v2_tuned.py` | Optuna-tuned CatBoost V2 training |
| `2_Model_Training/model_algorithm_race.py` | Benchmarks XGBoost vs CatBoost vs LightGBM |
| `dataset_loader.py` | Loads and preprocesses PKDD99 Czech banking data |
| `feature_definitions.md` | Mathematical definitions of all 9 behavioral signals |

---

## 🛡️ 6. Security & Git Policy

The root `.gitignore` enforces a "Strict-First" policy:

| Blocked | Whitelisted |
|---|---|
| `*.csv`, `*.parquet` (raw data) | `*.pkl` (trained models) |
| `moneyviz_surrogate/` (sensitive) | `*.png` (diagnostic plots) |
| `*.pdf` (investigation briefs) | `*.md` (documentation) |
| `__pycache__/`, `*.pyc` | All source code (`.py`) |

---

## 📋 7. Dependencies

```
pandas
numpy
scikit-learn
xgboost
catboost
joblib
matplotlib
seaborn
optuna
```

---
*VECTOR Pre-Delinquency Shield — Master Architecture v3.0*
*Last Updated: 2026-03-27*
