# Updated Implementation Plan: Real-Time Vector Risk Engine (v2)

This plan integrates the **VECTOR v2 Behavioral Model** into a real-time Kafka environment, specifically optimized to handle UK-style transaction streams (based on the MoneyVis schema) and isolated per-customer inference.

---

## 🏗️ 1. System Architecture

```mermaid
graph TD
    subgraph "Phase 1: Transaction Streaming"
        MS[MoneyVis Simulator] -->|Streams UK Schema| K[Kafka Topic: raw_transactions]
    end

    subgraph "Phase 2: Feature Engineering Layer"
        K -->|Consume| ML[Mapping Layer: standardizer.py]
        ML -->|Standardize Logic| FE[Feature Engine]
        FE -->|Save Universal Ledger| SS[(State Store: Isolated History per User)]
    end

    subgraph "Phase 3: Automated Risk Scoring"
        UI[Prediction Dashboard] -->|'Predict' Request| API[Inference API]
        API -->|Fetch isolated history| SS
        API -->|Apply PPP Scaler 35.0| RE[VECTOR v2 Engine]
        RE -->|Risk Probability| API
        API -->|Response| UI
    end
```

---

## 🛠️ 2. Component Breakdown

### I. Data Simulation (MoneyVis Producer)
*   **Source:** Uses [MoneyVis.csv](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/dataset/MoneyVis.csv) as the raw template.
*   **Schema:** Streams `Transaction Date`, `Type`, `Description`, `Debit`, `Credit`, `Balance`.
*   **Isolated Identities:** The producer will loop over the dataset but assign unique IDs to simulate multiple different customers acting simultaneously.

### II. Mapping & Transformation Layer
As detailed in [moneyvis_feature_engineering_report.md](file:///e:/Machine%20learning/barclayss/risk-prediction/Berka/moneyvis_feature_engineering_report.md), this middleware will:
*   **Standardize:** Convert UK descriptions (e.g., "BGC", "DD", "UNIV OF") into universal tags: `SALARY`, `BILL`, `UNCATEGORIZED`.
*   **Window Management:** Ensure every transaction is mapped to a customer's temporal sequence.
*   **Persistence:** Store transactions in a way that allows fast retrieval of the last 180 days for a *specific* user without data leakage from other users.

### III. Isolated Risk Processor (Vector v2 Engine)
Upon a trigger (button hit):
1.  **Context Loading:** Fetch the exact transaction history for the target `account_id` only.
2.  **Velocity Calculation:** Compute the **9 signals** (Income Erosion, Liquidity Momentum, etc.) using the 90-day T1/T2 split.
3.  **Economic Alignment:** Multiply absolute currency outputs (`avg_balance`, `total_out`, `min_balance`) by the **ECONOMIC_PPP_SCALER = 35.0**.
4.  **Threshold Enforcement:** Apply the **0.46 F2-Optimal threshold** from the V2 model to generate the `Distressed` flag.

---

## 🚀 3. Execution Roadmap

### Step 1: Schema Lockdown
Define the Pydantic/JSON schema for the `raw_transactions` topic to mirror [MoneyVis.csv](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/dataset/MoneyVis.csv).

### Step 2: The Standardization Worker
Implement a Python worker that consumes from Kafka and applies the [get_tag()](file:///e:/Machine%20learning/barclayss/risk-prediction/Berka/3_Experiments/moneyvis_inference.py#36-41) keyword-matching logic from the inference scripts.

### Step 3: State Store Setup 
Deploy a database (e.g., PostgreSQL) indexed by `account_id` and `timestamp` to ensure the risk engine pulls isolated histories in sub-100ms.

### Step 4: The V2 Model API 
Wrap the [behavioral_engine_v2.pkl](file:///e:/Machine%20learning/barclayss/VECTOR/Berka/Output_Artifacts/behavioral_engine_v2.pkl) in a FastAPI service that performs the final windowing and mathematical scaling.

---
*Drafted based on VECTOR behavioral model v2 architecture and MoneyVis feature engineering logic.*
