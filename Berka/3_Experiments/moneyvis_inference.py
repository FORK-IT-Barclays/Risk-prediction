"""
VECTOR: MoneyVis Testing Pipeline
=================================================
Converts actual UK MoneyVis transactions into the Universal Vector Schema,
generates ~50 sliding 6-month trajectory windows, and scores them through
the XGBoost Behavioral Expert.
"""

import os
import pandas as pd
import numpy as np
import joblib

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONEYVIS_PATH = os.path.join(WORKSPACE, "dataset", "MoneyVis.csv")
MODEL_PATH = os.path.join(WORKSPACE, "Output_Artifacts", "behavioral_engine_v2.pkl")

# Load compiled Model
try:
    engine = joblib.load(MODEL_PATH)
    model = engine["model"]
    FEATURES = engine["features"]
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 1. Load and Standardise MoneyVis
print("Loading MoneyVis dataset...")
mv = pd.read_csv(MONEYVIS_PATH)
mv["date"] = pd.to_datetime(mv["Transaction Date"], format="%d/%m/%Y")
mv["cash_in"] = mv["Credit Amount"].fillna(0.0).astype(float)
mv["cash_out"] = mv["Debit Amount"].fillna(0.0).astype(float)
mv["balance"] = mv["Balance"].astype(float)

# Map descriptions to universal tags mimicking the Berka approach
def get_tag(desc, ty):
    desc = str(desc).upper()
    if ty in ["BGC", "FPI"] or "UNIV OF" in desc: return "SALARY"
    if ty == "DD" or "VIRGIN" in desc or "O2" in desc or "OCTOPUS" in desc or "CITY COUNC" in desc: return "BILL"
    return "UNCATEGORIZED"

mv["tag"] = mv.apply(lambda x: get_tag(x["Transaction Description"], x["Transaction Type"]), axis=1)

# Sort chronologically
mv = mv.sort_values("date").reset_index(drop=True)
min_date, max_date = mv["date"].min(), mv["date"].max()

print(f"Found {len(mv)} transactions spanning {min_date.date()} to {max_date.date()}")

# 2. Generate 50 Sliding Windows
# We will slide a 180-day window starting from the end, stepping back every 7 days (1 week stride)
# to generate exactly 50 recent snapshots.

rows = []
# Start checking from the maximum date, stepping back 7 days at a time
ref_dates = [max_date - pd.Timedelta(days=7 * i) for i in range(50)]

for ref_date in ref_dates:
    t2_end = ref_date
    t2_start = t2_end - pd.Timedelta(days=90)
    t1_end = t2_start
    t1_start = t1_end - pd.Timedelta(days=90)
    
    if t1_start < min_date:
        break # Out of history
        
    df_t2 = mv[(mv["date"] >= t2_start) & (mv["date"] < t2_end)]
    df_t1 = mv[(mv["date"] >= t1_start) & (mv["date"] < t1_end)]
    
    if len(df_t1) < 5 or len(df_t2) < 5:
        continue # Insufficient data
        
    # Calculate Features exactly as in the training script
    in_t1 = df_t1["cash_in"].sum(); in_t2 = df_t2["cash_in"].sum()
    bal_t1 = df_t1["balance"].mean(); bal_t2 = df_t2["balance"].mean()
    od_t1 = (df_t1["balance"] < 0).sum(); od_t2 = (df_t2["balance"] < 0).sum()
    
    # Salary specific
    sal_t1 = df_t1[df_t1["tag"]=="SALARY"]["cash_in"].sum()
    sal_t2 = df_t2[df_t2["tag"]=="SALARY"]["cash_in"].sum()
    
    EPS = 1e-6
    inc_v = (in_t2 - in_t1) / (abs(in_t1) + EPS)
    bal_v = (bal_t2 - bal_t1) / (abs(bal_t1) + EPS)
    od_v  = od_t2 - od_t1
    sal_v = (sal_t2 - sal_t1) / (abs(sal_t1) + EPS)
    tx_v  = (len(df_t2) - len(df_t1)) / (abs(len(df_t1)) + EPS)
    
    # ── Macroeconomic Normalisation (1998 CZK vs 2021 GBP) ──
    # We do NOT normalise using MoneyVis statistical data (which would cause leakage).
    # Instead, we apply a strict macroeconomic conversion factor. 
    # In 1998, 1 GBP ≈ 55 CZK. Adjusting for 20 years of UK/CZ inflation parity 
    # to 2021 (when MoneyVis occurs), the effective purchasing power parity (PPP) 
    # multiplier for bringing 2021 GBP into the 1998 CZK model training space is roughly 35.0.
    ECONOMIC_PPP_SCALER = 35.0
    
    rows.append({
        "reference_date": ref_date.date(),
        "income_erosion_v": np.clip(inc_v, -5, 5),
        "liquidity_momentum_v": np.clip(bal_v, -5, 5),
        "overdraft_v": np.clip(od_v, -5, 5),
        "overdraft_t2": od_t2,
        "salary_drift_v": np.clip(sal_v, -5, 5),
        "tx_freq_v": np.clip(tx_v, -5, 5),
        "avg_balance_t2": bal_t2 * ECONOMIC_PPP_SCALER,
        "min_balance_t2": df_t2["balance"].min() * ECONOMIC_PPP_SCALER,
        "total_out_t2": df_t2["cash_out"].sum() * ECONOMIC_PPP_SCALER
    })

results_df = pd.DataFrame(rows)

# 3. Score against the Engine
if len(results_df) > 0:
    X_test = results_df[FEATURES].values
    probs = model.predict_proba(X_test)[:, 1]
    results_df["Default_Probability"] = np.round(probs * 100, 1)
    results_df["Is_Distressed"] = probs >= 0.46
    
    # Sort chronologically for display
    results_df = results_df.sort_values("reference_date").reset_index(drop=True)
    
    print(f"\n========================================================")
    print("MONEYVIS DEPLOYMENT RESULTS (Real-Time Inference Test)")
    print("========================================================")
    print(f"Generated {len(results_df)} temporal records tracing customer trajectory.\n")
    print(results_df[["reference_date", "liquidity_momentum_v", "avg_balance_t2", "Default_Probability", "Is_Distressed"]].to_string())
    
    # Save test predictions
    out_path = os.path.join(WORKSPACE, "Output_Artifacts", "moneyvis_predictions.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved inference trace -> {out_path}")
else:
    print("Could not generate enough windows due to history depth.")
