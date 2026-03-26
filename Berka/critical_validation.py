import pandas as pd
import numpy as np
import os

# Load data
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
df_train = pd.read_csv(os.path.join(WORKSPACE, "dataset", "vector_features_v4.csv"))
df_mv = pd.read_csv(os.path.join(WORKSPACE, "Output_Artifacts", "moneyvis_predictions.csv"))

# Calculate structural ratios
# Training Set (Berka)
df_train["min_rel_avg"] = df_train["min_balance_t2"] / df_train["avg_balance_t2"]
df_train["out_rel_avg"] = df_train["total_out_t2"] / df_train["avg_balance_t2"]

# Test Set (MoneyVis - Scaled)
# Note: Since I used different scalers, I need to check the RAW ratios or realize
# that the current CSV has the "Three-Tier" values.
# Let's calculate what the RAW ratio is in MoneyVis from the scaled values.
# RAW_MV_MIN / RAW_MV_AVG = (SCALED_MV_MIN / 86.5) / (SCALED_MV_AVG / 29.25)
df_mv["min_rel_avg_scaled"] = df_mv["min_balance_t2"] / df_mv["avg_balance_t2"]
df_mv["out_rel_avg_scaled"] = df_mv["total_out_t2"] / df_mv["avg_balance_t2"]

# RAW Ratio estimate:
df_mv["min_rel_avg_raw"] = (df_mv["min_balance_t2"] / 86.5) / (df_mv["avg_balance_t2"] / 29.25)
df_mv["out_rel_avg_raw"] = (df_mv["total_out_t2"] / 4.87) / (df_mv["avg_balance_t2"] / 29.25)

print("--- Structural Feature Ratios (Critical Validation) ---")
print(f"{'Metric':<30} | {'Berka (Train)':>15} | {'MoneyVis (Raw UK)':>15} | {'MoneyVis (Scaled)':>15}")
print("-" * 85)

metrics = [
    ("Min Balance / Avg Balance", "min_rel_avg", "min_rel_avg_raw", "min_rel_avg_scaled"),
    ("Total Out / Avg Balance", "out_rel_avg", "out_rel_avg_raw", "out_rel_avg_scaled")
]

for label, train_col, raw_col, scaled_col in metrics:
    m_train = df_train[train_col].median()
    m_raw = df_mv[raw_col].median()
    m_scaled = df_mv[scaled_col].median()
    print(f"{label:<30} | {m_train:>15.4f} | {m_raw:>15.4f} | {m_scaled:>15.4f}")

print("\n--- Insight ---")
print("If 'MoneyVis (Raw UK)' is significantly different from 'Berka', it represents a behavioral shift.")
print("If 'MoneyVis (Scaled)' matches 'Berka', we have 'normalized away' that shift.")
