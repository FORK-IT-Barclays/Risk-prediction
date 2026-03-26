import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set styles
sns.set_theme(style="whitegrid")

# Paths
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(WORKSPACE, "dataset", "vector_features_v4.csv")
MONEYVIS_PRED_PATH = os.path.join(WORKSPACE, "Output_Artifacts", "moneyvis_predictions.csv")

# 1. Load data
df_train = pd.read_csv(TRAIN_DATA_PATH)
df_mv = pd.read_csv(MONEYVIS_PRED_PATH)

# Features to compare (especially the absolute currency ones)
CURRENCY_FEATURES = ["avg_balance_t2", "min_balance_t2", "total_out_t2"]
VELOCITY_FEATURES = ["income_erosion_v", "liquidity_momentum_v", "overdraft_v", "tx_freq_v"]

print("--- Comparison of Feature Distributions (Mean Values) ---")
print(f"{'Feature':<20} | {'Berka (Train)':>15} | {'MoneyVis (Test)':>15}")
print("-" * 55)

for feat in CURRENCY_FEATURES + VELOCITY_FEATURES:
    m_train = df_train[feat].mean()
    m_mv = df_mv[feat].mean()
    print(f"{feat:<20} | {m_train:>15.2f} | {m_mv:>15.2f}")

# Visualize the main culprit: balance
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(data=[df_train["avg_balance_t2"], df_mv["avg_balance_t2"]], ax=axes[0], palette="Set2")
axes[0].set_xticklabels(["Berka (Training)", "MoneyVis (Scaled)"])
axes[0].set_title("Comparison of Average Balances (Scaled)")
axes[0].set_ylabel("Value (CZK/Scaled GBP)")

sns.boxplot(data=[df_train["total_out_t2"], df_mv["total_out_t2"]], ax=axes[1], palette="Set2")
axes[1].set_xticklabels(["Berka (Training)", "MoneyVis (Scaled)"])
axes[1].set_title("Comparison of Total Expenses (Scaled)")
axes[1].set_ylabel("Value (CZK/Scaled GBP)")

plt.tight_layout()
out_plot = os.path.join(WORKSPACE, "Output_Artifacts", "feature_mismatch_check.png")
plt.savefig(out_plot, dpi=150)
plt.close()

print(f"\nVisualization saved -> {out_plot}")
