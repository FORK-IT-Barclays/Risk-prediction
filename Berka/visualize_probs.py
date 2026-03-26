import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set styles
sns.set_theme(style="whitegrid")

# Paths
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(WORKSPACE, "Output_Artifacts", "behavioral_engine_v2.pkl")
MONEYVIS_PRED_PATH = os.path.join(WORKSPACE, "Output_Artifacts", "moneyvis_predictions.csv")

# 1. Load Model
engine = joblib.load(MODEL_PATH)
model = engine["model"]
features = engine["features"]

# 2. Get Probabilities for MoneyVis (External Test)
df_mv = pd.read_csv(MONEYVIS_PRED_PATH)
mv_probs = df_mv["Default_Probability"] / 100.0 # already scaled to 100 in CSV

# 3. Create a more detailed distribution view
plt.figure(figsize=(10, 6))

# Plot MoneyVis (Test Distribution)
sns.histplot(mv_probs, bins=25, kde=True, color='darkorange', alpha=0.6)
plt.axvline(0.46, color='red', linestyle='--', label='Distress Threshold (0.46)')
plt.title("Exact Probability Distribution: MoneyVis Test Set (UK Data)")
plt.xlabel("Probability of Default")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
out_plot = os.path.join(WORKSPACE, "Output_Artifacts", "prob_distribution_moneyvis.png")
plt.savefig(out_plot, dpi=150)
plt.close()

print(f"Visualization saved -> {out_plot}")
print("\n--- Summary Statistics (MoneyVis Probabilities) ---")
print(mv_probs.describe())
print(f"\nTotal Records: {len(mv_probs)}")
print(f"Number Distressed: {(mv_probs >= 0.46).sum()}")
print(f"Number Healthy: {(mv_probs < 0.46).sum()}")
