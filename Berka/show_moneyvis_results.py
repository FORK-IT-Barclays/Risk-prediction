import pandas as pd
import os

# Load the latest predictions
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
pred_path = os.path.join(WORKSPACE, "Output_Artifacts", "moneyvis_predictions.csv")

if os.path.exists(pred_path):
    df = pd.read_csv(pred_path)
    
    print("\n" + "="*70)
    print("MONEYVIS NORMALIZED TEST RESULTS (Last 15 Sliding Windows)")
    print("="*70)
    
    cols = ["reference_date", "Default_Probability", "Is_Distressed", "avg_balance_t2", "liquidity_momentum_v"]
    print(df[cols].tail(15).to_string(index=False))
    
    distressed_count = df["Is_Distressed"].sum()
    print("\n" + "-"*70)
    print(f"Summary: {distressed_count} distressed windows found out of {len(df)} total snapshots.")
    print(f"Mean Risk: {df['Default_Probability'].mean():.2f}%")
    print("="*70)
else:
    print(f"Error: Predictions file not found at {pred_path}")
