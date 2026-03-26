import pandas as pd
import numpy as np
import os

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(WORKSPACE, 'dataset', 'vector_features_v4.csv')
moneyvis_path = os.path.join(WORKSPACE, 'Output_Artifacts', 'moneyvis_predictions.csv') # But these are already scaled!
moneyvis_raw_path = os.path.join(WORKSPACE, 'dataset', 'MoneyVis.csv')

df_train = pd.read_csv(train_path)

# Let me use the existing moneyvis predictions and just divide by the scaler to get the raw unscaled
df_mv_scaled = pd.read_csv(moneyvis_path)
df_mv_unscaled = df_mv_scaled.copy()
df_mv_unscaled['avg_balance_t2'] /= 20.0
df_mv_unscaled['min_balance_t2'] /= 20.0
df_mv_unscaled['total_out_t2'] /= 20.0

feats = ['avg_balance_t2', 'min_balance_t2', 'total_out_t2']

for f in feats:
    b_med = df_train[f].median()
    m_med = df_mv_unscaled[f].median()
    ratio = b_med / m_med if m_med != 0 else 0
    print(f"{f}: Berka_Med={b_med:.1f}, MV_Raw_Med={m_med:.1f} -> Required_Multiplier={ratio:.2f}")

