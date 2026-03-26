"""
VECTOR: Behavioral Signal Feature Engineering
=============================================
Pipeline:
  1. Standardise Berka trans.csv into unified Ledger
  2. Cap reference_date at dataset cutoff (fixes missing records bug)
  3. Generate up to 10 non-overlapping 6-month windows per account (~5 years history)
     Window 0: ref_date - 0   to -6m  ...
     Window 9: ref_date - 54m to -60m
  4. Each window produces its own T1/T2 velocity signals (independent row)
  5. Output: vector_features_v3.csv  (target ~5,000+ rows, same 1:8 imbalance ratio)
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
DATASET_CUTOFF = pd.Timestamp("1998-12-31")
EPS = 1e-6

# ─────────────────────────────────────────────
# STEP 1: Load Berka as Unified Ledger
# ─────────────────────────────────────────────
def load_berka_as_ledger():
    print("[1/4] Loading Berka transactions...")
    trans = pd.read_csv(os.path.join(DATA_DIR, "trans.csv"), sep=";", low_memory=False)
    ledger = pd.DataFrame()
    ledger["account_id"] = trans["account_id"]
    ledger["date"]       = pd.to_datetime(trans["date"].astype(str), format="%y%m%d")
    ledger["cash_in"]    = np.where(trans["type"] == "PRIJEM", trans["amount"], 0.0)
    ledger["cash_out"]   = np.where(trans["type"].isin(["VYDAJ","VYBER"]), trans["amount"], 0.0)
    ledger["balance"]    = trans["balance"].astype(float)
    k_map = {"SIPO":"BILL","UVER":"LOAN_PAYMENT","DUCHOD":"SALARY","POJISTNE":"INSURANCE"}
    op_map = {"VKLAD":"DEPOSIT","VYBER":"ATM_CASH","PREVOD ZUCTU":"TRANSFER","PREVOD NA UCET":"TRANSFER"}
    ledger["tag"] = (
        trans["k_symbol"].map(k_map)
        .fillna(trans["operation"].map(op_map))
        .fillna("UNCATEGORIZED")
    )
    return ledger

# ─────────────────────────────────────────────
# STEP 2: Labels + reference_date with cutoff cap
# ─────────────────────────────────────────────
def load_labels():
    loans = pd.read_csv(os.path.join(DATA_DIR, "loan.csv"), sep=";")
    loans["default"]  = loans["status"].map({"A":0,"C":0,"B":1,"D":1})
    loans["loan_date"] = pd.to_datetime(loans["date"].astype(str), format="%y%m%d")
    loans["reference_date"] = loans["loan_date"] + pd.to_timedelta(loans["duration"] * 30, unit="D")
    # FIX: cap at dataset cutoff so running loans still get a valid window
    loans["reference_date"] = loans["reference_date"].clip(upper=DATASET_CUTOFF)
    return loans[["account_id","loan_date","reference_date","default"]]

# ─────────────────────────────────────────────
# STEP 3: Compute signals for ONE 6-month window
# ─────────────────────────────────────────────
def velocity(t1, t2):
    return ((t2 - t1) / (t1.abs() + EPS)).clip(-5, 5)

def compute_one_window(df, ref_date, window_offset_days):
    """
    Extracts T1/T2 signals for a single 6-month window.
    window_offset_days: how many days BEFORE ref_date this window starts
      Window 0 → offset=0:   T2=[ref-90d, ref],     T1=[ref-180d, ref-90d]
      Window 1 → offset=180: T2=[ref-270d, ref-180d], T1=[ref-360d, ref-270d]
      Window 2 → offset=360: T2=[ref-450d, ref-360d], T1=[ref-540d, ref-450d]
    """
    t2_end   = ref_date - pd.Timedelta(days=window_offset_days)
    t2_start = t2_end   - pd.Timedelta(days=90)
    t1_end   = t2_start
    t1_start = t1_end   - pd.Timedelta(days=90)

    t1_df = df[(df["date"] >= t1_start) & (df["date"] < t1_end) & (df["tag"] != "LOAN_PAYMENT")]
    t2_df = df[(df["date"] >= t2_start) & (df["date"] < t2_end) & (df["tag"] != "LOAN_PAYMENT")]

    # Need at least 2 rows in each sub-window to compute meaningful signals
    if len(t1_df) < 2 or len(t2_df) < 2:
        return None

    def agg(w):
        sal = w[w["tag"] == "SALARY"]["date"].dt.day.median() if (w["tag"] == "SALARY").any() else np.nan
        return {
            "tx_count":     len(w),
            "total_in":     w["cash_in"].sum(),
            "total_out":    w["cash_out"].sum(),
            "avg_balance":  w["balance"].mean(),
            "min_balance":  w["balance"].min(),
            "overdraft":    (w["balance"] < 0).sum(),
            "salary_day":   sal,
        }

    s1, s2 = agg(t1_df), agg(t2_df)

    row = {
        "income_erosion_v":     velocity(pd.Series([s1["total_in"]]),    pd.Series([s2["total_in"]])).iloc[0],
        "liquidity_momentum_v": velocity(pd.Series([s1["avg_balance"]]), pd.Series([s2["avg_balance"]])).iloc[0],
        "overdraft_v":          s2["overdraft"] - s1["overdraft"],
        "overdraft_t2":         s2["overdraft"],
        "salary_drift_v":       (s2["salary_day"] or 0) - (s1["salary_day"] or 0),
        "tx_freq_v":            velocity(pd.Series([s1["tx_count"]]),     pd.Series([s2["tx_count"]])).iloc[0],
        "avg_balance_t2":       s2["avg_balance"],
        "min_balance_t2":       s2["min_balance"],
        "total_out_t2":         s2["total_out"],
    }
    return row

# ─────────────────────────────────────────────
# STEP 4: Multi-window extraction per account
# ─────────────────────────────────────────────
def compute_features(ledger, labels):
    print("[2/4] Joining loan accounts to ledger...")
    loan_accounts = set(labels["account_id"])
    ledger = ledger[ledger["account_id"].isin(loan_accounts)]

    rows = []
    OFFSETS = list(range(0, 1350, 90))  # 15 windows at 90-day stride (overlapping by 3 months each)

    print(f"[3/4] Extracting up to {len(OFFSETS)} windows per account...")
    for _, loan in labels.iterrows():
        acc_trans = ledger[ledger["account_id"] == loan["account_id"]]
        for w_idx, offset in enumerate(OFFSETS):
            row = compute_one_window(acc_trans, loan["reference_date"], offset)
            if row is not None:
                row["account_id"] = loan["account_id"]
                row["window"]     = w_idx
                row["default"]    = loan["default"]
                rows.append(row)

    feat = pd.DataFrame(rows)
    print(f"[4/4] Feature matrix shape: {feat.shape}")
    print(f"      Total rows:     {len(feat)}")
    print(f"      Default rows:   {feat['default'].sum():.0f}  ({feat['default'].mean()*100:.1f}%)")
    print(f"      Accounts covered: {feat['account_id'].nunique()} / {len(labels)}")
    return feat

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ledger   = load_berka_as_ledger()
    labels   = load_labels()
    features = compute_features(ledger, labels)
    out_path = os.path.join(DATA_DIR, "vector_features_v4.csv")
    features.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(features.drop(columns=["account_id"]).head())
