import os
import sys

import pandas as pd

# Ensure the parent directory is in the path so we can import our new modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import RiskEngine


def run_smoke_test():
    print("Initializing dual-model risk engine smoke test...")

    try:
        engine = RiskEngine()
        print("Loaded behavioral and historian artifacts successfully.")
    except Exception as e:
        print(f"Failed to load engine: {e}")
        return

    profile = {
        "annual_inc": 72000.0,
        "loan_amnt": 18000.0,
        "dti": 16.5,
        "term_months": 36,
        "open_acc": 8,
        "total_acc": 19,
        "revol_bal": 8200.0,
        "revol_util": 41.0,
        "delinq_2yrs": 0,
        "pub_rec": 0,
        "inq_last_6mths": 1,
        "installment": 610.0,
    }

    dates = pd.date_range(end=pd.Timestamp.now(), periods=50, freq="4D")
    mock_data = pd.DataFrame(
        {
            "Transaction Date": dates.strftime("%d/%m/%Y"),
            "Transaction Description": (
                ["Monthly Salary", "O2 Mobile Bill", "Grocery Store"] * 16
                + ["Salary", "Rent"]
            ),
            "Transaction Type": ["BGC", "DD", "POS"] * 16 + ["FPI", "DD"],
            "Credit Amount": [2500.0, 0.0, 0.0] * 16 + [2500.0, 0.0],
            "Debit Amount": [0.0, 45.0, 30.0] * 16 + [0.0, 850.0],
            "Balance": [2500.0, 2455.0, 2425.0] * 16 + [4925.0, 4075.0],
        }
    )

    print(f"Generated {len(mock_data)} mock UK transactions.")
    print("Running unified risk prediction...")
    results = engine.predict_risk(raw_tx_df=mock_data, profile=profile)

    if results["status"] != "OK":
        print(f"Prediction failed: {results['status']}")
        return

    print("\n" + "=" * 40)
    print("DUAL-MODEL RISK RESULT")
    print("=" * 40)
    print(f"Account ID: {results['account_id']}")
    print(f"Historian:  {results['historian']['historian_score'] * 100:.2f}%")
    print(f"Behavioral: {results['behavioral']['behavioral_score'] * 100:.2f}%")
    print(f"Final:      {results['final_risk_score'] * 100:.2f}%")
    print("-" * 40)
    print("BEHAVIORAL SIGNALS (Top 3):")
    for key in ["liquidity_momentum_v", "income_erosion_v", "overdraft_v"]:
        print(f"- {key}: {results['behavioral']['signals'][key]:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    run_smoke_test()
