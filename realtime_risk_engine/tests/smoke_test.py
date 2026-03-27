import os
import sys
import random

import pandas as pd

# Ensure the parent directory is in the path so we can import our new modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.demo_data import build_random_profile, build_random_transactions
from src.inference import RiskEngine


def run_smoke_test():
    print("Initializing dual-model risk engine smoke test...")

    try:
        engine = RiskEngine()
        print("Loaded behavioral and historian artifacts successfully.")
    except Exception as e:
        print(f"Failed to load engine: {e}")
        return

    rng = random.Random()
    profile = build_random_profile(rng)
    mock_data = build_random_transactions(rng, profile)

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
    print(
        "Profile:"
        f" annual_inc={profile['annual_inc']:.2f},"
        f" loan_amnt={profile['loan_amnt']:.2f},"
        f" dti={profile['dti']:.2f},"
        f" installment={profile['installment']:.2f}"
    )
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
