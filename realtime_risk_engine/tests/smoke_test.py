import os
import sys

import pandas as pd
import random

# Ensure the parent directory is in the path so we can import our new modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import RiskEngine


def build_random_profile(rng: random.Random):
    annual_inc = round(rng.uniform(28000.0, 140000.0), 2)
    loan_amnt = round(rng.uniform(3000.0, min(50000.0, annual_inc * 0.75)), 2)
    term_months = rng.choice([12, 24, 36, 48, 60])
    open_acc = rng.randint(1, 14)
    total_acc = open_acc + rng.randint(3, 20)
    revol_bal = round(rng.uniform(0.0, 25000.0), 2)
    revol_util = round(rng.uniform(0.0, 95.0), 2)
    delinq_2yrs = rng.randint(0, 3)
    pub_rec = rng.randint(0, 2)
    inq_last_6mths = rng.randint(0, 5)
    dti = round(rng.uniform(4.0, 42.0), 2)
    installment = round(loan_amnt / max(term_months, 1) + rng.uniform(20.0, 260.0), 2)

    return {
        "annual_inc": annual_inc,
        "loan_amnt": loan_amnt,
        "dti": dti,
        "term_months": term_months,
        "open_acc": open_acc,
        "total_acc": total_acc,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "delinq_2yrs": delinq_2yrs,
        "pub_rec": pub_rec,
        "inq_last_6mths": inq_last_6mths,
        "installment": installment,
    }


def build_random_transactions(rng: random.Random, profile: dict) -> pd.DataFrame:
    end_date = pd.Timestamp.now().normalize()
    dates = pd.date_range(end=end_date, periods=60, freq="3D")

    salary_base = round(profile["annual_inc"] / 12.0, 2)
    balance = round(rng.uniform(800.0, 6000.0), 2)
    rows = []

    bill_choices = [
        ("O2 Mobile Bill", "DD"),
        ("Octopus Energy", "DD"),
        ("Virgin Media", "DD"),
        ("Rent Payment", "DD"),
    ]
    spend_choices = [
        ("Grocery Store", "POS"),
        ("ATM Withdrawal", "ATM"),
        ("Coffee Shop", "POS"),
        ("Restaurant", "POS"),
        ("Fuel Station", "POS"),
    ]

    for idx, date in enumerate(dates):
        cycle = idx % 3
        description = ""
        tx_type = ""
        credit = 0.0
        debit = 0.0

        if cycle == 0:
            description = rng.choice(["Monthly Salary", "Salary Credit", "Employer BGC"])
            tx_type = rng.choice(["BGC", "FPI"])
            credit = round(salary_base * rng.uniform(0.82, 1.18), 2)
        elif cycle == 1:
            description, tx_type = rng.choice(bill_choices)
            debit = round(rng.uniform(40.0, 1400.0), 2)
        else:
            description, tx_type = rng.choice(spend_choices)
            debit = round(rng.uniform(15.0, 420.0), 2)

        balance = round(balance + credit - debit + rng.uniform(-35.0, 35.0), 2)
        rows.append(
            {
                "Transaction Date": date.strftime("%d/%m/%Y"),
                "Transaction Description": description,
                "Transaction Type": tx_type,
                "Credit Amount": credit,
                "Debit Amount": debit,
                "Balance": balance,
            }
        )

    return pd.DataFrame(rows)


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
