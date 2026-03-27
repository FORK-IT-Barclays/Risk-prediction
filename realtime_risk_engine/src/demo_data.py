import random

import pandas as pd


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


def build_next_transaction(rng: random.Random, profile: dict, last_transaction: dict | None):
    """
    Generate the next incoming demo transaction for an existing customer.
    """
    if last_transaction is None:
        next_date = pd.Timestamp.now().normalize()
        last_balance = round(rng.uniform(800.0, 6000.0), 2)
    else:
        next_date = pd.to_datetime(last_transaction["transaction_date"], dayfirst=True) + pd.Timedelta(days=1)
        last_balance = float(last_transaction["balance"])

    salary_base = round(profile["annual_inc"] / 12.0, 2)
    cycle = rng.choice(["salary", "bill", "spend"])
    credit = 0.0
    debit = 0.0

    if cycle == "salary":
        description = rng.choice(["Monthly Salary", "Salary Credit", "Employer BGC"])
        tx_type = rng.choice(["BGC", "FPI"])
        credit = round(salary_base * rng.uniform(0.82, 1.18), 2)
    elif cycle == "bill":
        description, tx_type = rng.choice(
            [
                ("O2 Mobile Bill", "DD"),
                ("Octopus Energy", "DD"),
                ("Virgin Media", "DD"),
                ("Rent Payment", "DD"),
            ]
        )
        debit = round(rng.uniform(40.0, 1400.0), 2)
    else:
        description, tx_type = rng.choice(
            [
                ("Grocery Store", "POS"),
                ("ATM Withdrawal", "ATM"),
                ("Coffee Shop", "POS"),
                ("Restaurant", "POS"),
                ("Fuel Station", "POS"),
            ]
        )
        debit = round(rng.uniform(15.0, 420.0), 2)

    balance = round(last_balance + credit - debit + rng.uniform(-35.0, 35.0), 2)
    return {
        "transaction_date": next_date.strftime("%d/%m/%Y"),
        "description": description,
        "transaction_type": tx_type,
        "credit_amount": credit,
        "debit_amount": debit,
        "balance": balance,
    }
