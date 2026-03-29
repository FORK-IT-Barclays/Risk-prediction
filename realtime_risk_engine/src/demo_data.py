import os
import random

import pandas as pd


def get_reference_today() -> pd.Timestamp:
    configured = os.getenv("DEMO_REFERENCE_DATE")
    if configured:
        try:
            return pd.Timestamp(configured).normalize()
        except ValueError:
            pass
    return pd.Timestamp.now().normalize()


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


def build_random_transactions(
    rng: random.Random,
    profile: dict,
    end_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    # Seed a clear six-month history that ends yesterday so the initial
    # portfolio looks historical rather than "generated today".
    if end_date is None:
        end_date = get_reference_today() - pd.Timedelta(days=1)
    else:
        end_date = pd.Timestamp(end_date).normalize()

    dates = pd.date_range(end=end_date, periods=180, freq="D")

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


def _scenario_for_account(account_id: str | None) -> str:
    try:
        numeric = int(str(account_id or "").split("_")[-1])
    except ValueError:
        numeric = 0

    bucket = numeric % 5
    if bucket in (0, 1):
        return "worsening"
    if bucket == 2:
        return "recovering"
    return "stable"


def build_next_transaction(
    rng: random.Random,
    profile: dict,
    last_transaction: dict | None,
    account_id: str | None = None,
    tx_index: int = 0,
):
    """
    Generate the next incoming demo transaction for an existing customer.
    """
    if last_transaction is None:
        next_date = get_reference_today()
        last_balance = round(rng.uniform(800.0, 6000.0), 2)
    else:
        last_date = pd.to_datetime(last_transaction["transaction_date"], dayfirst=True).normalize()
        today = get_reference_today()
        # Keep the live demo anchored to the current calendar day instead of
        # letting repeated clicks push the stream into unrealistic future dates.
        if last_date >= today:
            next_date = today
        else:
            next_date = last_date + pd.Timedelta(days=1)
        last_balance = float(last_transaction["balance"])

    salary_base = round(profile["annual_inc"] / 12.0, 2)
    cycle = ["salary", "bill", "spend"][tx_index % 3]
    credit = 0.0
    debit = 0.0
    scenario = _scenario_for_account(account_id)
    demo_step = max(tx_index - 180, 0)
    phase = min(demo_step / 12.0, 5.0)

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

    if scenario == "worsening":
        if cycle == "salary":
            credit = round(credit * max(0.35, 0.82 - (0.08 * phase)), 2)
        else:
            debit = round(debit * (1.35 + (0.18 * phase)), 2)
    elif scenario == "recovering":
        if cycle == "salary":
            credit = round(credit * (1.08 + (0.04 * phase)), 2)
        else:
            debit = round(debit * max(0.45, 0.78 - (0.04 * phase)), 2)

    noise = rng.uniform(-35.0, 35.0)
    if scenario == "worsening":
        noise -= (20 + 10 * phase)
    elif scenario == "recovering":
        noise += (10 + 6 * phase)

    balance = round(last_balance + credit - debit + noise, 2)
    return {
        "transaction_date": next_date.strftime("%d/%m/%Y"),
        "description": description,
        "transaction_type": tx_type,
        "credit_amount": credit,
        "debit_amount": debit,
        "balance": balance,
    }
