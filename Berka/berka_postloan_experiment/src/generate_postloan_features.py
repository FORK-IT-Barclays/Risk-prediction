from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT.parent / "dataset"
OUTPUT_DIR = ROOT / "outputs"

DATASET_CUTOFF = pd.Timestamp("1998-12-31")
WINDOW_DAYS = 180
HALF_WINDOW_DAYS = 90
STRIDE_DAYS = 30
MIN_TX_PER_HALF = 2
EPS = 1e-6

FEATURE_COLUMNS = [
    "income_erosion_v",
    "liquidity_momentum_v",
    "overdraft_v",
    "overdraft_t2",
    "salary_drift_v",
    "tx_freq_v",
    "avg_balance_t2",
    "min_balance_t2",
    "total_out_t2",
]


def velocity(value_t1: float, value_t2: float) -> float:
    return float(np.clip((value_t2 - value_t1) / (abs(value_t1) + EPS), -5, 5))


def load_ledger() -> pd.DataFrame:
    trans = pd.read_csv(DATASET_DIR / "trans.csv", sep=";", low_memory=False)

    ledger = pd.DataFrame(
        {
            "account_id": trans["account_id"],
            "date": pd.to_datetime(trans["date"].astype(str), format="%y%m%d"),
            "cash_in": np.where(trans["type"] == "PRIJEM", trans["amount"], 0.0),
            "cash_out": np.where(trans["type"].isin(["VYDAJ", "VYBER"]), trans["amount"], 0.0),
            "balance": trans["balance"].astype(float),
        }
    )

    k_map = {
        "SIPO": "BILL",
        "UVER": "LOAN_PAYMENT",
        "DUCHOD": "SALARY",
        "POJISTNE": "INSURANCE",
    }
    op_map = {
        "VKLAD": "DEPOSIT",
        "VYBER": "ATM_CASH",
        "PREVOD Z UCTU": "TRANSFER",
        "PREVOD NA UCET": "TRANSFER",
    }

    tag = trans["k_symbol"].map(k_map)
    if "operation" in trans.columns:
        tag = tag.fillna(trans["operation"].map(op_map))
    ledger["tag"] = tag.fillna("UNCATEGORIZED")

    return ledger.sort_values(["account_id", "date"]).reset_index(drop=True)


def load_loans() -> pd.DataFrame:
    loans = pd.read_csv(DATASET_DIR / "loan.csv", sep=";")
    loans["loan_date"] = pd.to_datetime(loans["date"].astype(str), format="%y%m%d")
    loans["default"] = loans["status"].map({"A": 0, "C": 0, "B": 1, "D": 1})
    loans["term_end_date"] = loans["loan_date"] + pd.to_timedelta(loans["duration"] * 30, unit="D")
    loans["term_end_date"] = loans["term_end_date"].clip(upper=DATASET_CUTOFF)
    loans["usable_days"] = (loans["term_end_date"] - loans["loan_date"]).dt.days

    return loans[
        [
            "account_id",
            "loan_date",
            "term_end_date",
            "duration",
            "status",
            "default",
            "usable_days",
        ]
    ].copy()


def salary_day(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.dt.day.median())


def aggregate_window(window_df: pd.DataFrame) -> dict[str, float]:
    return {
        "tx_count": float(len(window_df)),
        "total_in": float(window_df["cash_in"].sum()),
        "total_out": float(window_df["cash_out"].sum()),
        "avg_balance": float(window_df["balance"].mean()),
        "min_balance": float(window_df["balance"].min()),
        "overdraft_count": float((window_df["balance"] < 0).sum()),
        "salary_day": salary_day(window_df.loc[window_df["tag"] == "SALARY", "date"]),
    }


def compute_postloan_window(
    account_tx: pd.DataFrame,
    loan_date: pd.Timestamp,
    ref_date: pd.Timestamp,
) -> dict[str, float] | None:
    t1_start = ref_date - pd.Timedelta(days=WINDOW_DAYS)
    t1_end = ref_date - pd.Timedelta(days=HALF_WINDOW_DAYS)
    t2_start = t1_end
    t2_end = ref_date

    t1_df = account_tx[
        (account_tx["date"] >= t1_start)
        & (account_tx["date"] < t1_end)
        & (account_tx["tag"] != "LOAN_PAYMENT")
    ]
    t2_df = account_tx[
        (account_tx["date"] >= t2_start)
        & (account_tx["date"] < t2_end)
        & (account_tx["tag"] != "LOAN_PAYMENT")
    ]

    if len(t1_df) < MIN_TX_PER_HALF or len(t2_df) < MIN_TX_PER_HALF:
        return None

    s1 = aggregate_window(t1_df)
    s2 = aggregate_window(t2_df)

    if np.isnan(s1["salary_day"]) or np.isnan(s2["salary_day"]):
        salary_drift = 0.0
    else:
        salary_drift = float(s2["salary_day"] - s1["salary_day"])

    return {
        "income_erosion_v": velocity(s1["total_in"], s2["total_in"]),
        "liquidity_momentum_v": velocity(s1["avg_balance"], s2["avg_balance"]),
        "overdraft_v": float(np.clip(s2["overdraft_count"] - s1["overdraft_count"], -5, 5)),
        "overdraft_t2": s2["overdraft_count"],
        "salary_drift_v": salary_drift,
        "tx_freq_v": velocity(s1["tx_count"], s2["tx_count"]),
        "avg_balance_t2": s2["avg_balance"],
        "min_balance_t2": s2["min_balance"],
        "total_out_t2": s2["total_out"],
        "t1_tx_count": int(len(t1_df)),
        "t2_tx_count": int(len(t2_df)),
        "days_since_loan": int((ref_date - loan_date).days),
    }


def build_postloan_features(ledger: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    loan_accounts = set(loans["account_id"])
    ledger = ledger[ledger["account_id"].isin(loan_accounts)].copy()

    grouped_tx = {
        account_id: frame.sort_values("date").reset_index(drop=True)
        for account_id, frame in ledger.groupby("account_id", sort=False)
    }

    rows: list[dict[str, float]] = []

    for loan in loans.itertuples(index=False):
        account_tx = grouped_tx.get(loan.account_id)
        if account_tx is None:
            continue

        account_tx = account_tx[
            (account_tx["date"] >= loan.loan_date) & (account_tx["date"] <= loan.term_end_date)
        ]
        if account_tx.empty:
            continue

        ref_date = loan.loan_date + pd.Timedelta(days=WINDOW_DAYS)
        window_index = 0

        while ref_date <= loan.term_end_date:
            feature_row = compute_postloan_window(account_tx, loan.loan_date, ref_date)
            if feature_row is not None:
                feature_row.update(
                    {
                        "account_id": int(loan.account_id),
                        "window_index": window_index,
                        "window_end_date": ref_date.date().isoformat(),
                        "duration_months": int(loan.duration),
                        "status": loan.status,
                        "default": int(loan.default),
                    }
                )
                rows.append(feature_row)

            ref_date += pd.Timedelta(days=STRIDE_DAYS)
            window_index += 1

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Berka ledger and loan tables...")
    ledger = load_ledger()
    loans = load_loans()

    print(f"Loan rows: {len(loans):,}")
    print(f"Loan accounts: {loans['account_id'].nunique():,}")

    features = build_postloan_features(ledger, loans)
    if features.empty:
        raise RuntimeError("No post-loan windows were generated. Check the filtering logic.")

    features["salary_drift_v"] = features["salary_drift_v"].fillna(0.0)

    out_path = OUTPUT_DIR / "postloan_vector_features.csv"
    features.to_csv(out_path, index=False)

    covered_accounts = features["account_id"].nunique()
    default_rate = features["default"].mean() * 100

    print(f"Saved: {out_path}")
    print(f"Rows: {len(features):,}")
    print(f"Accounts covered: {covered_accounts:,} / {loans['account_id'].nunique():,}")
    print(f"Default window rate: {default_rate:.2f}%")
    print("Feature columns:")
    for feature_name in FEATURE_COLUMNS:
        print(f"  - {feature_name}")


if __name__ == "__main__":
    main()
