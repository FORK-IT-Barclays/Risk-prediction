import os
import sys
import random
from datetime import datetime, timezone

# Add the project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from realtime_risk_engine.src.mongo_store import MongoRiskRepository
from realtime_risk_engine.src.demo_data import build_random_transactions

INITIAL_CUSTOMERS = [
    {
        "account_id": "CUST_001",
        "profile": {
            "annual_inc": 52000.0,
            "loan_amnt": 12000.0,
            "dti": 14.2,
            "term_months": 36,
            "open_acc": 5,
            "total_acc": 13,
            "revol_bal": 6100.0,
            "revol_util": 35.4,
            "delinq_2yrs": 0,
            "pub_rec": 0,
            "inq_last_6mths": 1,
            "installment": 420.0
        }
    },
    {
        "account_id": "CUST_002",
        "profile": {
            "annual_inc": 68000.0,
            "loan_amnt": 18000.0,
            "dti": 19.1,
            "term_months": 48,
            "open_acc": 7,
            "total_acc": 19,
            "revol_bal": 9200.0,
            "revol_util": 42.8,
            "delinq_2yrs": 1,
            "pub_rec": 0,
            "inq_last_6mths": 2,
            "installment": 545.0
        }
    },
    {
        "account_id": "CUST_003",
        "profile": {
            "annual_inc": 44000.0,
            "loan_amnt": 9000.0,
            "dti": 16.5,
            "term_months": 24,
            "open_acc": 4,
            "total_acc": 11,
            "revol_bal": 3500.0,
            "revol_util": 28.0,
            "delinq_2yrs": 0,
            "pub_rec": 0,
            "inq_last_6mths": 1,
            "installment": 470.0
        }
    },
    {
        "account_id": "CUST_004",
        "profile": {
            "annual_inc": 91000.0,
            "loan_amnt": 25000.0,
            "dti": 21.3,
            "term_months": 60,
            "open_acc": 10,
            "total_acc": 24,
            "revol_bal": 14800.0,
            "revol_util": 51.6,
            "delinq_2yrs": 1,
            "pub_rec": 0,
            "inq_last_6mths": 3,
            "installment": 715.0
        }
    },
    {
        "account_id": "CUST_005",
        "profile": {
            "annual_inc": 39000.0,
            "loan_amnt": 7000.0,
            "dti": 11.8,
            "term_months": 36,
            "open_acc": 3,
            "total_acc": 9,
            "revol_bal": 2400.0,
            "revol_util": 24.5,
            "delinq_2yrs": 0,
            "pub_rec": 0,
            "inq_last_6mths": 0,
            "installment": 315.0
        }
    },
    {
        "account_id": "CUST_006",
        "profile": {
            "annual_inc": 74000.0,
            "loan_amnt": 20000.0,
            "dti": 23.7,
            "term_months": 60,
            "open_acc": 8,
            "total_acc": 20,
            "revol_bal": 13200.0,
            "revol_util": 58.2,
            "delinq_2yrs": 2,
            "pub_rec": 1,
            "inq_last_6mths": 4,
            "installment": 690.0
        }
    },
    {
        "account_id": "CUST_007",
        "profile": {
            "annual_inc": 58000.0,
            "loan_amnt": 15000.0,
            "dti": 17.6,
            "term_months": 48,
            "open_acc": 6,
            "total_acc": 15,
            "revol_bal": 7600.0,
            "revol_util": 37.0,
            "delinq_2yrs": 0,
            "pub_rec": 0,
            "inq_last_6mths": 2,
            "installment": 495.0
        }
    },
    {
        "account_id": "CUST_008",
        "profile": {
            "annual_inc": 102000.0,
            "loan_amnt": 30000.0,
            "dti": 26.4,
            "term_months": 60,
            "open_acc": 12,
            "total_acc": 28,
            "revol_bal": 18100.0,
            "revol_util": 62.4,
            "delinq_2yrs": 1,
            "pub_rec": 0,
            "inq_last_6mths": 3,
            "installment": 860.0
        }
    },
    {
        "account_id": "CUST_009",
        "profile": {
            "annual_inc": 47000.0,
            "loan_amnt": 11000.0,
            "dti": 13.9,
            "term_months": 36,
            "open_acc": 5,
            "total_acc": 12,
            "revol_bal": 4900.0,
            "revol_util": 33.8,
            "delinq_2yrs": 0,
            "pub_rec": 0,
            "inq_last_6mths": 1,
            "installment": 398.0
        }
    },
    {
        "account_id": "CUST_010",
        "profile": {
            "annual_inc": 83000.0,
            "loan_amnt": 22000.0,
            "dti": 20.8,
            "term_months": 48,
            "open_acc": 9,
            "total_acc": 22,
            "revol_bal": 11600.0,
            "revol_util": 46.7,
            "delinq_2yrs": 1,
            "pub_rec": 0,
            "inq_last_6mths": 2,
            "installment": 640.0
        }
    }
]

def main():
    rng = random.Random()
    repo = MongoRiskRepository.from_env()
    try:
        repo.ping()
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return

    repo.ensure_indexes()
    print(f"Seeding {len(INITIAL_CUSTOMERS)} customers with history into collection '{repo.customers.name}'...")

    for cust in INITIAL_CUSTOMERS:
        account_id = cust["account_id"]
        profile = cust["profile"]
        
        # 1. Upsert Customer Profile
        repo.upsert_customer(account_id, profile)
        
        # 2. Generate and Store 180 Days of Historical Transactions
        history_df = build_random_transactions(rng, profile)
        repo.upsert_transactions(account_id, history_df)
        
        print(f" - Seeded {account_id} with {len(history_df)} historical transactions.")

    print("\nInitial customer and history seeding complete.")

if __name__ == "__main__":
    main()
