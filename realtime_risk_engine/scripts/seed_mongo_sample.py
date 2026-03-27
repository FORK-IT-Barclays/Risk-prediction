import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.demo_data import build_random_profile, build_random_transactions
from src.mongo_store import MongoRiskRepository


def main():
    account_prefix = os.getenv("SAMPLE_ACCOUNT_PREFIX", "CUST")
    start_index = int(os.getenv("SAMPLE_ACCOUNT_START", "1"))
    customer_count = int(os.getenv("SAMPLE_CUSTOMER_COUNT", "10"))
    rng = random.Random()

    repo = MongoRiskRepository.from_env()
    repo.ping()
    repo.ensure_indexes()

    print(f"Seeding {customer_count} sample customers into MongoDB...")

    for offset in range(customer_count):
        account_id = f"{account_prefix}_{start_index + offset:03d}"
        profile = build_random_profile(rng)
        transactions = build_random_transactions(rng, profile)

        repo.upsert_customer(account_id, profile)
        repo.upsert_transactions(account_id, transactions)

        print(
            f"- {account_id}: stored {len(transactions)} transactions, "
            f"annual_inc={profile['annual_inc']:.2f}, "
            f"loan_amnt={profile['loan_amnt']:.2f}"
        )

    print("Sample portfolio seeding complete.")


if __name__ == "__main__":
    main()
