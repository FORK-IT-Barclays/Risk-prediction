import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mongo_store import MongoRiskRepository


def main():
    repo = MongoRiskRepository.from_env()
    repo.ping()
    repo.ensure_indexes()
    print("MongoDB connection successful.")
    print("Indexes ensured for customers.account_id and transactions.account_id.")


if __name__ == "__main__":
    main()
