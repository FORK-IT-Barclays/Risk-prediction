import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import RiskEngine
from src.mongo_store import MongoRiskRepository
from src.portfolio_scoring import score_all_customers


def main():
    repo = MongoRiskRepository.from_env()
    repo.ping()
    engine = RiskEngine()
    account_ids = repo.list_account_ids()

    if not account_ids:
        print("No customers or transactions found in MongoDB.")
        return

    print(f"Scoring {len(account_ids)} customers from MongoDB...")
    print("-" * 72)

    for result in score_all_customers(repo, engine):
        account_id = result["account_id"]
        historian_score = (
            None
            if result["historian"] is None
            else result["historian"]["historian_score"]
        )
        behavioral_score = (
            None
            if result["behavioral"] is None
            else result["behavioral"]["behavioral_score"]
        )

        print(f"Account ID: {account_id}")
        print(f"Status: {result['status']}")
        print(f"Final Risk Score: {result['final_risk_score']}")
        print(f"Historian Score: {historian_score}")
        print(f"Behavioral Score: {behavioral_score}")
        latest_prediction = repo.get_customer(account_id).get("latest_prediction", {})
        print(f"Calculated At: {latest_prediction.get('calculated_at')}")
        print("-" * 72)


if __name__ == "__main__":
    main()
