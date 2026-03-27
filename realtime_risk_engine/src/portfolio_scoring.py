from .inference import RiskEngine
from .mongo_store import MongoRiskRepository


def score_all_customers(repo: MongoRiskRepository, engine: RiskEngine):
    """
    Trigger a portfolio-wide prediction pass and persist timestamped results
    into each customer document.
    """
    account_ids = repo.list_account_ids()
    results = []

    for account_id in account_ids:
        profile = repo.load_profile(account_id)
        transactions = repo.load_transaction_frame(account_id)
        result = engine.predict_risk(
            raw_tx_df=transactions,
            profile=profile,
            account_id=account_id,
        )
        repo.save_prediction_result(account_id, result)
        results.append(result)

    return results
