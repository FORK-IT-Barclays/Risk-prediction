from .inference import RiskEngine
from .intervention.service import run_automatic_intervention
from .mongo_store import MongoRiskRepository


def score_all_customers(repo: MongoRiskRepository, engine: RiskEngine):
    """
    Trigger a portfolio-wide prediction pass and persist timestamped results
    into each customer document.

    The risk_history is loaded from MongoDB and passed into predict_risk()
    so the Meta-Physics Engine can compute trajectory analysis inline.
    """
    account_ids = repo.list_account_ids()
    results = []

    for account_id in account_ids:
        profile = repo.load_profile(account_id)
        transactions = repo.load_transaction_frame(account_id)

        # Load accumulated risk history for Meta-Physics pipeline stage
        risk_history = repo.load_risk_history(account_id)

        result = engine.predict_risk(
            raw_tx_df=transactions,
            profile=profile,
            account_id=account_id,
            risk_history=risk_history,
        )
        repo.save_prediction_result(account_id, result)

        customer_doc = repo.get_customer(account_id) or {}
        auto_intervention = run_automatic_intervention(account_id, customer_doc)
        result["auto_intervention"] = auto_intervention
        if auto_intervention.get("report") is not None:
            repo.save_intervention_report(account_id, auto_intervention["report"])
        if auto_intervention.get("email_delivery") is not None:
            repo.save_email_delivery(account_id, auto_intervention["email_delivery"])

        results.append(result)

    return results
