import os
import logging
from datetime import datetime, timezone

import pandas as pd
from pymongo import MongoClient

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional helper only
    load_dotenv = None


DEFAULT_URI = "mongodb://localhost:27017"
DEFAULT_DB = "realtime_risk_engine"
DEFAULT_CUSTOMERS_COLLECTION = "customers"
DEFAULT_TRANSACTIONS_COLLECTION = "transactions"
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
logger = logging.getLogger("realtime_risk_engine.transactions")


class MongoRiskRepository:
    def __init__(
        self,
        uri: str,
        db_name: str = DEFAULT_DB,
        customers_collection: str = DEFAULT_CUSTOMERS_COLLECTION,
        transactions_collection: str = DEFAULT_TRANSACTIONS_COLLECTION,
    ):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.customers = self.db[customers_collection]
        self.transactions = self.db[transactions_collection]

    @classmethod
    def from_env(cls):
        if load_dotenv is not None:
            load_dotenv(ENV_PATH, override=True)

        uri = os.getenv("MONGODB_URI", DEFAULT_URI)
        db_name = os.getenv("MONGODB_DB_NAME", DEFAULT_DB)
        customers_collection = os.getenv(
            "MONGODB_CUSTOMERS_COLLECTION", DEFAULT_CUSTOMERS_COLLECTION
        )
        transactions_collection = os.getenv(
            "MONGODB_TRANSACTIONS_COLLECTION", DEFAULT_TRANSACTIONS_COLLECTION
        )
        return cls(uri, db_name, customers_collection, transactions_collection)

    def ping(self):
        return self.client.admin.command("ping")

    def ensure_indexes(self):
        self.customers.create_index("account_id", unique=True)
        self.transactions.create_index("account_id", unique=True)

    def upsert_customer(self, account_id: str, profile: dict):
        now = datetime.now(timezone.utc).isoformat()
        self.customers.update_one(
            {"account_id": account_id},
            {
                "$set": {
                    "account_id": account_id,
                    "profile": profile,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

    def save_prediction_result(self, account_id: str, result: dict):
        """
        Persist a timestamped risk calculation only when an explicit prediction
        run happens. Transactions can keep flowing independently of scoring.
        Now also persists Meta-Physics trajectory data.
        """
        calculated_at = result.get("calculated_at") or datetime.now(timezone.utc).isoformat()

        # Extract trajectory fields if present
        trajectory = result.get("trajectory")
        trajectory_snapshot = None
        if trajectory:
            trajectory_snapshot = {
                "zone": trajectory.get("zone"),
                "velocity": trajectory.get("velocity"),
                "acceleration": trajectory.get("acceleration"),
                "r_squared": trajectory.get("r_squared"),
                "confidence": trajectory.get("confidence"),
                "trend": trajectory.get("trend"),
                "days_to_default": trajectory.get("days_to_default"),
                "current_risk": trajectory.get("current_risk"),
                "trajectory_window": trajectory.get("trajectory_window"),
                "analysed_at": trajectory.get("analysed_at"),
            }

        decision_matrix = result.get("decision_matrix")
        decision_snapshot = None
        if decision_matrix:
            decision_snapshot = {
                "zone": decision_matrix.get("zone"),
                "confidence": decision_matrix.get("confidence"),
                "risk_band": decision_matrix.get("risk_band"),
                "intervention_required": decision_matrix.get(
                    "intervention_required"
                ),
                "stress_classification_required": decision_matrix.get(
                    "stress_classification_required"
                ),
                "reason": decision_matrix.get("reason"),
            }

        stress_profile = result.get("stress_profile") or {}

        snapshot = {
            "calculated_at": calculated_at,
            "status": result["status"],
            "final_risk_score": result["final_risk_score"],
            "historian_score": (
                None
                if result["historian"] is None
                else result["historian"]["historian_score"]
            ),
            "behavioral_score": (
                None
                if result["behavioral"] is None
                else result["behavioral"]["behavioral_score"]
            ),
            "stress_type": stress_profile.get("stress_type"),
            "secondary_stress_type": stress_profile.get("secondary_stress_type"),
            "stress_score_share": stress_profile.get("stress_score_share"),
            "stress_confidence_band": stress_profile.get("stress_confidence_band"),
            "trajectory": trajectory_snapshot,
            "decision_matrix": decision_snapshot,
        }
        shap_snapshot = {
            "calculated_at": calculated_at,
            "historian_shap": (
                None
                if result["historian"] is None
                else result["historian"].get("historian_shap")
            ),
            "historian_shap_bias": (
                None
                if result["historian"] is None
                else result["historian"].get("historian_shap_bias")
            ),
            "behavioral_shap": (
                None
                if result["behavioral"] is None
                else result["behavioral"].get("behavioral_shap")
            ),
            "behavioral_shap_bias": (
                None
                if result["behavioral"] is None
                else result["behavioral"].get("behavioral_shap_bias")
            ),
        }

        self.customers.update_one(
            {"account_id": account_id},
            {
                "$set": {
                    "account_id": account_id,
                    "latest_prediction": snapshot,
                    "latest_shap": shap_snapshot,
                    "updated_at": calculated_at,
                },
                "$push": {"risk_history": snapshot},
            },
            upsert=True,
        )

    def save_intervention_report(self, account_id: str, report: dict):
        generated_at = datetime.now(timezone.utc).isoformat()
        snapshot = {
            "generated_at": generated_at,
            "provider_status": report.get("provider_status"),
            "context_packet": report.get("context_packet"),
            "draft": report.get("draft"),
        }
        self.customers.update_one(
            {"account_id": account_id},
            {
                "$set": {
                    "latest_intervention_report": snapshot,
                    "updated_at": generated_at,
                },
                "$push": {"intervention_report_history": snapshot},
            },
            upsert=True,
        )

    def save_email_delivery(self, account_id: str, delivery: dict):
        sent_at = datetime.now(timezone.utc).isoformat()
        snapshot = {
            "sent_at": sent_at,
            "provider": delivery.get("provider"),
            "to": delivery.get("to"),
            "from": delivery.get("from"),
            "subject": delivery.get("subject"),
            "message_id": delivery.get("message_id"),
            "status": delivery.get("status"),
            "status_code": delivery.get("status_code"),
            "response_body": delivery.get("response_body"),
        }
        self.customers.update_one(
            {"account_id": account_id},
            {
                "$set": {
                    "latest_email_delivery": snapshot,
                    "updated_at": sent_at,
                },
                "$push": {"email_delivery_history": snapshot},
            },
            upsert=True,
        )

    def upsert_transactions(self, account_id: str, transactions_df: pd.DataFrame):
        now = datetime.now(timezone.utc).isoformat()
        tx_records = []
        for row in transactions_df.to_dict(orient="records"):
            tx_records.append(
                {
                    "transaction_date": row["Transaction Date"],
                    "description": row["Transaction Description"],
                    "transaction_type": row["Transaction Type"],
                    "credit_amount": float(row["Credit Amount"]),
                    "debit_amount": float(row["Debit Amount"]),
                    "balance": float(row["Balance"]),
                }
            )

        self.transactions.update_one(
            {"account_id": account_id},
            {
                "$set": {
                    "account_id": account_id,
                    "transactions": tx_records,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

    def append_transaction(self, account_id: str, transaction_record: dict):
        now = datetime.now(timezone.utc).isoformat()
        self.transactions.update_one(
            {"account_id": account_id},
            {
                "$set": {
                    "account_id": account_id,
                    "updated_at": now,
                },
                "$push": {"transactions": transaction_record},
            },
            upsert=True,
        )
        logger.info(
            "transaction_appended account_id=%s date=%s type=%s desc=%s credit=%.2f debit=%.2f balance=%.2f",
            account_id,
            transaction_record.get("transaction_date"),
            transaction_record.get("transaction_type"),
            transaction_record.get("description"),
            float(transaction_record.get("credit_amount", 0.0)),
            float(transaction_record.get("debit_amount", 0.0)),
            float(transaction_record.get("balance", 0.0)),
        )

    def get_customer(self, account_id: str):
        return self.customers.find_one({"account_id": account_id}, {"_id": 0})

    def get_transactions(self, account_id: str):
        return self.transactions.find_one({"account_id": account_id}, {"_id": 0})

    def load_transaction_records(self, account_id: str):
        doc = self.get_transactions(account_id)
        if doc is None:
            return []
        return doc.get("transactions", [])

    def load_profile(self, account_id: str):
        doc = self.get_customer(account_id)
        return None if doc is None else doc.get("profile")

    def load_risk_history(self, account_id: str) -> list:
        """Load the accumulated risk_history array for Meta-Physics analysis."""
        doc = self.get_customer(account_id)
        if doc is None:
            return []
        return doc.get("risk_history", [])

    def load_transaction_frame(self, account_id: str):
        doc = self.get_transactions(account_id)
        if doc is None or not doc.get("transactions"):
            return None

        rows = []
        for tx in doc["transactions"]:
            rows.append(
                {
                    "Transaction Date": tx["transaction_date"],
                    "Transaction Description": tx["description"],
                    "Transaction Type": tx["transaction_type"],
                    "Credit Amount": tx["credit_amount"],
                    "Debit Amount": tx["debit_amount"],
                    "Balance": tx["balance"],
                }
            )
        return pd.DataFrame(rows)

    def list_account_ids(self):
        """
        Return the union of account ids present in either collection.
        This supports portfolio-wide scoring without assuming every customer
        has both a profile document and a transaction document at all times.
        """
        customer_ids = set(self.customers.distinct("account_id"))
        transaction_ids = set(self.transactions.distinct("account_id"))
        return sorted(customer_ids | transaction_ids)

    def list_latest_scores(self):
        """
        Fetch stored risk snapshots for every customer document.
        This is a read-only view and does not trigger model inference.
        """
        rows = []
        cursor = self.customers.find(
            {},
            {
                "_id": 0,
                "account_id": 1,
                "latest_prediction": 1,
                "latest_shap": 1,
                "risk_history": 1,
                "updated_at": 1,
            },
        )
        for doc in cursor:
            latest = doc.get("latest_prediction")
            latest_shap = doc.get("latest_shap")
            history = doc.get("risk_history", [])
            timestamps_all = [entry.get("calculated_at") for entry in history if entry.get("calculated_at")]
            score_history = [
                {
                    "calculated_at": entry.get("calculated_at"),
                    "final_risk_score": entry.get("final_risk_score"),
                    "historian_score": entry.get("historian_score"),
                    "behavioral_score": entry.get("behavioral_score"),
                    "stress_type": entry.get("stress_type"),
                    "secondary_stress_type": entry.get("secondary_stress_type"),
                    "stress_score_share": entry.get("stress_score_share"),
                    "stress_confidence_band": entry.get("stress_confidence_band"),
                    "decision_matrix": entry.get("decision_matrix"),
                    "status": entry.get("status"),
                }
                for entry in history
            ]
            rows.append(
                {
                    "account_id": doc.get("account_id"),
                    "updated_at": doc.get("updated_at"),
                    "has_score": latest is not None,
                    "risk_score_timestamp": None
                    if latest is None
                    else latest.get("calculated_at"),
                    "final_risk_score": None
                    if latest is None
                    else latest.get("final_risk_score"),
                    "prediction_count": len(history),
                    "risk_score_timestamps_all": timestamps_all,
                    "risk_score_history": score_history,
                    "latest_prediction": latest,
                    "latest_shap": latest_shap,
                }
            )
        rows.sort(key=lambda x: x["account_id"] or "")
        return rows
