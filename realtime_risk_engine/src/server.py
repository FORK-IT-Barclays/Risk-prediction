from fastapi import FastAPI

from .demo_stream import DemoStreamController
from .inference import RiskEngine
from .mongo_store import MongoRiskRepository
from .portfolio_scoring import score_all_customers

app = FastAPI(title="Realtime Risk Engine")
demo_controller = DemoStreamController()


@app.get("/demo")
def start_demo():
    started = demo_controller.start()
    stats = demo_controller.get_stats()
    return {
        "status": "started" if started else "already_running",
        "demo_running": stats["running"],
        "customer_count": stats["account_count"],
        "interval_seconds": stats["interval_seconds"],
        "estimated_transactions_per_second": stats["estimated_transactions_per_second"],
        "total_generated": stats["total_generated"],
        "message": "Transactions are now flowing. Trigger /risk-score to calculate and store risk snapshots.",
    }


@app.get("/risk-score")
def trigger_risk_score():
    repo = MongoRiskRepository.from_env()
    repo.ping()
    engine = RiskEngine()
    results = score_all_customers(repo, engine)

    return {
        "status": "ok",
        "customer_count": len(results),
        "results": [
            {
                "account_id": result["account_id"],
                "status": result["status"],
                "final_risk_score": result["final_risk_score"],
                "historian_score": None
                if result["historian"] is None
                else result["historian"]["historian_score"],
                "behavioral_score": None
                if result["behavioral"] is None
                else result["behavioral"]["behavioral_score"],
                "current_shap": {
                    "historian_shap": None
                    if result["historian"] is None
                    else result["historian"].get("historian_shap"),
                    "historian_shap_bias": None
                    if result["historian"] is None
                    else result["historian"].get("historian_shap_bias"),
                    "behavioral_shap": None
                    if result["behavioral"] is None
                    else result["behavioral"].get("behavioral_shap"),
                    "behavioral_shap_bias": None
                    if result["behavioral"] is None
                    else result["behavioral"].get("behavioral_shap_bias"),
                },
            }
            for result in results
        ],
    }


@app.get("/all_scores")
def all_scores():
    """
    Return the latest stored risk snapshot for all customers.
    This endpoint is read-only and does not run inference.
    """
    repo = MongoRiskRepository.from_env()
    repo.ping()
    rows = repo.list_latest_scores()
    missing_count = sum(1 for row in rows if not row["has_score"])
    return {
        "status": "ok",
        "customer_count": len(rows),
        "scored_count": len(rows) - missing_count,
        "missing_count": missing_count,
        "results": rows,
    }


@app.post("/stop-demo")
def stop_demo():
    stopped = demo_controller.stop()
    stats = demo_controller.get_stats()
    return {
        "status": "stopped" if stopped else "already_stopped",
        "demo_running": stats["running"],
        "interval_seconds": stats["interval_seconds"],
        "estimated_transactions_per_second": stats["estimated_transactions_per_second"],
        "last_cycle_generated": stats["last_cycle_generated"],
        "total_generated": stats["total_generated"],
        "last_cycle_at": stats["last_cycle_at"],
    }


@app.get("/demo-stats")
def demo_stats():
    """Inspect live transaction generation stats while the demo stream runs."""
    return demo_controller.get_stats()
