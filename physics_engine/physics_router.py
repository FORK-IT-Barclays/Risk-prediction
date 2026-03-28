from fastapi import APIRouter
from realtime_risk_engine.src.mongo_store import MongoRiskRepository

router = APIRouter()

@router.get("/physics_engine")
def get_physics_analysis():
    """
    Read-only view of the latest stored trajectory decisions.
    The authoritative physics calculation happens inside /risk-score.
    """
    repo = MongoRiskRepository.from_env()
    physics_reports = []

    for row in repo.list_latest_scores():
        latest = row.get("latest_prediction") or {}
        trajectory = latest.get("trajectory")
        decision = latest.get("decision_matrix")

        if not trajectory:
            physics_reports.append({
                "account_id": row.get("account_id"),
                "status": "INSUFFICIENT_HISTORY",
                "trajectory": None,
                "decision_matrix": decision,
            })
            continue

        physics_reports.append({
            "account_id": row.get("account_id"),
            "status": "OK",
            "current_risk": trajectory.get("current_risk"),
            "velocity": trajectory.get("velocity"),
            "acceleration": trajectory.get("acceleration"),
            "zone": trajectory.get("zone"),
            "days_to_default": trajectory.get("days_to_default"),
            "confidence": trajectory.get("confidence"),
            "decision_matrix": decision,
            "last_updated": trajectory.get("analysed_at"),
        })

    return {
        "status": "ok",
        "total_accounts": len(physics_reports),
        "reports": physics_reports
    }
