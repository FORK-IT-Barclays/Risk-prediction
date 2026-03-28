from fastapi import APIRouter
from realtime_risk_engine.src.mongo_store import MongoRiskRepository
from .meta_physics import MetaPhysicsEngine
from typing import List, Dict, Any

router = APIRouter()

@router.get("/physics_engine")
def get_physics_analysis():
    """
    Independent trajectory analysis endpoint.
    Retrieves risk history from MongoDB and computes Layer 3 Physics.
    """
    repo = MongoRiskRepository.from_env()
    engine = MetaPhysicsEngine()
    
    account_ids = repo.list_account_ids()
    physics_reports = []

    for account_id in account_ids:
        # Pull history (as calculated by /risk-score or /all_scores)
        history = repo.load_risk_history(account_id)
        
        if not history or len(history) < 2:
            physics_reports.append({
                "account_id": account_id,
                "status": "INSUFFICIENT_HISTORY",
                "trajectory": None
            })
            continue

        analysis = engine.analyze(history)
        physics_reports.append({
            "account_id": account_id,
            "status": "OK",
            "current_risk": analysis.get("current_risk"),
            "velocity": analysis.get("velocity"),
            "acceleration": analysis.get("acceleration"),
            "zone": analysis.get("zone"),
            "days_to_default": analysis.get("days_to_default"),
            "confidence": analysis.get("confidence"),
            "last_updated": analysis.get("analysed_at")
        })

    return {
        "status": "ok",
        "total_accounts": len(physics_reports),
        "reports": physics_reports
    }
