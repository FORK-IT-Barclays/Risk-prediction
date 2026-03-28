from __future__ import annotations

from typing import Any, Dict, List


ZONE_MESSAGE_STRATEGY = {
    "SAFE_RECOVERING": {
        "tone": "praise_reinforcement",
        "message_goal": "acknowledge improvement and reinforce positive momentum",
        "recommended_channel": "dashboard",
    },
    "SAFE_STABLE": {
        "tone": "informational",
        "message_goal": "monitor silently without outreach",
        "recommended_channel": "dashboard",
    },
    "SAFE_WORSENING": {
        "tone": "early_warning",
        "message_goal": "surface an internal early warning without customer outreach",
        "recommended_channel": "dashboard",
    },
    "WATCH_RECOVERING": {
        "tone": "encouragement",
        "message_goal": "encourage continued recovery and avoid alarmist language",
        "recommended_channel": "sms",
    },
    "WATCH_STABLE": {
        "tone": "gentle_check_in",
        "message_goal": "offer support while maintaining a calm, non-urgent tone",
        "recommended_channel": "sms",
    },
    "WATCH_WORSENING": {
        "tone": "empathetic_assistance",
        "message_goal": "offer flexible support before the customer worsens further",
        "recommended_channel": "sms",
    },
    "CRITICAL_RECOVERING": {
        "tone": "supportive_recovery",
        "message_goal": "support recovery while acknowledging risk remains high",
        "recommended_channel": "email",
    },
    "CRITICAL_STABLE": {
        "tone": "urgent_support",
        "message_goal": "offer immediate support options for persistently high risk",
        "recommended_channel": "email",
    },
    "CRITICAL_SPIRAL": {
        "tone": "crisis_intervention",
        "message_goal": "offer immediate support and prioritize human follow-up",
        "recommended_channel": "email",
    },
}


def _derive_top_drivers(
    stress_profile: Dict[str, Any] | None, latest_shap: Dict[str, Any] | None
) -> List[Dict[str, Any]]:
    if stress_profile and stress_profile.get("stress_top_drivers"):
        return list(stress_profile["stress_top_drivers"])

    latest_shap = latest_shap or {}
    drivers: List[Dict[str, Any]] = []
    for branch_name in ("historian_shap", "behavioral_shap"):
        branch = latest_shap.get(branch_name) or {}
        for feature, value in branch.items():
            contribution = float(value)
            if contribution > 0:
                drivers.append({"feature": feature, "shap_contribution": round(contribution, 6)})
    drivers.sort(key=lambda item: item["shap_contribution"], reverse=True)
    return drivers[:5]


def build_intervention_context(account_id: str, customer_doc: Dict[str, Any]) -> Dict[str, Any]:
    latest_prediction = customer_doc.get("latest_prediction") or {}
    latest_shap = customer_doc.get("latest_shap") or {}
    profile = customer_doc.get("profile") or {}
    decision_matrix = latest_prediction.get("decision_matrix") or {}
    trajectory = latest_prediction.get("trajectory") or {}

    zone = decision_matrix.get("zone") or trajectory.get("zone") or "UNKNOWN"
    strategy = ZONE_MESSAGE_STRATEGY.get(
        zone,
        {
            "tone": "supportive_neutral",
            "message_goal": "offer calm, non-judgmental support",
            "recommended_channel": "email",
        },
    )
    stress_profile = {
        "stress_type": latest_prediction.get("stress_type"),
        "secondary_stress_type": latest_prediction.get("secondary_stress_type"),
        "stress_score_share": latest_prediction.get("stress_score_share"),
        "stress_confidence_band": latest_prediction.get("stress_confidence_band"),
    }
    top_drivers = _derive_top_drivers(stress_profile, latest_shap)

    return {
        "account_id": account_id,
        "customer_profile": {
            "annual_inc": profile.get("annual_inc"),
            "loan_amnt": profile.get("loan_amnt"),
            "term_months": profile.get("term_months"),
        },
        "historian_score": latest_prediction.get("historian_score"),
        "behavioral_score": latest_prediction.get("behavioral_score"),
        "final_risk_score": latest_prediction.get("final_risk_score"),
        "trajectory": trajectory,
        "decision_matrix": decision_matrix,
        "stress_profile": stress_profile,
        "top_drivers": top_drivers,
        "message_strategy": strategy,
    }

