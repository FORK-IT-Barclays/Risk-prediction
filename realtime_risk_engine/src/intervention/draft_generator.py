from __future__ import annotations

from typing import Any, Dict, List


def _humanize_feature(feature: str) -> str:
    return feature.replace("_", " ")


def _readiness_from_zone(zone: str) -> str:
    if zone.startswith("CRITICAL"):
        return "High"
    if zone.startswith("WATCH"):
        return "Medium"
    return "Low"


def _build_reason_phrase(drivers: List[Dict[str, Any]]) -> str:
    labels = [_humanize_feature(item["feature"]) for item in drivers[:2]]
    if not labels:
        return "things may be feeling tighter than usual"
    if len(labels) == 1:
        return f"signals around {labels[0]} suggest pressure"
    return f"signals around {labels[0]} and {labels[1]} suggest pressure"


def generate_intervention_draft(context_packet: Dict[str, Any]) -> Dict[str, Any]:
    decision = context_packet.get("decision_matrix") or {}
    strategy = context_packet.get("message_strategy") or {}
    zone = decision.get("zone") or "UNKNOWN"
    drivers = context_packet.get("top_drivers") or []
    reason_phrase = _build_reason_phrase(drivers)

    if not decision.get("intervention_required"):
        customer_message = (
            "No customer outreach is recommended right now. "
            "Keep monitoring this account and review again after the next scoring cycle."
        )
    elif zone.startswith("CRITICAL"):
        customer_message = (
            f"We noticed {reason_phrase}, and support is available right away if things feel difficult. "
            "We can help review flexible payment or hardship options with you."
        )
    elif zone.endswith("WORSENING") or zone.endswith("SPIRAL"):
        customer_message = (
            f"We noticed {reason_phrase}, and we wanted to check in early before it becomes harder to manage. "
            "If helpful, we can talk through a more flexible repayment arrangement."
        )
    else:
        customer_message = (
            "We’re here if you need support, and we can help review options that may make things easier. "
            "Please reach out if a flexible plan would help."
        )

    return {
        "email_subject": "Support is available if things feel tighter right now",
        "customer_message": customer_message,
        "internal_summary": {
            "zone": zone,
            "tone": strategy.get("tone"),
            "message_goal": strategy.get("message_goal"),
            "top_drivers": drivers[:3],
        },
        "emotional_readiness": _readiness_from_zone(zone),
        "recommended_channel": strategy.get("recommended_channel"),
        "provider": "local_fallback",
    }
