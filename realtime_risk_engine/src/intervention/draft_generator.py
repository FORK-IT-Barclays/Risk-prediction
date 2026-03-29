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
    stress = context_packet.get("stress_profile") or {}
    strategy = context_packet.get("message_strategy") or {}
    zone = decision.get("zone") or "UNKNOWN"
    drivers = context_packet.get("top_drivers") or []
    reason_phrase = _build_reason_phrase(drivers)

    if not decision.get("intervention_required"):
        customer_message = (
            "No customer outreach is recommended right now. "
            "Keep monitoring this account and review again after the next scoring cycle."
        )
        email_subject = "Monitoring update only"
        preview_text = "No customer outreach is recommended for this account right now."
    elif zone.startswith("CRITICAL"):
        customer_message = (
            f"We noticed {reason_phrase}, and support is available right away if things feel difficult. "
            "We can help review flexible payment or hardship options with you."
        )
        email_subject = "Support is available if things feel tighter right now"
        preview_text = "We are reaching out because support is available right away if it would help."
    elif zone.endswith("WORSENING") or zone.endswith("SPIRAL"):
        customer_message = (
            f"We noticed {reason_phrase}, and we wanted to check in early before it becomes harder to manage. "
            "If helpful, we can talk through a more flexible repayment arrangement."
        )
        email_subject = "A quick check-in and support options"
        preview_text = "We wanted to reach out early and let you know support is available."
    else:
        customer_message = (
            "We are here if you need support, and we can help review options that may make things easier. "
            "Please reach out if a flexible plan would help."
        )
        email_subject = "We are here to support you"
        preview_text = "Support is available if a more flexible plan would help."

    email_body = (
        f"{preview_text}\n\n"
        f"{customer_message}\n\n"
        "If speaking with us would help, we can review practical options together and help you choose the next best step.\n\n"
        "Admin\nBarclays Support"
    )

    decision_summary = (
        f"Decision matrix outcome: zone={zone}, confidence={decision.get('confidence')}, "
        f"intervention_required={decision.get('intervention_required')}, "
        f"stress_classification_required={decision.get('stress_classification_required')}, "
        f"reason={decision.get('reason')}."
    )

    internal_summary = (
        f"Primary stress type is {stress.get('stress_type')} and secondary stress type is "
        f"{stress.get('secondary_stress_type')}. "
        f"The outreach strategy uses tone={strategy.get('tone')} with message goal "
        f"'{strategy.get('message_goal')}'. "
        f"Top drivers are {', '.join(_humanize_feature(item['feature']) for item in drivers[:3]) or 'none'}."
    )

    return {
        "email_subject": email_subject,
        "preview_text": preview_text,
        "customer_message": customer_message,
        "email_body": email_body,
        "internal_summary": internal_summary,
        "decision_summary": decision_summary,
        "emotional_readiness": _readiness_from_zone(zone),
        "recommended_channel": strategy.get("recommended_channel"),
        "provider": "local_fallback",
    }
