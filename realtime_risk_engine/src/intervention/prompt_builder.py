from __future__ import annotations

from typing import Any, Dict


def build_intervention_prompt(context_packet: Dict[str, Any]) -> str:
    stress = context_packet.get("stress_profile") or {}
    trajectory = context_packet.get("trajectory") or {}
    decision = context_packet.get("decision_matrix") or {}
    strategy = context_packet.get("message_strategy") or {}
    drivers = ", ".join(
        f"{item['feature']} ({item['shap_contribution']})"
        for item in context_packet.get("top_drivers", [])[:3]
    ) or "no dominant positive SHAP drivers"

    return f"""
You are an empathetic financial coach drafting an outreach message for a bank agent.

Customer account: {context_packet.get('account_id')}
Historian score: {context_packet.get('historian_score')}
Behavioral score: {context_packet.get('behavioral_score')}
Final risk score: {context_packet.get('final_risk_score')}
Physics zone: {decision.get('zone')}
Physics confidence: {decision.get('confidence')}
Trend: {trajectory.get('trend')}
Velocity: {trajectory.get('velocity')}
Acceleration: {trajectory.get('acceleration')}
Stress type: {stress.get('stress_type')}
Secondary stress type: {stress.get('secondary_stress_type')}
Top SHAP drivers: {drivers}
Desired tone: {strategy.get('tone')}
Message goal: {strategy.get('message_goal')}

Rules:
- Write this as a real customer email from Admin / Barclays Support.
- Use a warm, professional, supportive tone.
- Do not use judgmental language or accuse the customer of overspending.
- The customer-facing text must not mention SHAP, velocity, acceleration, zones, or internal model names.
- Keep the email concise: subject line, one preview line, and a short body with at most 3 short paragraphs.
- Include a supportive sign-off from "Admin, Barclays Support".
- In addition to the customer email, provide a richer internal explanation that clearly states:
  - the decision-matrix verdict,
  - why intervention was allowed,
  - the primary and secondary stress types,
  - and the top 3 drivers in plain language.
- Return valid JSON with keys: email_subject, preview_text, customer_message, email_body, internal_summary, decision_summary, emotional_readiness.
""".strip()
