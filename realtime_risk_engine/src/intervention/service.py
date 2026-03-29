from __future__ import annotations

import os
from typing import Any, Dict

from .context_builder import build_intervention_context
from .draft_generator import generate_intervention_draft
from .messaging import ResendEmailProvider
from .prompt_builder import build_intervention_prompt
from .providers import GeminiInterventionProvider


def build_intervention_report(account_id: str, customer_doc: Dict[str, Any]) -> Dict[str, Any]:
    context_packet = build_intervention_context(account_id, customer_doc)
    prompt = build_intervention_prompt(context_packet)
    fallback_draft = generate_intervention_draft(context_packet)
    provider = GeminiInterventionProvider()

    if provider.is_configured():
        try:
            draft = provider.generate(prompt)
            provider_status = {"provider": "gemini", "mode": "live"}
        except Exception as exc:
            draft = fallback_draft
            draft["provider_error"] = str(exc)
            provider_status = {"provider": "gemini", "mode": "fallback"}
    else:
        draft = fallback_draft
        provider_status = {"provider": "local_fallback", "mode": "deterministic"}

    return {
        "account_id": account_id,
        "context_packet": context_packet,
        "llm_prompt": prompt,
        "draft": draft,
        "provider_status": provider_status,
    }


def send_intervention_email(account_id: str, customer_doc: Dict[str, Any]) -> Dict[str, Any]:
    report = build_intervention_report(account_id, customer_doc)
    context_packet = report.get("context_packet") or {}
    decision = context_packet.get("decision_matrix") or {}
    if not decision.get("intervention_required"):
        raise RuntimeError("Decision matrix does not require intervention for this customer")

    to_email = (
        customer_doc.get("email")
        or customer_doc.get("contact", {}).get("email")
    )
    if not to_email:
        raise RuntimeError("No customer email found in the customer document")

    draft = report.get("draft") or {}
    customer_message = draft.get("customer_message") or draft.get("message")
    email_subject = draft.get("email_subject") or "Support is available if things feel tighter right now"
    if not customer_message:
        raise RuntimeError("No customer-facing message available to send")

    provider = ResendEmailProvider()
    response = provider.send_message(to_email, email_subject, customer_message)

    return {
        "account_id": account_id,
        "report": report,
        "email_delivery": {
            "provider": "resend",
            "to": to_email,
            "from": provider.from_email,
            "subject": email_subject,
            "message_id": response.get("message_id"),
            "status": "accepted" if response.get("status_code") in (200, 202) else "unknown",
            "status_code": response.get("status_code"),
            "response_body": response.get("response_body"),
        },
    }


def run_automatic_intervention(account_id: str, customer_doc: Dict[str, Any]) -> Dict[str, Any]:
    latest_prediction = customer_doc.get("latest_prediction") or {}
    decision = latest_prediction.get("decision_matrix") or {}

    if not decision.get("intervention_required"):
        return {
            "account_id": account_id,
            "status": "skipped",
            "reason": "intervention_not_required",
            "report": None,
            "email_delivery": None,
        }

    report = build_intervention_report(account_id, customer_doc)
    draft = report.get("draft") or {}
    to_email = customer_doc.get("email") or customer_doc.get("contact", {}).get("email")

    if not to_email:
        return {
            "account_id": account_id,
            "status": "report_only",
            "reason": "missing_customer_email",
            "report": report,
            "email_delivery": {
                "provider": "resend",
                "status": "skipped",
                "status_code": None,
                "response_body": {"reason": "missing_customer_email"},
            },
        }

    customer_message = draft.get("customer_message") or draft.get("message")
    email_subject = draft.get("email_subject") or "Support is available if things feel tighter right now"
    if not customer_message:
        return {
            "account_id": account_id,
            "status": "report_only",
            "reason": "missing_customer_message",
            "report": report,
            "email_delivery": {
                "provider": "resend",
                "status": "skipped",
                "status_code": None,
                "response_body": {"reason": "missing_customer_message"},
            },
        }

    provider = ResendEmailProvider()
    auto_email_enabled = os.getenv("AUTO_EMAIL_INTERVENTIONS", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not auto_email_enabled:
        return {
            "account_id": account_id,
            "status": "report_only",
            "reason": "auto_email_disabled",
            "report": report,
            "email_delivery": {
                "provider": "resend",
                "status": "skipped",
                "status_code": None,
                "response_body": {"reason": "auto_email_disabled"},
            },
        }
    if not provider.is_configured():
        return {
            "account_id": account_id,
            "status": "report_only",
            "reason": "email_provider_not_configured",
            "report": report,
            "email_delivery": {
                "provider": "resend",
                "status": "skipped",
                "status_code": None,
                "response_body": {"reason": "email_provider_not_configured"},
            },
        }

    try:
        response = provider.send_message(to_email, email_subject, customer_message)
        return {
            "account_id": account_id,
            "status": "sent",
            "reason": None,
            "report": report,
            "email_delivery": {
                "provider": "resend",
                "to": to_email,
                "from": provider.from_email,
                "subject": email_subject,
                "message_id": response.get("message_id"),
                "status": "accepted" if response.get("status_code") in (200, 202) else "unknown",
                "status_code": response.get("status_code"),
                "response_body": response.get("response_body"),
            },
        }
    except RuntimeError as exc:
        return {
            "account_id": account_id,
            "status": "report_only",
            "reason": str(exc),
            "report": report,
            "email_delivery": {
                "provider": "resend",
                "status": "failed",
                "status_code": None,
                "response_body": {"reason": str(exc)},
            },
        }
