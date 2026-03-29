from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _looks_configured(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return not (
        lowered.startswith("your_")
        or "<" in text
        or "placeholder" in lowered
    )


class GeminiInterventionProvider:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.base_url = os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL).rstrip("/")

    def is_configured(self) -> bool:
        return _looks_configured(self.api_key)

    def generate(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json",
            },
        }
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini API connection error: {exc.reason}") from exc

        candidates = body.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini API returned no candidates: {body}")

        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        if not text:
            raise RuntimeError(f"Gemini API returned empty text: {body}")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {
                "customer_message": text,
                "internal_summary": {
                    "provider_note": "Gemini returned non-JSON text"
                },
                "emotional_readiness": "Medium",
            }

        if "message" in parsed and "customer_message" not in parsed:
            parsed["customer_message"] = parsed.pop("message")
        if "rationale" in parsed and "internal_summary" not in parsed:
            parsed["internal_summary"] = parsed.pop("rationale")
        if "email_subject" not in parsed:
            parsed["email_subject"] = "Support is available if things feel tighter right now"

        parsed["provider"] = "gemini"
        parsed["model"] = self.model
        return parsed
