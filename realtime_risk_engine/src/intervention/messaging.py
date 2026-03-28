from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict


RESEND_API_BASE = "https://api.resend.com/emails"


class ResendEmailProvider:
    def __init__(self):
        self.api_key = os.getenv("RESEND_API_KEY")
        self.from_email = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")
        self.from_name = os.getenv("RESEND_FROM_NAME", "Barclays Support")

    def is_configured(self) -> bool:
        return all([self.api_key, self.from_email])

    def send_message(
        self,
        to_email: str,
        subject: str,
        plain_text_body: str,
    ) -> Dict[str, Any]:
        if not self.is_configured():
            raise RuntimeError("Resend environment variables are not fully configured")

        payload = {
            "from": f"{self.from_name} <{self.from_email}>",
            "to": [to_email],
            "subject": subject,
            "text": plain_text_body,
        }
        request = urllib.request.Request(
            RESEND_API_BASE,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
                "Origin": "https://resend.com",
                "Referer": "https://resend.com/",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
                return {
                    "status_code": response.status,
                    "response_body": body,
                    "message_id": body.get("id"),
                }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Resend API error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Resend connection error: {exc.reason}") from exc
