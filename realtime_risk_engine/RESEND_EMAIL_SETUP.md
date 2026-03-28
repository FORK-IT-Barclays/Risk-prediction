# Resend Email Setup

This is the simplest third-party email option currently wired into the engine.

## Endpoint

- `POST /send-email/{account_id}`

It sends the customer-facing intervention message using Resend.

## Environment variables

Add these to `realtime_risk_engine/.env`:

```env
RESEND_API_KEY=your_resend_api_key
RESEND_FROM_EMAIL=onboarding@resend.dev
RESEND_FROM_NAME=Barclays Support
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash
```

## Customer data requirement

Each customer document must contain an email:

```json
{
  "account_id": "CUST_001",
  "email": "yourtestinbox@example.com"
}
```

## Run flow

1. Start server
2. `GET /risk-score`
3. `GET /intervention-report/CUST_001`
4. `POST /send-email/CUST_001`

## Stored in MongoDB

- `latest_intervention_report`
- `intervention_report_history`
- `latest_email_delivery`
- `email_delivery_history`

## Note

Only the customer-facing message is sent in the email body.
Internal SHAP and decision-matrix details remain stored for dashboard use.
