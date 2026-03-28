# Twilio SMS Setup

Use this when you want to send the generated customer-facing intervention message as an SMS.

## What the code now supports

- `GET /intervention-report/{account_id}`
  - builds the context packet
  - calls Gemini if configured
  - falls back to a deterministic draft otherwise
  - stores the latest intervention report in MongoDB

- `POST /send-message/{account_id}`
  - rebuilds the latest intervention report
  - checks the decision matrix
  - sends the customer-facing message through Twilio
  - stores SMS delivery metadata in MongoDB

## Twilio trial notes

Twilio trial projects currently have important limits, including:

- only one Twilio number per trial account
- messages begin with a Twilio trial prefix
- trial SMS can only be sent to verified phone numbers
- trial accounts are limited to 50 messages per day

Official Twilio help:

- https://help.twilio.com/articles/360036052753

Official send SMS tutorial:

- https://www.twilio.com/docs/messaging/tutorials/how-to-send-sms-messages

## Step-by-step setup

1. Create a Twilio account:

- https://www.twilio.com/try-twilio

2. In the Twilio Console, collect:

- `Account SID`
- `Auth Token`
- your Twilio phone number

3. If you are on a trial project, verify the destination phone number in Twilio.

4. Add these values to `realtime_risk_engine/.env`:

```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.5-flash
```

5. Restart the FastAPI server.

6. Run a scoring cycle:

```text
GET /risk-score
```

7. Generate an intervention report:

```text
GET /intervention-report/CUST_001
```

8. Send the SMS:

```text
POST /send-message/CUST_001
```

## What gets stored in MongoDB

In the customer document:

- `latest_intervention_report`
- `intervention_report_history`
- `latest_sms_delivery`
- `sms_delivery_history`

## Important implementation detail

The SMS body uses the customer-facing field:

- `draft.customer_message`

It does **not** send:

- SHAP values
- zone labels
- velocity / acceleration
- internal matrix data

Those remain in the internal report only.
