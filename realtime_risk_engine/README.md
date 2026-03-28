# Realtime Risk Engine

Dual-model real-time credit risk engine with:

- Universal Historian (structural baseline risk)
- VECTOR Behavioral model (transactional/velocity risk)
- Fusion layer (single final score)
- MongoDB-backed runtime + FastAPI endpoints

## Project Structure

```text
realtime_risk_engine/
  models/
    behavioral_engine_v2.pkl
    universal_historian_v1.pkl
    universal_features_map.pkl
  scripts/
    init_mongo.py
    seed_mongo_sample.py
    score_from_mongo.py
  src/
    server.py
    demo_stream.py
    inference.py
    historian.py
    feature_engine.py
    fusion.py
    mongo_store.py
```

## Requirements

- Python 3.10+
- Local MongoDB running on `localhost:27017`
- Dependencies from repo root `requirements.txt`

Install dependencies in your venv:

```bash
pip install -r requirements.txt
```

## MongoDB Setup

1. Ensure MongoDB service is running locally.
2. Create/update `realtime_risk_engine/.env`.
3. Initialize DB connection + indexes:

```bash
python realtime_risk_engine/scripts/init_mongo.py
```

Expected success output:

```text
MongoDB connection successful.
Indexes ensured for customers.account_id and transactions.account_id.
```

## Environment Variables

Use `realtime_risk_engine/.env.example` as base:

```env
MONGODB_URI=mongodb://<username>:<password>@localhost:27017/?authSource=admin
MONGODB_DB_NAME=realtime_risk_engine
MONGODB_CUSTOMERS_COLLECTION=customers
MONGODB_TRANSACTIONS_COLLECTION=transactions
SAMPLE_ACCOUNT_PREFIX=CUST
SAMPLE_ACCOUNT_START=1
SAMPLE_CUSTOMER_COUNT=10
DEMO_INTERVAL_SECONDS=2
```

Notes:

- If auth is disabled locally, URI can be `mongodb://localhost:27017`.
- If auth is enabled, include valid username/password and `authSource=admin` (or your auth DB).

## Seed Initial Data

Seed sample customer profiles + transaction history:

```bash
python realtime_risk_engine/scripts/seed_mongo_sample.py
```

This creates:

- one customer document per account in `customers`
- one transaction document per account in `transactions`

For a fixed starter dataset, see:

- [INITIAL_CUSTOMERS.md](/e:/Machine learning/barclayss/risk-prediction/realtime_risk_engine/INITIAL_CUSTOMERS.md)

## Start API Server

From repo root:

```bash
uvicorn realtime_risk_engine.src.server:app --host 127.0.0.1 --port 8000 --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

- `GET /demo`
  Starts background transaction flow for all customers. If DB is empty, it seeds sample customers automatically.

- `POST /stop-demo`
  Stops the background transaction flow.

- `GET /demo-stats`
  Returns runtime stream stats:
  `running`, `interval_seconds`, `estimated_transactions_per_second`, `account_count`, `last_cycle_generated`, `total_generated`, `last_cycle_at`.

- `GET /risk-score`
  Runs one scoring pass for all customers and stores a timestamped snapshot into each customer document (`latest_prediction` + append to `risk_history`).

- `GET /all_scores`
  Returns latest and historical risk score data for all customers, including:
  `risk_score_timestamp`, `risk_score_timestamps_all`, `risk_score_history`, `prediction_count`.

## Risk Logic

1. Historian scores structural profile risk.
2. Behavioral model scores recent transaction behavior.
3. Fusion combines both:

```text
final_risk_score = clamp(historian_score + 0.30 * (behavioral_score - 0.46), 0.0, 1.0)
```

Fallbacks:

- missing historian -> use behavioral score
- missing behavioral -> use historian score
- both missing -> `INSUFFICIENT_DATA`

## Typical Test Flow

1. `python realtime_risk_engine/scripts/init_mongo.py`
2. `python realtime_risk_engine/scripts/seed_mongo_sample.py`
3. start server with `uvicorn ...`
4. call `GET /demo`
5. wait 10-20 seconds
6. call `GET /risk-score`
7. call `GET /all_scores`
8. call `POST /stop-demo`

## Data Model (MongoDB)

### `customers` collection

- `account_id`
- `profile` (raw structural inputs)
- `latest_prediction` (most recent score snapshot + timestamp)
- `risk_history` (array of historical score snapshots)
- `updated_at`

### `transactions` collection

- `account_id`
- `transactions` (array of transaction records)
- `updated_at`
