from pathlib import Path
import os
import random
from datetime import timedelta
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .intervention.service import build_intervention_report
from .intervention.service import send_intervention_email
from .demo_stream import DemoStreamController
from .demo_data import build_random_profile, build_random_transactions, get_reference_today
from .inference import RiskEngine
from .mongo_store import MongoRiskRepository
from .portfolio_scoring import score_all_customers
from physics_engine.physics_router import router as physics_router

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
LEGACY_UI_DIR = BASE_DIR / "ui"
FRONTEND_DIST_DIR = REPO_ROOT / "prod" / "FullStack" / "dist"

app = FastAPI(title="Realtime Risk Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(physics_router)
if LEGACY_UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=LEGACY_UI_DIR), name="ui")
if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="frontend-assets")
demo_controller = DemoStreamController()
score_lock = threading.Lock()


def _run_portfolio_score():
    with score_lock:
        repo = MongoRiskRepository.from_env()
        repo.ping()
        engine = RiskEngine()
        return score_all_customers(repo, engine)


class AutoScoringController:
    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._interval_seconds = float(os.getenv("AUTO_SCORING_INTERVAL_SECONDS", "10"))
        self._total_runs = 0
        self._last_run_at = None
        self._last_customer_count = 0
        self._last_error = None

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self):
        return {
            "running": self.is_running(),
            "interval_seconds": self._interval_seconds,
            "total_runs": self._total_runs,
            "last_run_at": self._last_run_at,
            "last_customer_count": self._last_customer_count,
            "last_error": self._last_error,
        }

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                results = _run_portfolio_score()
                self._total_runs += 1
                self._last_customer_count = len(results)
                self._last_run_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
            self._stop_event.wait(self._interval_seconds)

    def start(self):
        with self._lock:
            if self.is_running():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return True

    def stop(self):
        with self._lock:
            if not self.is_running():
                return False
            self._stop_event.set()
            self._thread.join(timeout=5)
            return True


auto_scoring_controller = AutoScoringController()


def _auto_scoring_enabled_on_demo() -> bool:
    return os.getenv("AUTO_SCORING_ON_DEMO", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _seed_sample_portfolio(repo: MongoRiskRepository, count: int = 10) -> int:
    account_ids = repo.list_account_ids()
    if account_ids:
        return len(account_ids)

    rng = random.Random(42)
    for offset in range(count):
        account_id = f"CUST_{offset + 1:03d}"
        profile = build_random_profile(rng)
        seed_end_date = get_reference_today().date() - timedelta(days=1 + (offset % 30))
        transactions = build_random_transactions(rng, profile, end_date=seed_end_date)
        repo.upsert_customer(account_id, profile)
        repo.upsert_transactions(account_id, transactions)

    return len(repo.list_account_ids())


@app.get("/", include_in_schema=False)
def root():
    if FRONTEND_DIST_DIR.exists():
        return FileResponse(FRONTEND_DIST_DIR / "index.html")
    if LEGACY_UI_DIR.exists():
        return FileResponse(LEGACY_UI_DIR / "dashboard.html")
    raise HTTPException(status_code=404, detail="Frontend build not found")


@app.get("/api/setup/status")
def setup_status():
    repo = MongoRiskRepository.from_env()
    repo.ping()
    return {
        "status": "ok",
        "database": "connected",
        "frontend_available": FRONTEND_DIST_DIR.exists(),
        "legacy_ui_available": LEGACY_UI_DIR.exists(),
        "customers": repo.count_customers(),
        "transaction_documents": repo.count_transaction_docs(),
        "scored_customers": repo.count_scored_customers(),
        "demo": demo_controller.get_stats(),
        "auto_scoring": auto_scoring_controller.get_stats(),
    }


@app.post("/api/setup/init")
def initialize_database():
    repo = MongoRiskRepository.from_env()
    repo.ping()
    repo.ensure_indexes()
    return {
        "status": "ok",
        "message": "MongoDB indexes are ready.",
        "customers": repo.count_customers(),
        "transaction_documents": repo.count_transaction_docs(),
    }


@app.post("/api/setup/seed")
def seed_sample_data(count: int = 10):
    repo = MongoRiskRepository.from_env()
    repo.ping()
    repo.ensure_indexes()
    total_accounts = _seed_sample_portfolio(repo, count=count)
    return {
        "status": "ok",
        "message": "Sample portfolio is available.",
        "customers": total_accounts,
        "transaction_documents": repo.count_transaction_docs(),
    }


@app.post("/api/setup/seed-and-score")
def seed_and_score(count: int = 10):
    repo = MongoRiskRepository.from_env()
    repo.ping()
    repo.ensure_indexes()
    _seed_sample_portfolio(repo, count=count)
    engine = RiskEngine()
    results = score_all_customers(repo, engine)
    return {
        "status": "ok",
        "message": "Sample portfolio scored successfully.",
        "customers": repo.count_customers(),
        "scored_customers": len(results),
    }


@app.get("/demo")
def start_demo():
    started = demo_controller.start()
    auto_started = auto_scoring_controller.start() if _auto_scoring_enabled_on_demo() else False
    stats = demo_controller.get_stats()
    return {
        "status": "started" if started else "already_running",
        "demo_running": stats["running"],
        "auto_scoring_running": auto_scoring_controller.get_stats()["running"],
        "customer_count": stats["account_count"],
        "interval_seconds": stats["interval_seconds"],
        "estimated_transactions_per_second": stats["estimated_transactions_per_second"],
        "total_generated": stats["total_generated"],
        "auto_scoring_started": auto_started,
        "message": "Transactions are now flowing."
        if not _auto_scoring_enabled_on_demo()
        else "Transactions are now flowing and portfolio scoring is running on a fixed interval.",
    }


@app.get("/risk-score")
def trigger_risk_score():
    results = _run_portfolio_score()

    return {
        "status": "ok",
        "customer_count": len(results),
        "results": [
            {
                "account_id": result["account_id"],
                "status": result["status"],
                "final_risk_score": result["final_risk_score"],
                "historian_score": None
                if result["historian"] is None
                else result["historian"]["historian_score"],
                "behavioral_score": None
                if result["behavioral"] is None
                else result["behavioral"]["behavioral_score"],
                "trajectory": result.get("trajectory"),
                "decision_matrix": result.get("decision_matrix"),
                "stress_profile": result.get("stress_profile"),
                "auto_intervention": result.get("auto_intervention"),
                "current_shap": {
                    "historian_shap": None
                    if result["historian"] is None
                    else result["historian"].get("historian_shap"),
                    "historian_shap_bias": None
                    if result["historian"] is None
                    else result["historian"].get("historian_shap_bias"),
                    "behavioral_shap": None
                    if result["behavioral"] is None
                    else result["behavioral"].get("behavioral_shap"),
                    "behavioral_shap_bias": None
                    if result["behavioral"] is None
                    else result["behavioral"].get("behavioral_shap_bias"),
                },
            }
            for result in results
        ],
    }


@app.get("/all_scores")
def all_scores():
    """
    Return the latest stored risk snapshot for all customers.
    This endpoint is read-only and does not run inference.
    """
    repo = MongoRiskRepository.from_env()
    repo.ping()
    rows = repo.list_latest_scores()
    missing_count = sum(1 for row in rows if not row["has_score"])
    return {
        "status": "ok",
        "customer_count": len(rows),
        "scored_count": len(rows) - missing_count,
        "missing_count": missing_count,
        "results": rows,
    }


@app.get("/api/dashboard/summary")
def dashboard_summary():
    repo = MongoRiskRepository.from_env()
    repo.ping()
    rows = repo.list_latest_scores()

    zone_counts = {}
    intervention_counts = {
        "intervention_required": 0,
        "stress_classification_required": 0,
        "email_ready": 0,
    }

    customers = []
    for row in rows:
        latest = row.get("latest_prediction") or {}
        decision_matrix = latest.get("decision_matrix") or {}
        trajectory = latest.get("trajectory") or {}
        zone = (decision_matrix.get("zone") or trajectory.get("zone") or "UNSCORED")
        zone_counts[zone] = zone_counts.get(zone, 0) + 1

        if decision_matrix.get("intervention_required"):
            intervention_counts["intervention_required"] += 1
        if decision_matrix.get("stress_classification_required"):
            intervention_counts["stress_classification_required"] += 1

        customer_doc = repo.get_customer(row["account_id"]) or {}
        if customer_doc.get("email"):
            intervention_counts["email_ready"] += 1
        latest_delivery = customer_doc.get("latest_email_delivery") or {}

        customers.append(
            {
                "account_id": row["account_id"],
                "has_score": row["has_score"],
                "final_risk_score": row["final_risk_score"],
                "zone": zone,
                "trend": trajectory.get("trend"),
                "intervention_required": bool(
                    decision_matrix.get("intervention_required")
                ),
                "stress_type": latest.get("stress_type"),
                "email": customer_doc.get("email"),
                "latest_email_status": latest_delivery.get("status"),
                "latest_email_sent_at": latest_delivery.get("sent_at"),
                "updated_at": row.get("updated_at"),
            }
        )

    customers.sort(
        key=lambda item: (
            item["final_risk_score"] is None,
            -(item["final_risk_score"] or -1.0),
            item["account_id"],
        )
    )

    return {
        "status": "ok",
        "customer_count": len(customers),
        "demo": demo_controller.get_stats(),
        "auto_scoring": auto_scoring_controller.get_stats(),
        "zone_counts": zone_counts,
        "intervention_counts": intervention_counts,
        "customers": customers,
    }


@app.get("/api/customers/{account_id}")
def customer_detail(account_id: str):
    repo = MongoRiskRepository.from_env()
    repo.ping()
    customer_doc = repo.get_customer(account_id)
    if not customer_doc:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {
        "status": "ok",
        "customer": customer_doc,
        "transactions": repo.get_transactions(account_id) or {},
    }


@app.get("/api/transactions/recent")
def recent_transactions(limit: int = 100):
    repo = MongoRiskRepository.from_env()
    repo.ping()
    return {
        "status": "ok",
        "count": limit,
        "results": repo.list_recent_transactions(limit=limit),
    }


@app.get("/api/health")
def api_health():
    repo = MongoRiskRepository.from_env()
    repo.ping()
    return {
        "status": "ok",
        "database": "connected",
        "frontend_available": FRONTEND_DIST_DIR.exists(),
        "demo": demo_controller.get_stats(),
        "auto_scoring": auto_scoring_controller.get_stats(),
    }


@app.get("/stop-demo")
def stop_demo():
    stopped = demo_controller.stop()
    auto_scoring_stopped = auto_scoring_controller.stop()
    stats = demo_controller.get_stats()
    return {
        "status": "stopped" if stopped else "already_stopped",
        "demo_running": stats["running"],
        "auto_scoring_running": auto_scoring_controller.get_stats()["running"],
        "auto_scoring_stopped": auto_scoring_stopped,
        "interval_seconds": stats["interval_seconds"],
        "estimated_transactions_per_second": stats["estimated_transactions_per_second"],
        "last_cycle_generated": stats["last_cycle_generated"],
        "total_generated": stats["total_generated"],
        "last_cycle_at": stats["last_cycle_at"],
    }


@app.get("/demo-stats")
def demo_stats():
    """Inspect live transaction generation stats while the demo stream runs."""
    return demo_controller.get_stats()


@app.get("/intervention-report/{account_id}")
def intervention_report(account_id: str):
    repo = MongoRiskRepository.from_env()
    repo.ping()
    customer_doc = repo.get_customer(account_id)
    if not customer_doc:
        raise HTTPException(status_code=404, detail="Customer not found")
    if not customer_doc.get("latest_prediction"):
        raise HTTPException(
            status_code=400,
            detail="No scored prediction found for this customer. Trigger /risk-score first.",
        )
    report = build_intervention_report(account_id, customer_doc)
    repo.save_intervention_report(account_id, report)
    return report


@app.post("/send-email/{account_id}")
def send_email(account_id: str):
    repo = MongoRiskRepository.from_env()
    repo.ping()
    customer_doc = repo.get_customer(account_id)
    if not customer_doc:
        raise HTTPException(status_code=404, detail="Customer not found")
    try:
        result = send_intervention_email(account_id, customer_doc)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    repo.save_intervention_report(account_id, result["report"])
    repo.save_email_delivery(account_id, result["email_delivery"])
    return result


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_fallback(full_path: str):
    if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
        raise HTTPException(status_code=404, detail="Not found")

    if FRONTEND_DIST_DIR.exists():
        requested = FRONTEND_DIST_DIR / full_path
        if full_path and requested.exists() and requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST_DIR / "index.html")

    raise HTTPException(status_code=404, detail="Frontend build not found")
