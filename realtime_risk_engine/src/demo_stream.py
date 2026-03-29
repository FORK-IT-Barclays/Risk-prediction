import os
import random
import threading
import time
import logging

from .demo_data import build_next_transaction, build_random_profile, build_random_transactions
from .mongo_store import MongoRiskRepository

logger = logging.getLogger("realtime_risk_engine.demo_stream")


class DemoStreamController:
    """
    Background transaction simulator.

    Transactions keep flowing while the demo is active. Prediction remains an
    explicit separate action triggered by the risk-score endpoint.
    """

    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._rng = random.Random()
        self._interval_seconds = float(os.getenv("DEMO_INTERVAL_SECONDS", "2"))
        self._total_generated = 0
        self._last_cycle_generated = 0
        self._last_cycle_at = None

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self):
        account_count = 0
        try:
            repo = MongoRiskRepository.from_env()
            account_count = len(repo.list_account_ids())
        except Exception:
            # Keep stats endpoint resilient even if DB is temporarily unavailable.
            pass

        estimated_tps = 0.0
        if self._interval_seconds > 0:
            estimated_tps = account_count / self._interval_seconds

        return {
            "running": self.is_running(),
            "interval_seconds": self._interval_seconds,
            "estimated_transactions_per_second": round(estimated_tps, 4),
            "account_count": account_count,
            "last_cycle_generated": self._last_cycle_generated,
            "total_generated": self._total_generated,
            "last_cycle_at": self._last_cycle_at,
        }

    def _ensure_seed_data(self, repo: MongoRiskRepository):
        account_ids = repo.list_account_ids()
        if account_ids:
            return account_ids

        prefix = os.getenv("SAMPLE_ACCOUNT_PREFIX", "CUST")
        start_index = int(os.getenv("SAMPLE_ACCOUNT_START", "1"))
        customer_count = int(os.getenv("SAMPLE_CUSTOMER_COUNT", "10"))
        for offset in range(customer_count):
            account_id = f"{prefix}_{start_index + offset:03d}"
            profile = build_random_profile(self._rng)
            transactions = build_random_transactions(self._rng, profile)
            repo.upsert_customer(account_id, profile)
            repo.upsert_transactions(account_id, transactions)

        return repo.list_account_ids()

    def _run_loop(self):
        repo = MongoRiskRepository.from_env()
        repo.ping()
        repo.ensure_indexes()
        self._ensure_seed_data(repo)
        logger.info(
            "Demo stream started. Interval=%ss",
            self._interval_seconds,
        )

        while not self._stop_event.is_set():
            cycle_generated = 0
            account_ids = repo.list_account_ids()
            for account_id in account_ids:
                profile = repo.load_profile(account_id)
                if profile is None:
                    continue

                history = repo.load_transaction_records(account_id)
                last_transaction = history[-1] if history else None
                next_tx = build_next_transaction(
                    self._rng,
                    profile,
                    last_transaction,
                    account_id=account_id,
                    tx_index=len(history),
                )
                repo.append_transaction(account_id, next_tx)
                cycle_generated += 1

            self._last_cycle_generated = cycle_generated
            self._total_generated += cycle_generated
            self._last_cycle_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.info(
                "Demo stream tick: generated=%s, accounts=%s, interval=%ss, total_generated=%s",
                cycle_generated,
                len(account_ids),
                self._interval_seconds,
                self._total_generated,
            )
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
            logger.info(
                "Demo stream stopped. Total generated transactions=%s",
                self._total_generated,
            )
            return True
