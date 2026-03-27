# 🧠 Project Memory & Decision Log: Risk-Prediction

This document serves as the **"Living Memory"** of our technical journey. It captures every major architectural change, critical decision, and the technical reasoning behind our implementation of the **Financial Historian**.

---

## 🗓️ 2026-03-25: Phase 1 - Repository Restructuring
*   **Decision:** Transitioned from a single `lending_club` directory to a dual-engine structure.
*   **Reasoning:** To isolate the **Universal Historian** (Agnostic) from the native **Financial Historian** (Bureau-heavy).
*   **Action:** Relocated the entire project to `E:\Risk-prediction\` to ensure a dedicated workspace and updated all absolute paths in Python scripts.

---

## 🏗️ 2026-03-26: Phase 2 - The Universal Engine
*   **Decision:** Built the **Agnostic Structural Underwriter** using 13 foundational features.
*   **Reasoning:** We explicitly removed FICO and proprietary grades to create a model that works for "Credit Invisible" or "Thin-File" borrowers globally.
*   **Technical Choice:** Used **Winsorization (99.5%)** and **Sentinel Imputation** to handle extreme wealth and data glitches mathematically, rather than manually.
*   **Outcome:** Achieved **80.5% Recall** on 929,896 unseen 2017-2018 test loans.

---

## 🛡️ 2026-03-26: Phase 3 - Manifest & Security Consolidation
*   **Decision:** Merged redundant `.gitignore` files into a single root manifest.
*   **Reasoning:** Simplified project management and ensured that **MoneyViz** (quarantined) is never pushed, while allowing **ML artifacts** (.pkl and .png) to be tracked.
*   **Action:** Whitelisted `.pkl` and `.png` in the new root `.gitignore`.

---

## 🧩 2026-03-26: Phase 4 - Technical Knowledge Base
*   **Decision:** Created a modular **FAQ Library** in `docs/faq/`.
*   **Reasoning:** To capture specific technical explanations (XGBoost logic, Static vs. Dynamic profiles, Pickling) for future reference.
*   **Action:** Authored 7 dedicated documents covering every major project question.

---

## 📋 Ongoing Strategy: Memory Synchronization
*   **Rule 1:** Always update `task.md` for overall progress tracking.
*   **Rule 2:** Always update `walkthrough.md` for major implementation milestones.
*   **Rule 3:** Use this **`project_memory_log.md`** for capturing the **"Why"** behind our chats and decisions.
