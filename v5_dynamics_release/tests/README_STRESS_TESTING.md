# 🚀 VECTOR Stress Testing & Simulation Guide

This directory contains the infrastructure for simulating high-volume, high-velocity customer behaviors (Layer 3 Physics).

## 📂 Test Suite Overview

| Script | Purpose |
| :--- | :--- |
| `test_meta_physics.py` | **Unit Level:** Validates the mathematics of WLS, Parabolic Solver, and Zone Classification. |
| `reliability_audit.py` | **Calibration:** Benchmarks the engine against 5 archetypal risk curves (Linear, Spiral, Pivot, etc.). |
| `stress_test_physics.py`| **Scale:** Simulates hundreds of users with randomized histories to measure performance and "Danger Capture" rate. |
| `pipeline_simulation.py`| **Business Logic:** Generates a full portfolio report with the Strategic 9-Zones and Intervention Tiers. |

---

## 🏗️ Developing New "Stress Types"

To develop a new customer behavior (e.g., "The Stealth Default"), follow these steps in `stress_test_physics.py`:

1.  **Define the Archetype Parameters:** Update the `generate_random_history` factory with new base risk, velocity ($v$), and acceleration ($a$) coefficients.
2.  **Add Noise Injection:** Use `random.uniform` to simulate "Hot Coffee" spikes (noise) vs. "Fever" trends (signal).
3.  **Run the Validation:** 
    ```bash
    python physics_engine/tests/stress_test_physics.py --num_users 500
    ```
4.  **Analyze the "Capture Rate":** Check how many of your new archetypes were correctly classified into `IMMEDIATE_ACTION`.

### Recommended "Stress Types" for Phase 5:
*   **The Flat-line Crash:** Very low risk ($<0.2$) for 175 days, followed by a massive positive acceleration in the last 5 days.
*   **The Seasonal Spike:** Cyclical noise that mimics monthly spending spikes without a long-term worsening trend.
*   **The "Shadow" Recovery:** Customer behavior improves ($v < 0$) but acceleration is positive ($a > 0$), indicating they are about to stop getting better.

---

## 📊 Interpreting the Results
The stress scripts output a **Portfolio Risk Summary**. 
*   **Precision:** If `EXPONENTIAL_CRASH` precision is below 90%, check the $R^2$ Sensitivity in `config.py`.
*   **Latency:** The time between a "Pivot" happening and the engine detecting a `TIER_1_RED` alert.

---
*Drafted by Antigravity AI - 2026-03-29*
