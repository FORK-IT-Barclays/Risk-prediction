# Zone-Gated Stress Profiling Implementation Plan (v1)

## 1. Summary

Build a standalone **Intervention Context Engine** that runs after risk scoring and consumes:

- risk outputs from the dual-model engine
- zone/velocity/acceleration from the physics engine (owned by teammate)

This module will decide **how to intervene** (not whether to score), and prepare a rich context object for GenAI personalization.

v1 is **training-free**: no pseudo-labeling pipeline, no new classifier training.

---

## 2. Design Principles

1. **Physics decides intervention gate**  
   Zone policy is the hard trigger for intervention actions.

2. **Stress profiler decides personalization**  
   Personalization is based on SHAP attributions + risk scores + motion features.

3. **Explainability first**  
   Every recommendation must include evidence drivers.

4. **Safe defaults**  
   If data is missing or ambiguous, degrade gracefully to monitoring-safe action templates.

---

## 3. Inputs, Outputs, and Contracts

### 3.1 Required Inputs per customer cycle

- `account_id`
- `scored_at` (ISO timestamp)
- `final_risk_score`
- `historian_score` (optional)
- `behavioral_score` (optional)
- `historian_shap` (feature->contribution map, optional)
- `behavioral_shap` (feature->contribution map, optional)
- physics module output:
  - `zone_id` (1-9)
  - `velocity`
  - `acceleration`
  - `motion_state` (`recovering|holding|deteriorating`)

### 3.2 Output Contract: `InterventionContext`

```json
{
  "account_id": "CUST_001",
  "generated_at": "2026-03-28T10:30:00Z",
  "risk_snapshot": {
    "final_risk_score": 0.74,
    "historian_score": 0.71,
    "behavioral_score": 0.81
  },
  "physics_snapshot": {
    "zone_id": 3,
    "velocity": 0.08,
    "acceleration": 0.03,
    "motion_state": "deteriorating"
  },
  "intervention_triggered": true,
  "urgency_band": "high",
  "primary_cause": "liquidity_pressure",
  "secondary_cause": "income_instability",
  "cause_distribution": {
    "income_instability": 0.29,
    "liquidity_pressure": 0.43,
    "debt_burden": 0.11,
    "credit_strain": 0.08,
    "spending_escalation": 0.05,
    "structural_fragility": 0.04
  },
  "top_drivers": [
    { "name": "overdraft_t2", "direction": "risk_up", "strength": 0.22 },
    { "name": "min_balance_t2", "direction": "risk_up", "strength": 0.17 },
    { "name": "income_erosion_v", "direction": "risk_up", "strength": 0.16 }
  ],
  "recommended_action": "immediate_rm_intervention",
  "recommended_channel": "sms",
  "recommended_tone": "urgent_supportive",
  "policy_reason": "zone_3_immediate_action"
}
```

---

## 4. Stress Taxonomy and Evidence Mapping

### 4.1 Stress families

- `income_instability`
- `liquidity_pressure`
- `debt_burden`
- `credit_strain`
- `spending_escalation`
- `structural_fragility`

### 4.2 Evidence sources used in v1

- Historian SHAP contribution values
- Behavioral SHAP contribution values
- `historian_score`, `behavioral_score`, `final_risk_score`
- physics motion features (`velocity`, `acceleration`, `zone_id`)

### 4.3 Cause scoring method (v1)

Use deterministic weighted evidence aggregation:

1. map each SHAP feature contribution to one or more stress families
2. compute signed SHAP evidence score per family
3. clamp negative family totals to zero
4. normalize to sum=1 for `cause_distribution`
5. choose top-2 as `primary_cause`, `secondary_cause`

Then apply a small risk-motion prior:

- increase urgency for high `final_risk_score` + positive acceleration
- keep cause ranking SHAP-driven, not raw-signal-driven

No additional model training required.

---

## 5. Zone-Gated Policy Layer

Physics zone governs action routing. Stress profile governs personalization.

### 5.1 Policy behavior

- intervention zones -> produce intervention actions
- monitoring zones -> no outreach; log context only

### 5.2 Example zone mappings (final matrix owned by product/physics team)

- Zone 3 -> `immediate_rm_intervention`
- Zone 6 -> `scheduled_approval_queue`
- Zone 9 -> `second_intervention_attempt`
- Zone 2 -> `educational_nudge_sms`
- Zone 5/8 -> monitor-only actions

### 5.3 Urgency derivation

Use zone priority + motion intensity (`|velocity|`, positive acceleration bias) to classify:

- `low`
- `medium`
- `high`

---

## 6. GenAI Personalization Interface

The module returns a structured prompt payload, not final message text.

### Required payload fields for GenAI

- who: `account_id`, optional profile metadata
- when/why now: zone, velocity, acceleration, urgency
- what is wrong: primary/secondary causes + cause distribution
- evidence: top drivers with direction and strength
- action goal: recommended action + channel + tone
- guardrails: compliant phrasing constraints

This allows personalized messaging grounded in model evidence.

---

## 7. Persistence and Observability

Add these fields to `customers` documents:

- `latest_intervention_context`
- `intervention_context_history` (append-only snapshots)
- `last_intervention_action`
- `last_intervention_at`

Log per cycle:

- zone id
- trigger decision
- primary cause
- action chosen

---

## 8. Implementation Steps

1. Implement `intervention_context` schema and validators.
2. Implement stress evidence mapper and cause scorer.
3. Implement zone-policy adapter that consumes physics output.
4. Implement urgency band calculator.
5. Implement GenAI payload builder.
6. Implement Mongo persistence methods for intervention context snapshots.
7. Add offline runner script for portfolio-wide context generation.
8. Validate with seeded customers and demo transaction flow.
9. After approval, integrate into FastAPI scoring pipeline.

---

## 9. Test Plan and Acceptance Criteria

### 9.1 Functional tests

1. All customers receive valid `InterventionContext` each cycle.
2. Non-intervention zones do not emit intervention actions.
3. Intervention zones emit action/channel/tone deterministically.
4. `cause_distribution` is valid and sums to 1.
5. `top_drivers` align with selected primary cause.
6. Missing historian or behavioral branch still returns context with fallback markers.

### 9.2 Data and persistence tests

1. `latest_intervention_context` updates correctly.
2. `intervention_context_history` appends with timestamps.
3. Risk snapshot and intervention snapshot timestamps are linkable.

### 9.3 Operational tests

1. Portfolio run works for multi-customer collections.
2. Context generation latency is acceptable for batch trigger workflow.

---

## 10. Assumptions and Defaults

- Physics engine outputs are available before this module runs.
- v1 remains training-free (no stress classifier training).
- FastAPI integration remains paused until standalone validation is complete.
- If evidence is weak/ambiguous, default to safe monitor/nudge messaging policy.
- If SHAP is missing for one branch, use available branch SHAP + risk/motion priors.
