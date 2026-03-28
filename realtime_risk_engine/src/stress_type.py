from __future__ import annotations

from typing import Dict, List, Tuple


STRESS_FAMILIES = [
    "income_instability",
    "liquidity_pressure",
    "debt_burden",
    "credit_strain",
    "spending_escalation",
    "structural_fragility",
]


FEATURE_TO_STRESS = {
    # Behavioral SHAP features
    "income_erosion_v": "income_instability",
    "salary_drift_v": "income_instability",
    "liquidity_momentum_v": "liquidity_pressure",
    "overdraft_v": "liquidity_pressure",
    "overdraft_t2": "liquidity_pressure",
    "avg_balance_t2": "liquidity_pressure",
    "min_balance_t2": "liquidity_pressure",
    "total_out_t2": "spending_escalation",
    "tx_freq_v": "spending_escalation",
    # Historian SHAP features
    "dti": "debt_burden",
    "loan_to_income_ratio": "debt_burden",
    "installment_burden": "debt_burden",
    "loan_amnt": "debt_burden",
    "term_months": "debt_burden",
    "revol_util": "credit_strain",
    "revol_bal": "credit_strain",
    "delinq_2yrs": "credit_strain",
    "pub_rec": "credit_strain",
    "inq_last_6mths": "credit_strain",
    "open_acc": "credit_strain",
    "total_acc": "credit_strain",
    "annual_inc": "structural_fragility",
}


def _positive_drivers(shap_map: Dict[str, float]) -> List[Tuple[str, float]]:
    drivers: List[Tuple[str, float]] = []
    for feature, value in shap_map.items():
        contrib = float(value)
        if contrib > 0:
            drivers.append((feature, contrib))
    drivers.sort(key=lambda x: x[1], reverse=True)
    return drivers


def classify_stress_from_shap(
    historian_shap: Dict[str, float] | None,
    behavioral_shap: Dict[str, float] | None,
) -> Dict[str, object]:
    """
    SHAP-only stress typing.

    Uses only positive SHAP contributions (risk-increasing effects) to compute
    stress-family evidence scores.
    """
    historian_shap = historian_shap or {}
    behavioral_shap = behavioral_shap or {}

    family_scores = {family: 0.0 for family in STRESS_FAMILIES}
    drivers = _positive_drivers(historian_shap) + _positive_drivers(behavioral_shap)
    drivers.sort(key=lambda x: x[1], reverse=True)

    for feature, contrib in drivers:
        family = FEATURE_TO_STRESS.get(feature)
        if family is None:
            continue
        family_scores[family] += float(contrib)

    total = sum(family_scores.values())
    if total <= 0:
        return {
            "stress_type": "UNKNOWN",
            "secondary_stress_type": None,
            "stress_confidence": 0.0,
            "stress_distribution": {k: 0.0 for k in STRESS_FAMILIES},
            "stress_top_drivers": [],
        }

    distribution = {k: round(v / total, 4) for k, v in family_scores.items()}
    ranked = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    primary, primary_score = ranked[0]
    secondary = ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0 else None

    return {
        "stress_type": primary,
        "secondary_stress_type": secondary,
        "stress_confidence": round(primary_score, 4),
        "stress_distribution": distribution,
        "stress_top_drivers": [
            {"feature": feature, "shap_contribution": round(value, 6)}
            for feature, value in drivers[:5]
        ],
    }

