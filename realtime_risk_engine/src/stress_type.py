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
    "income_erosion_v": "income_instability",
    "salary_drift_v": "income_instability",
    "liquidity_momentum_v": "liquidity_pressure",
    "overdraft_v": "liquidity_pressure",
    "overdraft_t2": "liquidity_pressure",
    "avg_balance_t2": "liquidity_pressure",
    "min_balance_t2": "liquidity_pressure",
    "total_out_t2": "spending_escalation",
    "tx_freq_v": "spending_escalation",
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
        contribution = float(value)
        if contribution > 0:
            drivers.append((feature, contribution))
    drivers.sort(key=lambda item: item[1], reverse=True)
    return drivers


def _family_scores_from_shap(shap_map: Dict[str, float]) -> Dict[str, float]:
    scores = {family: 0.0 for family in STRESS_FAMILIES}
    for feature, contribution in _positive_drivers(shap_map):
        family = FEATURE_TO_STRESS.get(feature)
        if family is not None:
            scores[family] += contribution
    return scores


def _normalize_family_scores(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return {family: 0.0 for family in STRESS_FAMILIES}
    return {family: score / total for family, score in scores.items()}


def _score_share_band(score_share: float) -> str:
    if score_share >= 0.55:
        return "high"
    if score_share >= 0.40:
        return "medium"
    return "low"


def classify_stress_from_shap(
    historian_shap: Dict[str, float] | None,
    behavioral_shap: Dict[str, float] | None,
) -> Dict[str, object]:
    """
    Derive stress type using only risk-increasing SHAP contributions.
    """
    historian_shap = historian_shap or {}
    behavioral_shap = behavioral_shap or {}

    historian_family_scores = _family_scores_from_shap(historian_shap)
    behavioral_family_scores = _family_scores_from_shap(behavioral_shap)

    historian_distribution = _normalize_family_scores(historian_family_scores)
    behavioral_distribution = _normalize_family_scores(behavioral_family_scores)

    # Blend the two branches using the same relative influence used by score fusion.
    historian_weight = 1.0 if sum(historian_family_scores.values()) > 0 else 0.0
    behavioral_weight = 0.30 if sum(behavioral_family_scores.values()) > 0 else 0.0
    combined_weight = historian_weight + behavioral_weight

    family_scores = {family: 0.0 for family in STRESS_FAMILIES}
    if combined_weight > 0:
        for family in STRESS_FAMILIES:
            family_scores[family] = (
                historian_distribution[family] * historian_weight
                + behavioral_distribution[family] * behavioral_weight
            ) / combined_weight

    drivers = _positive_drivers(historian_shap) + _positive_drivers(behavioral_shap)
    drivers.sort(key=lambda item: item[1], reverse=True)

    total_score = sum(family_scores.values())
    if total_score <= 0:
        return {
            "stress_type": "UNKNOWN",
            "secondary_stress_type": None,
            "stress_score_share": 0.0,
            "stress_confidence_band": "low",
            "stress_distribution": {family: 0.0 for family in STRESS_FAMILIES},
            "stress_top_drivers": [],
        }

    distribution = {
        family: round(score / total_score, 4)
        for family, score in family_scores.items()
    }
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    primary_family, primary_score = ranked[0]
    secondary_family = ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0 else None

    return {
        "stress_type": primary_family,
        "secondary_stress_type": secondary_family,
        "stress_score_share": round(primary_score, 4),
        "stress_confidence_band": _score_share_band(primary_score),
        "stress_distribution": distribution,
        "stress_top_drivers": [
            {"feature": feature, "shap_contribution": round(value, 6)}
            for feature, value in drivers[:5]
        ],
    }
