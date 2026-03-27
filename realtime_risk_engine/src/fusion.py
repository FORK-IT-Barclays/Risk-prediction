from typing import Optional

from .config import FUSION_BEHAVIORAL_CENTER, FUSION_BEHAVIORAL_WEIGHT


def clamp_score(score: float) -> float:
    """Keep all exposed scores within probability bounds."""
    return max(0.0, min(1.0, float(score)))


def fuse_scores(
    historian_score: Optional[float],
    behavioral_score: Optional[float],
) -> Optional[float]:
    """
    Fuse the two model scores without collapsing into a weighted ensemble.

    The historian score anchors long-horizon structural risk while the
    behavioral score applies a bounded runtime adjustment around a neutral
    center of 0.50.
    """
    if historian_score is None and behavioral_score is None:
        return None
    if historian_score is None:
        return clamp_score(behavioral_score)
    if behavioral_score is None:
        return clamp_score(historian_score)

    adjustment = FUSION_BEHAVIORAL_WEIGHT * (
        float(behavioral_score) - FUSION_BEHAVIORAL_CENTER
    )
    return clamp_score(float(historian_score) + adjustment)
