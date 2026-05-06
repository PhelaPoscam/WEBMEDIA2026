"""
Configurable affective matching strategies for playlist generation.

Supports multiple distance metrics for benchmarking different approaches
to capturing human perception of affect in the valence-arousal space.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol

import numpy as np


def normalize_valence(val: float) -> float:
    """
    Normalize valence to the [0, 1] range.

    Supports both [-1, 1] legacy/normalized scale and [1, 9] scale.
    """
    val_f = float(val)
    if -1.0 <= val_f <= 1.0:
        return (val_f + 1.0) / 2.0
    # [1, 9] scale
    clamped = max(1.0, min(9.0, val_f))
    return (clamped - 1.0) / 8.0


def normalize_arousal(aro: float) -> float:
    """
    Normalize arousal to the [0, 1] range.

    Supports both [-1, 1] scale (mapped to 1-9 first) and [1, 9] scale.
    """
    aro_f = float(aro)
    if -1.0 <= aro_f <= 1.0:
        aro_f = (aro_f * 4.0) + 5.0
    clamped = max(1.0, min(9.0, aro_f))
    return (clamped - 1.0) / 8.0


class MatchingStrategy(str, Enum):
    """Available affective matching strategies."""

    EUCLIDEAN = "euclidean"  # Linear Euclidean distance (baseline)
    MANHATTAN = "manhattan"  # Taxicab/L1 distance
    CHEBYSHEV = "chebyshev"  # L-infinity distance (max dimension)
    QUADRATIC = "quadratic"  # Non-linear quadratic distance for affect perception
    MAHALANOBIS = "mahalanobis"  # Distance considering correlation structure


class AffectiveDistanceMetric(ABC):
    """Abstract base for affective distance calculations."""

    @abstractmethod
    def distance(
        self,
        target_point: np.ndarray,
        stimulus_point: np.ndarray,
    ) -> float:
        """
        Calculate distance between target and stimulus in affect space.

        Args:
            target_point: Target valence-arousal coordinates [normalized_val, normalized_aro]
            stimulus_point: Stimulus valence-arousal coordinates [normalized_val, normalized_aro]

        Returns:
            Distance metric value (lower = better match)
        """
        pass


class EuclideanDistance(AffectiveDistanceMetric):
    """Linear Euclidean distance (baseline)."""

    def distance(
        self,
        target_point: np.ndarray,
        stimulus_point: np.ndarray,
    ) -> float:
        """Standard L2 norm: sqrt(sum((x-y)^2))"""
        return float(np.linalg.norm(target_point - stimulus_point))


class ManhattanDistance(AffectiveDistanceMetric):
    """Taxicab/L1 distance - sum of absolute differences."""

    def distance(
        self,
        target_point: np.ndarray,
        stimulus_point: np.ndarray,
    ) -> float:
        """Manhattan distance: sum(|x-y|)"""
        return float(np.sum(np.abs(target_point - stimulus_point)))


class ChebyshevDistance(AffectiveDistanceMetric):
    """L-infinity distance - maximum absolute difference across dimensions."""

    def distance(
        self,
        target_point: np.ndarray,
        stimulus_point: np.ndarray,
    ) -> float:
        """Chebyshev distance: max(|x-y|)"""
        return float(np.max(np.abs(target_point - stimulus_point)))


class QuadraticDistance(AffectiveDistanceMetric):
    """Non-linear quadratic distance for affect perception.

    Emphasizes larger deviations in the affect space, reflecting the
    non-linear nature of human perception of emotional intensity.
    """

    def distance(
        self,
        target_point: np.ndarray,
        stimulus_point: np.ndarray,
    ) -> float:
        """Quadratic distance: sqrt(sum((x-y)^2 * weights))

        Applies squared deviations to emphasize perceptual differences.
        """
        diff = target_point - stimulus_point
        # Weight each dimension to reflect affect space curvature
        if diff.shape[0] == 1:
            weights = np.array([1.2])
        else:
            weights = np.array([1.2, 1.0])  # Slightly emphasize valence
        weighted_sq_diff = (diff**2) * weights
        return float(np.sqrt(np.sum(weighted_sq_diff)))


class MahalanobisDistance(AffectiveDistanceMetric):
    """Mahalanobis distance - accounts for correlation in affect dimensions.

    Based on typical covariance structure between valence and arousal.
    An optional covariance matrix can be injected for custom configurations.
    """

    _DEFAULT_COV = np.array(
        [
            [0.25, -0.08],
            [-0.08, 0.20],
        ]
    )

    def __init__(self, cov_matrix: np.ndarray | None = None):
        self.cov_matrix = cov_matrix if cov_matrix is not None else self._DEFAULT_COV.copy()
        try:
            self.inv_cov = np.linalg.inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            self.inv_cov = np.eye(self.cov_matrix.shape[0])

    def distance(
        self,
        target_point: np.ndarray,
        stimulus_point: np.ndarray,
    ) -> float:
        """Mahalanobis distance: sqrt((x-y)^T * Sigma^-1 * (x-y))"""
        diff = target_point - stimulus_point
        if diff.shape[0] == 1:
            # 1D fallback using valence variance (0.25)
            val_var = self.cov_matrix[0, 0]  # 0.25
            mahal_dist = np.sqrt((diff[0] ** 2) / val_var)
        else:
            # Ensure 2D shape for matrix multiplication
            if diff.ndim == 1:
                diff = diff.reshape(-1)
            mahal_dist = np.sqrt(diff @ self.inv_cov @ diff.T)
        return float(mahal_dist)


class AffectiveMatchingEngine:
    """Factory and manager for affective matching strategies."""

    _strategies = {
        MatchingStrategy.EUCLIDEAN: EuclideanDistance,
        MatchingStrategy.MANHATTAN: ManhattanDistance,
        MatchingStrategy.CHEBYSHEV: ChebyshevDistance,
        MatchingStrategy.QUADRATIC: QuadraticDistance,
        MatchingStrategy.MAHALANOBIS: MahalanobisDistance,
    }

    @classmethod
    def get_metric(cls, strategy: MatchingStrategy) -> AffectiveDistanceMetric:
        """
        Get a distance metric for the specified strategy.

        Args:
            strategy: The MatchingStrategy to use

        Returns:
            Instantiated distance metric

        Raises:
            ValueError: If strategy is not supported
        """
        metric_class = cls._strategies.get(strategy)
        if metric_class is None:
            valid_strategies = ", ".join(s.value for s in MatchingStrategy)
            raise ValueError(
                f"Unknown matching strategy '{strategy}'. " f"Valid options: {valid_strategies}"
            )
        return metric_class()

    @classmethod
    def get_strategy_info(cls) -> dict[str, str]:
        """Return descriptions of all available strategies."""
        return {
            MatchingStrategy.EUCLIDEAN.value: (
                "Linear Euclidean distance (baseline) - "
                "standard L2 norm in normalized affect space"
            ),
            MatchingStrategy.MANHATTAN.value: (
                "Manhattan distance - sum of absolute differences, " "less sensitive to outliers"
            ),
            MatchingStrategy.CHEBYSHEV.value: (
                "Chebyshev distance - maximum absolute difference, " "worst-case dimension mismatch"
            ),
            MatchingStrategy.QUADRATIC.value: (
                "Quadratic distance - non-linear metric emphasizing "
                "larger deviations in affect perception"
            ),
            MatchingStrategy.MAHALANOBIS.value: (
                "Mahalanobis distance - accounts for correlation "
                "between valence and arousal dimensions"
            ),
        }


class UnifiedStimulus(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def uri(self) -> str: ...

    @property
    def normative_valence(self) -> float: ...

    @property
    def normative_arousal(self) -> float: ...

    @property
    def duration_ms(self) -> int: ...


def compute_effective_affect(
    normative_val: float,
    normative_aro: float,
    preference_score: float | None,
    match_arousal: bool,
    valence_shift_strength: float = 0.5,
    arousal_shift_strength: float = 0.5,
    likert_min: float = 1.0,
    likert_max: float = 5.0,
) -> tuple[float, float]:
    """Applies a subjective preference shift to normative affect scores."""
    use_legacy_valence_scale = -1.0 <= normative_val <= 1.0
    val_min = -1.0 if use_legacy_valence_scale else 1.0
    val_max = 1.0 if use_legacy_valence_scale else 9.0

    if preference_score is None:
        eff_val = normative_val
        eff_aro = normative_aro
    else:
        pref_clamped = max(likert_min, min(likert_max, float(preference_score)))
        likert_midpoint = (likert_min + likert_max) / 2.0
        likert_half_range = (likert_max - likert_min) / 2.0
        pref_norm = (pref_clamped - likert_midpoint) / likert_half_range

        if pref_norm == 0.0:
            eff_val = normative_val
            eff_aro = normative_aro
        else:
            v_target = val_max if pref_norm > 0 else val_min
            magnitude = abs(pref_norm)
            eff_val = normative_val + valence_shift_strength * magnitude * (
                v_target - normative_val
            )

            if match_arousal:
                a_target = 9.0 if pref_norm > 0 else 1.0
                eff_aro = normative_aro + arousal_shift_strength * magnitude * (
                    a_target - normative_aro
                )
            else:
                eff_aro = normative_aro

    return max(val_min, min(val_max, eff_val)), max(1.0, min(9.0, eff_aro))


def select_stimuli(
    pool: list[UnifiedStimulus],
    target_valence: float,
    target_arousal: float,
    min_duration_ms: int,
    max_duration_ms: int,
    distance_metric: Any,
    match_arousal: bool = True,
    preferences: dict[str, float] | None = None,
    valence_shift_strength: float = 0.5,
    arousal_shift_strength: float = 0.5,
    likert_min: float = 1.0,
    likert_max: float = 5.0,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """
    Selects stimuli from a pool to satisfy duration constraints, minimizing distance to target.
    Returns: (selected_stimuli_list, achieved_duration_ms, phase_audit_data)
    """
    scored_pool = []
    if match_arousal:
        target_pt = np.array([normalize_valence(target_valence), normalize_arousal(target_arousal)])
    else:
        target_pt = np.array([normalize_valence(target_valence)])

    for s in pool:
        pref = preferences.get(s.id) if preferences else None
        eff_v, eff_a = compute_effective_affect(
            s.normative_valence,
            s.normative_arousal,
            pref,
            match_arousal,
            valence_shift_strength,
            arousal_shift_strength,
            likert_min,
            likert_max,
        )

        if match_arousal:
            stim_pt = np.array([normalize_valence(eff_v), normalize_arousal(eff_a)])
        else:
            stim_pt = np.array([normalize_valence(eff_v)])

        dist = distance_metric.distance(target_pt, stim_pt)
        scored_pool.append(
            {
                "stimulus": s,
                "distance": dist,
                "eff_valence": eff_v,
                "eff_arousal": eff_a,
            }
        )

    phase_audit_data = {
        "candidate_pool_size": len(pool),
        "selections": [],
        "rejected_candidates": [],
    }

    selected = []
    current_duration = 0
    remaining_pool = list(scored_pool)

    while current_duration < min_duration_ms and remaining_pool:
        remaining = max_duration_ms - current_duration

        fits = [item for item in remaining_pool if item["stimulus"].duration_ms <= remaining]
        if fits:
            fits.sort(key=lambda x: (x["distance"], -x["stimulus"].duration_ms, x["stimulus"].id))
            chosen = fits[0]
            selection_reason = "closest_affective_match"
        else:
            remaining_pool.sort(
                key=lambda x: (
                    abs(remaining - x["stimulus"].duration_ms),
                    x["distance"],
                    x["stimulus"].duration_ms,
                    x["stimulus"].id,
                )
            )
            chosen = remaining_pool[0]
            selection_reason = "closest_duration_fit"

        selected.append(chosen)
        phase_audit_data["selections"].append(
            {
                "stimulus_id": chosen["stimulus"].id,
                "distance": float(chosen["distance"]),
                "duration_ms": int(chosen["stimulus"].duration_ms),
                "selection_reason": selection_reason,
            }
        )
        current_duration += chosen["stimulus"].duration_ms
        remaining_pool.remove(chosen)

    # Take up to 5 remaining items with the lowest distance as rejected near-candidates
    remaining_pool.sort(key=lambda x: (x["distance"], x["stimulus"].id))
    for item in remaining_pool[:5]:
        phase_audit_data["rejected_candidates"].append(
            {
                "stimulus_id": item["stimulus"].id,
                "distance": float(item["distance"]),
                "duration_ms": int(item["stimulus"].duration_ms),
            }
        )

    formatted = []
    for item in selected:
        s = item["stimulus"]
        formatted.append(
            {
                "stimulus_id": s.id,
                "stimulus_code": getattr(s, "code", "N/A"),
                "uri": s.uri,
                "modality": getattr(s, "modality", "unknown"),
                "distance": float(item["distance"]),
                "eff_valence": float(item["eff_valence"]),
                "eff_arousal": float(item["eff_arousal"]),
                "raw_valence": float(s.normative_valence),
                "raw_arousal": float(s.normative_arousal),
                "duration_ms": int(s.duration_ms),
            }
        )

    return formatted, current_duration, phase_audit_data
