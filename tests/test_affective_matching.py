import numpy as np
import pytest

from src.affective_matching import (
    AffectiveMatchingEngine,
    ChebyshevDistance,
    EuclideanDistance,
    MahalanobisDistance,
    ManhattanDistance,
    MatchingStrategy,
    QuadraticDistance,
    compute_effective_affect,
    normalize_arousal,
    normalize_valence,
)


class TestNormalization:
    def test_normalize_valence_legacy_range(self):
        assert normalize_valence(-1.0) == 0.0
        assert normalize_valence(0.0) == 0.5
        assert normalize_valence(1.0) == 1.0

    def test_normalize_valence_nine_range(self):
        assert normalize_valence(5.0) == 0.5
        assert normalize_valence(9.0) == 1.0
        # Values in [-1, 1] always take the legacy path
        assert normalize_valence(1.0) == 1.0

    def test_normalize_arousal_legacy_range(self):
        assert normalize_arousal(-1.0) == 0.0
        assert normalize_arousal(0.0) == 0.5
        assert normalize_arousal(1.0) == 1.0

    def test_normalize_arousal_nine_range(self):
        assert normalize_arousal(5.0) == 0.5
        assert normalize_arousal(9.0) == 1.0
        # Values in [-1, 1] always take the legacy path
        assert normalize_arousal(1.0) == 1.0


class TestEuclideanDistance:
    def test_zero_distance(self):
        m = EuclideanDistance()
        pt = np.array([0.5, 0.5])
        assert m.distance(pt, pt) == 0.0

    def test_unit_distance(self):
        m = EuclideanDistance()
        d = m.distance(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        assert d == 1.0

    def test_diagonal_distance(self):
        m = EuclideanDistance()
        d = m.distance(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert abs(d - np.sqrt(2)) < 1e-12

    def test_symmetric(self):
        m = EuclideanDistance()
        a, b = np.array([0.2, 0.7]), np.array([0.8, 0.3])
        assert abs(m.distance(a, b) - m.distance(b, a)) < 1e-12


class TestManhattanDistance:
    def test_zero_distance(self):
        m = ManhattanDistance()
        pt = np.array([0.5, 0.5])
        assert m.distance(pt, pt) == 0.0

    def test_cardinal_distance(self):
        m = ManhattanDistance()
        d = m.distance(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        assert d == 1.0

    def test_diagonal_distance(self):
        m = ManhattanDistance()
        d = m.distance(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert d == 2.0

    def test_symmetric(self):
        m = ManhattanDistance()
        a, b = np.array([0.2, 0.7]), np.array([0.8, 0.3])
        assert abs(m.distance(a, b) - m.distance(b, a)) < 1e-12


class TestChebyshevDistance:
    def test_zero_distance(self):
        m = ChebyshevDistance()
        pt = np.array([0.5, 0.5])
        assert m.distance(pt, pt) == 0.0

    def test_max_dimension_dominated(self):
        m = ChebyshevDistance()
        d = m.distance(np.array([0.0, 0.0]), np.array([1.0, 5.0]))
        assert d == 5.0

    def test_symmetric(self):
        m = ChebyshevDistance()
        a, b = np.array([0.2, 0.7]), np.array([0.8, 0.3])
        assert abs(m.distance(a, b) - m.distance(b, a)) < 1e-12


class TestQuadraticDistance:
    def test_zero_distance(self):
        m = QuadraticDistance()
        pt = np.array([0.5, 0.5])
        assert m.distance(pt, pt) == 0.0

    def test_valence_weighted_more_than_arousal(self):
        m = QuadraticDistance()
        d_valence = m.distance(np.array([0.0, 0.0]), np.array([0.1, 0.0]))
        d_arousal = m.distance(np.array([0.0, 0.0]), np.array([0.0, 0.1]))
        assert d_valence > d_arousal

    def test_1d_fallback(self):
        m = QuadraticDistance()
        d = m.distance(np.array([0.0]), np.array([1.0]))
        assert d > 0

    def test_symmetric(self):
        m = QuadraticDistance()
        a, b = np.array([0.2, 0.7]), np.array([0.8, 0.3])
        assert abs(m.distance(a, b) - m.distance(b, a)) < 1e-12


class TestMahalanobisDistance:
    def test_zero_distance(self):
        m = MahalanobisDistance()
        pt = np.array([0.5, 0.5])
        assert m.distance(pt, pt) == 0.0

    def test_non_negative(self):
        m = MahalanobisDistance()
        d = m.distance(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert d >= 0
        assert isinstance(d, float)

    def test_1d_fallback(self):
        m = MahalanobisDistance()
        d = m.distance(np.array([0.0]), np.array([1.0]))
        assert d >= 0

    def test_symmetric(self):
        m = MahalanobisDistance()
        a, b = np.array([0.2, 0.7]), np.array([0.8, 0.3])
        assert abs(m.distance(a, b) - m.distance(b, a)) < 1e-12

    def test_custom_covariance(self):
        custom_cov = np.array([[0.1, 0.0], [0.0, 0.1]])
        m = MahalanobisDistance(cov_matrix=custom_cov)
        d = m.distance(np.array([0.0, 0.0]), np.array([0.5, 0.5]))
        assert d > 0


class TestMatchingEngineFactory:
    def test_get_euclidean(self):
        m = AffectiveMatchingEngine.get_metric(MatchingStrategy.EUCLIDEAN)
        assert isinstance(m, EuclideanDistance)

    def test_get_manhattan(self):
        m = AffectiveMatchingEngine.get_metric(MatchingStrategy.MANHATTAN)
        assert isinstance(m, ManhattanDistance)

    def test_get_chebyshev(self):
        m = AffectiveMatchingEngine.get_metric(MatchingStrategy.CHEBYSHEV)
        assert isinstance(m, ChebyshevDistance)

    def test_get_quadratic(self):
        m = AffectiveMatchingEngine.get_metric(MatchingStrategy.QUADRATIC)
        assert isinstance(m, QuadraticDistance)

    def test_get_mahalanobis(self):
        m = AffectiveMatchingEngine.get_metric(MatchingStrategy.MAHALANOBIS)
        assert isinstance(m, MahalanobisDistance)

    def test_get_strategy_info(self):
        info = AffectiveMatchingEngine.get_strategy_info()
        expected = {"euclidean", "manhattan", "chebyshev", "quadratic", "mahalanobis"}
        assert set(info.keys()) == expected

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown matching strategy"):
            AffectiveMatchingEngine.get_metric("nonexistent")


class TestComputeEffectiveAffect:
    def test_no_preference_no_shift(self):
        v, a = compute_effective_affect(
            normative_val=0.5, normative_aro=5.0, preference_score=None, match_arousal=True
        )
        assert v == 0.5
        assert a == 5.0

    def test_low_preference_shifts_down(self):
        v, a = compute_effective_affect(
            normative_val=0.5, normative_aro=5.0, preference_score=1.0, match_arousal=True
        )
        assert v < 0.5

    def test_high_preference_shifts_up(self):
        v, a = compute_effective_affect(
            normative_val=0.5, normative_aro=5.0, preference_score=5.0, match_arousal=True
        )
        assert v > 0.5

    def test_clamped_to_valid_range(self):
        v, a = compute_effective_affect(
            normative_val=8.0, normative_aro=6.0, preference_score=1.0, match_arousal=True
        )
        assert 1.0 <= v <= 9.0
        assert 1.0 <= a <= 9.0

    def test_match_arousal_false_leaves_arousal_unchanged(self):
        v, a = compute_effective_affect(
            normative_val=0.5, normative_aro=5.0, preference_score=5.0, match_arousal=False
        )
        assert v > 0.5
        assert a == 5.0

    def test_custom_likert_range(self):
        v, a = compute_effective_affect(
            normative_val=0.5,
            normative_aro=5.0,
            preference_score=3.0,
            match_arousal=True,
            likert_min=0.0,
            likert_max=10.0,
        )
        # midpoint = 5, half_range = 5; pref 3 → pref_norm = -0.4
        # valence shifts toward -1.0, arousal shifts toward 1.0
        assert abs(v - 0.2) < 1e-12
        assert abs(a - 4.2) < 1e-12

    def test_neutral_preference_no_shift(self):
        v, a = compute_effective_affect(
            normative_val=0.5, normative_aro=5.0, preference_score=3.0, match_arousal=True
        )
        assert v == 0.5
        assert a == 5.0

    def test_legacy_valence_scale_preserved(self):
        v, a = compute_effective_affect(
            normative_val=0.0, normative_aro=5.0, preference_score=1.0, match_arousal=True
        )
        assert v < 0.0  # should shift down from midpoint 0

    def test_nine_point_valence_scale(self):
        v, a = compute_effective_affect(
            normative_val=5.0, normative_aro=5.0, preference_score=1.0, match_arousal=True
        )
        # Should shift toward 1.0 (low end of 1-9 scale)
        assert v > 1.0  # not all the way, but shifted down
        assert v < 5.0
