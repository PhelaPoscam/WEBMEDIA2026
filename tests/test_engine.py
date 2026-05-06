import pytest

from src.affective_matching import AffectiveMatchingEngine
from src.engine import InsufficientStimuliError, MatchEngine
from src.models import ParticipantProfile, StimulusMetadata
from src.schemas import ExperimentConstraint, ExperimentPhase


def test_engine_returns_valid_playlist(db_session):
    # Setup test data
    participant = ParticipantProfile(
        id="p1",
        baseline_valence=0.0,
        baseline_arousal=0.0,
        phobias=["trypophobia"],
        familiar_media=["s1"],
        preferences={},
    )
    db_session.add(participant)

    # Excluded due to familiarity
    s1 = StimulusMetadata(
        id="s1",
        uri="url1",
        modality="visual",
        content_tags=[],
        normative_valence=1.0,
        normative_arousal=1.0,
        duration_ms=1000,
    )
    # Excluded due to phobia
    s2 = StimulusMetadata(
        id="s2",
        uri="url2",
        modality="visual",
        content_tags=["trypophobia"],
        normative_valence=1.0,
        normative_arousal=1.0,
        duration_ms=1000,
    )
    # Valid, perfectly matches target
    s3 = StimulusMetadata(
        id="s3",
        uri="url3",
        modality="visual",
        content_tags=[],
        normative_valence=1.0,
        normative_arousal=1.0,
        duration_ms=15000,
    )
    # Valid, but far from target
    s4 = StimulusMetadata(
        id="s4",
        uri="url4",
        modality="visual",
        content_tags=[],
        normative_valence=-1.0,
        normative_arousal=-1.0,
        duration_ms=10000,
    )

    db_session.add_all([s1, s2, s3, s4])
    db_session.commit()

    # Constraint
    constraint = ExperimentConstraint(
        experiment_id="exp1",
        allowed_modalities=["visual"],
        phases=[
            ExperimentPhase(
                phase_id=1,
                target_valence=1.0,
                target_arousal=1.0,
                duration_range_ms=[10000, 20000],
                allowed_modalities=["visual"],
                requires_novelty=True,
            )
        ],
    )

    # Run Engine
    engine = MatchEngine(db=db_session, participant_id="p1")
    playlist = engine.generate_playlist(constraint)

    assert len(playlist["phases"]) == 1
    phase_data = playlist["phases"][0]

    # Should only return s3, as s1 is familiar, s2 is phobia, s4 is far
    # but s3 already fulfills minimum duration (15000 >= 10000)
    # Actually greedy algorithm picks s3 first. If it meets min dur, it stops.
    assert len(phase_data["selected_stimuli"]) == 1
    assert phase_data["selected_stimuli"][0]["stimulus_id"] == "s3"
    assert phase_data["total_duration_ms"] == 15000


def test_effective_affect_preference_shift(db_session):
    participant = ParticipantProfile(
        id="p2",
        baseline_valence=0.0,
        baseline_arousal=0.0,
    )
    db_session.add(participant)
    db_session.commit()

    engine = MatchEngine(db=db_session, participant_id="p2")

    normative_val, normative_aro = 8.0, 6.0

    # Strong dislike should push affect toward the low end (1.0)
    disliked_val, disliked_aro = engine._get_effective_affect(
        normative_val,
        normative_aro,
        preference_score=1.0,
    )

    # Strong liking should push affect toward the high end (9.0)
    loved_val, loved_aro = engine._get_effective_affect(
        normative_val,
        normative_aro,
        preference_score=5.0,
    )

    assert disliked_val < normative_val
    assert loved_val > normative_val
    assert abs(disliked_val - 1.0) < abs(normative_val - 1.0)
    assert abs(loved_val - 9.0) < abs(normative_val - 9.0)

    # Coordinates are always clamped into natural 1-9 bounds.
    assert 1.0 <= disliked_val <= 9.0
    assert 1.0 <= disliked_aro <= 9.0
    assert 1.0 <= loved_val <= 9.0
    assert 1.0 <= loved_aro <= 9.0


def test_engine_raises_insufficient_stimuli(db_session):
    # Setup participant
    participant = ParticipantProfile(id="p3", baseline_valence=0.0, baseline_arousal=0.0)
    db_session.add(participant)

    # Only 5 seconds of stimuli
    s1 = StimulusMetadata(
        id="s1",
        uri="url1",
        modality="visual",
        content_tags=[],
        normative_valence=1.0,
        normative_arousal=1.0,
        duration_ms=5000,
    )
    db_session.add(s1)
    db_session.commit()

    # Request 10-20 seconds
    constraint = ExperimentConstraint(
        experiment_id="exp_fail",
        allowed_modalities=["visual"],
        phases=[
            ExperimentPhase(
                phase_id=1,
                target_valence=1.0,
                target_arousal=1.0,
                duration_range_ms=[10000, 20000],
                allowed_modalities=["visual"],
            )
        ],
    )

    engine = MatchEngine(db=db_session, participant_id="p3")
    with pytest.raises(InsufficientStimuliError) as excinfo:
        engine.generate_playlist(constraint)

    assert "Insufficient stimuli" in str(excinfo.value)
    assert excinfo.value.missing_duration_ms == 5000


def test_engine_enforces_uniqueness(db_session):
    # Setup participant
    participant = ParticipantProfile(id="p4", baseline_valence=0.0, baseline_arousal=0.0)
    db_session.add(participant)

    # Stimuli
    s1 = StimulusMetadata(
        id="s1",
        uri="url1",
        modality="visual",
        content_tags=[],
        normative_valence=1.0,
        normative_arousal=1.0,
        duration_ms=10000,
    )
    s2 = StimulusMetadata(
        id="s2",
        uri="url2",
        modality="visual",
        content_tags=[],
        normative_valence=1.0,
        normative_arousal=1.0,
        duration_ms=10000,
    )
    db_session.add_all([s1, s2])
    db_session.commit()

    # Request two phases, each 10s. If s1 is used in Phase 1, only s2 can be in Phase 2.
    constraint = ExperimentConstraint(
        experiment_id="exp_unique",
        allowed_modalities=["visual"],
        phases=[
            ExperimentPhase(
                phase_id=1,
                target_valence=1.0,
                target_arousal=1.0,
                duration_range_ms=[10000, 10000],
                allowed_modalities=["visual"],
            ),
            ExperimentPhase(
                phase_id=2,
                target_valence=1.0,
                target_arousal=1.0,
                duration_range_ms=[10000, 10000],
                allowed_modalities=["visual"],
            ),
        ],
    )

    engine = MatchEngine(db=db_session, participant_id="p4")
    playlist = engine.generate_playlist(constraint)

    p1_stim = playlist["phases"][0]["selected_stimuli"][0]["stimulus_id"]
    p2_stim = playlist["phases"][1]["selected_stimuli"][0]["stimulus_id"]

    assert p1_stim != p2_stim
    assert {p1_stim, p2_stim} == {"s1", "s2"}


def test_engine_match_arousal_false_with_quadratic_and_mahalanobis(db_session):
    # Setup test data
    participant = ParticipantProfile(
        id="p5",
        baseline_valence=0.0,
        baseline_arousal=0.0,
        phobias=[],
        familiar_media=[],
        preferences={},
    )
    db_session.add(participant)

    s1 = StimulusMetadata(
        id="s1",
        uri="url1",
        modality="visual",
        content_tags=[],
        normative_valence=0.5,
        normative_arousal=4.0,
        duration_ms=10000,
    )
    db_session.add(s1)
    db_session.commit()

    constraint = ExperimentConstraint(
        experiment_id="exp_arousal_false",
        allowed_modalities=["visual"],
        phases=[
            ExperimentPhase(
                phase_id=1,
                target_valence=0.5,
                target_arousal=0.0,  # Ignored when match_arousal=False
                duration_range_ms=[10000, 10000],
                allowed_modalities=["visual"],
            )
        ],
    )

    # Test with quadratic strategy
    engine_quad = MatchEngine(db=db_session, participant_id="p5")
    engine_quad.distance_metric = AffectiveMatchingEngine.get_metric("quadratic")
    playlist_quad = engine_quad.generate_playlist(constraint, match_arousal=False)
    assert len(playlist_quad["phases"][0]["selected_stimuli"]) == 1

    # Test with mahalanobis strategy
    engine_mahal = MatchEngine(db=db_session, participant_id="p5")
    engine_mahal.distance_metric = AffectiveMatchingEngine.get_metric("mahalanobis")
    playlist_mahal = engine_mahal.generate_playlist(constraint, match_arousal=False)
    assert len(playlist_mahal["phases"][0]["selected_stimuli"]) == 1


def test_engine_transparency_report(db_session):
    participant = ParticipantProfile(
        id="p_report",
        baseline_valence=0.0,
        baseline_arousal=0.0,
        phobias=["trypophobia"],
        familiar_media=["s1"],
        preferences={},
    )
    db_session.add(participant)

    s1 = StimulusMetadata(
        id="s1",
        uri="url1",
        modality="visual",
        content_tags=[],
        normative_valence=1.0,
        normative_arousal=1.0,
        duration_ms=10000,
    )
    s2 = StimulusMetadata(
        id="s2",
        uri="url2",
        modality="visual",
        content_tags=["trypophobia"],
        normative_valence=1.0,
        normative_arousal=1.0,
        duration_ms=10000,
    )
    s3 = StimulusMetadata(
        id="s3",
        uri="url3",
        modality="visual",
        content_tags=[],
        normative_valence=1.0,
        normative_arousal=1.0,
        duration_ms=15000,
    )
    db_session.add_all([s1, s2, s3])
    db_session.commit()

    constraint = ExperimentConstraint(
        experiment_id="exp_report",
        allowed_modalities=["visual"],
        phases=[
            ExperimentPhase(
                phase_id=1,
                target_valence=1.0,
                target_arousal=1.0,
                duration_range_ms=[10000, 20000],
                allowed_modalities=["visual"],
            )
        ],
    )

    engine = MatchEngine(db=db_session, participant_id="p_report")
    playlist = engine.generate_playlist(constraint)

    assert "transparency_report" in playlist
    report = playlist["transparency_report"]
    assert report["run_metadata"]["valence_column"] == "normative_valence"
    assert report["run_metadata"]["input_pool_size"] == 3

    exclusions = report["exclusions"]
    excluded_ids = {e["stimulus_id"] for e in exclusions}
    assert "s1" in excluded_ids  # familiar
    assert "s2" in excluded_ids  # phobia

    assert len(report["phases"]) == 1
    phase_audit = report["phases"][0]
    assert phase_audit["phase_id"] == 1
    assert len(phase_audit["selections"]) == 1
    assert phase_audit["selections"][0]["stimulus_id"] == "s3"
