import pandas as pd
import pytest

from src.macro_pipeline import PlaylistGenerator


def test_playlist_generator_init():
    df = pd.DataFrame(
        {
            "Film": ["Movie 1", "Movie 2"],
            "Arousal mean": [5.0, 6.0],
            "Model_C (Dominant)": [0.1, -0.2],
            "Duration_seconds": [60, 120],
        }
    )
    generator = PlaylistGenerator(df, matching_strategy="euclidean")
    assert generator.matching_strategy == "euclidean"
    assert len(generator.df) == 2


def test_playlist_generator_invalid_strategy():
    df = pd.DataFrame({"Film": [], "Arousal mean": []})
    with pytest.raises(ValueError, match="Invalid matching strategy"):
        PlaylistGenerator(df, matching_strategy="invalid_strategy")


def test_playlist_generator_missing_columns():
    df = pd.DataFrame({"WrongColumn": []})
    with pytest.raises(ValueError, match="CSV is missing required columns"):
        PlaylistGenerator(df)


def test_playlist_generator_rest_phase():
    df = pd.DataFrame(
        {
            "Film": ["Movie 1"],
            "Arousal mean": [5.0],
            "Model_C (Dominant)": [0.1],
            "Duration_seconds": [60],
        }
    )
    generator = PlaylistGenerator(df, matching_strategy="euclidean")
    participant = {"participant_id": "P-TEST", "familiar_films": []}
    blueprint = {
        "experiment_id": "EXP-REST",
        "valence_model": "Model_C",
        "phases": [{"phase_id": "REST_1", "target_duration_seconds": 120}],
    }
    playlist = generator.generate_playlist(participant, blueprint)
    assert len(playlist["phases"]) == 1
    phase = playlist["phases"][0]
    assert phase["phase_id"] == "REST_1"
    assert phase["achieved_duration_seconds"] == 120
    assert len(phase["selected_films"]) == 1
    assert phase["selected_films"][0]["film"] == "REST_PERIOD"
    assert phase["selected_films"][0]["duration_seconds"] == 120


def test_playlist_generator_normalization():
    df = pd.DataFrame(
        {
            "Film": ["Movie 1", "Movie 2"],
            "Arousal mean": [9.0, 1.0],
            "Model_C (Dominant)": [0.9, -0.9],
            "Duration_seconds": [60, 60],
        }
    )
    generator = PlaylistGenerator(df, matching_strategy="euclidean")
    participant = {"participant_id": "P-TEST", "familiar_films": []}
    blueprint = {
        "experiment_id": "EXP-NORM",
        "valence_model": "Model_C",
        "phases": [
            {
                "phase_id": "Phase_1",
                "target_valence": 0.9,
                "target_arousal": 9.0,
                "target_duration_seconds": 60,
            }
        ],
    }
    playlist = generator.generate_playlist(participant, blueprint)
    phase = playlist["phases"][0]
    assert phase["selected_films"][0]["film"] == "Movie 1"


def test_alignment_match_engine_and_playlist_generator(db_session):
    df = pd.DataFrame(
        {
            "Film": ["Movie A", "Movie B", "Movie C"],
            "Arousal mean": [3.0, 7.0, 5.0],
            "Model_C (Dominant)": [-0.5, 0.8, 0.1],
            "Duration_seconds": [60, 90, 120],
        }
    )

    from src.engine import MatchEngine
    from src.models import ParticipantProfile, StimulusMetadata
    from src.schemas import ExperimentConstraint, ExperimentPhase

    for film, arousal, valence, duration in zip(
        df["Film"], df["Arousal mean"], df["Model_C (Dominant)"], df["Duration_seconds"]
    ):
        s = StimulusMetadata(
            id=film,
            uri=f"csv://{film}",
            modality="video",
            normative_valence=valence,
            normative_arousal=arousal,
            duration_ms=duration * 1000,
            content_tags=[],
        )
        db_session.add(s)

    p = ParticipantProfile(id="p_align", familiar_media=[], phobias=[])
    db_session.add(p)
    db_session.commit()

    # DB MatchEngine run
    engine = MatchEngine(db=db_session, participant_id="p_align")
    constraint = ExperimentConstraint(
        experiment_id="align_exp",
        allowed_modalities=["video"],
        phases=[
            ExperimentPhase(
                phase_id="Phase_1",
                target_valence=0.0,
                target_arousal=0.5,
                duration_range_ms=[150000, 150000],
                allowed_modalities=["video"],
            )
        ],
    )
    db_playlist = engine.generate_playlist(constraint)

    # CSV PlaylistGenerator run
    generator = PlaylistGenerator(df, matching_strategy="euclidean")
    participant = {"participant_id": "p_align", "familiar_films": []}
    blueprint = {
        "experiment_id": "align_exp",
        "valence_model": "Model_C",
        "phases": [
            {
                "phase_id": "Phase_1",
                "target_valence": 0.0,
                "target_arousal": 0.5,
                "target_duration_seconds": 150,
            }
        ],
    }
    csv_playlist = generator.generate_playlist(participant, blueprint)

    db_films = [item["stimulus_id"] for item in db_playlist["phases"][0]["selected_stimuli"]]
    csv_films = [item["film"] for item in csv_playlist["phases"][0]["selected_films"]]

    assert db_films == csv_films
    assert len(db_films) > 0


def test_playlist_generator_exclusions_and_report():
    df = pd.DataFrame(
        {
            "Film": ["Movie A", "Movie B", "Movie C", "Movie D"],
            "Arousal mean": [5.0, 5.0, 5.0, 5.0],
            "Model_C (Dominant)": [0.0, 0.0, 0.0, 0.0],
            "Duration_seconds": [60, 60, 60, 60],
            "Content_Tags": ["spiders", "heights", "", "snakes,spiders"],
            "Modality": ["video", "audio", "video", "video"],
        }
    )

    generator = PlaylistGenerator(df, matching_strategy="euclidean")
    participant = {
        "participant_id": "P-EXCLUDE",
        "familiar_films": ["Movie C"],
        "phobias": ["spiders"],
    }
    blueprint = {
        "experiment_id": "EXP-EXCLUDE",
        "valence_model": "Model_C",
        "phases": [
            {
                "phase_id": "Phase_1",
                "target_valence": 0.0,
                "target_arousal": 5.0,
                "target_duration_seconds": 60,
                "allowed_modalities": ["video"],
            }
        ],
    }

    playlist = generator.generate_playlist(participant, blueprint)

    # Exclusions assertion
    report = playlist.get("transparency_report", {})
    assert report is not None
    exclusions = report.get("exclusions", [])

    # Excluded films should include:
    # Movie A (phobia: spiders)
    # Movie C (familiar: Movie C)
    # Movie D (phobia: snakes,spiders)
    # Movie B should NOT appear — modality is a phase-level filter, not a global exclusion
    excluded_ids = {e["stimulus_id"] for e in exclusions}
    assert "Movie A" in excluded_ids
    assert "Movie C" in excluded_ids
    assert "Movie D" in excluded_ids
    assert "Movie B" not in excluded_ids

    # Check phase selection: only Movie B, Movie A, C, D are excluded.
    # Let's verify that the transparency report contains selections and candidate details
    assert len(report.get("phases", [])) == 1
    phase_audit = report["phases"][0]
    assert phase_audit["phase_id"] == "Phase_1"

    # Let's write the transparency report to check it prints correctly
    from src.exporter import generate_transparency_report

    report_text = generate_transparency_report(playlist)
    assert "Run Metadata" in report_text
    assert "Exclusions Log" in report_text
    assert "Movie A" in report_text
    # Movie B (modality-constrained) no longer appears in global exclusions log
    assert "Movie B" not in report_text


def test_playlist_generator_honors_requested_valence_model():
    df = pd.DataFrame(
        {
            "Film": ["Model A Best", "Model C Best"],
            "Arousal mean": [5.0, 5.0],
            "Model_A (PANAS)": [0.9, -0.9],
            "Model_C (Dominant)": [-0.9, 0.9],
            "Duration_seconds": [60, 60],
        }
    )
    generator = PlaylistGenerator(df, matching_strategy="euclidean")
    participant = {"participant_id": "P-MODEL", "familiar_films": []}
    blueprint = {
        "experiment_id": "EXP-MODEL",
        "valence_model": "Model_A",
        "phases": [
            {
                "phase_id": "Phase_1",
                "target_valence": 0.9,
                "target_arousal": 5.0,
                "target_duration_seconds": 60,
            }
        ],
    }

    playlist = generator.generate_playlist(participant, blueprint)

    assert playlist["valence_model"] == "Model_A"
    assert playlist["valence_column"] == "Model_A (PANAS)"
    assert playlist["phases"][0]["selected_films"][0]["film"] == "Model A Best"


def test_playlist_generator_phase_payload_contains_targets():
    df = pd.DataFrame(
        {
            "Film": ["Movie 1"],
            "Arousal mean": [5.0],
            "Model_C (Dominant)": [0.25],
            "Duration_seconds": [60],
        }
    )
    generator = PlaylistGenerator(df, matching_strategy="euclidean")
    playlist = generator.generate_playlist(
        {"participant_id": "P-TARGETS", "familiar_films": []},
        {
            "experiment_id": "EXP-TARGETS",
            "valence_model": "Model_C",
            "phases": [
                {
                    "phase_id": "Phase_1",
                    "target_valence": 0.25,
                    "target_arousal": 5.0,
                    "target_duration_seconds": 60,
                }
            ],
        },
    )

    phase = playlist["phases"][0]
    assert phase["target_valence"] == 0.25
    assert phase["target_arousal"] == 5.0
