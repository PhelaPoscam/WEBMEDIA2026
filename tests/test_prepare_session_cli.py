import json
from unittest.mock import patch

from src import models
from src.prepare_session_cli import export_blueprint_files, prepare_session


def test_prepare_session_from_questionnaire_and_blueprint(tmp_path):
    participant_xml = tmp_path / "participant.xml"
    participant_xml.write_text(
        """
<questionnaire>
  <participant_id>P-CLI-1</participant_id>
  <baseline_valence>0.0</baseline_valence>
  <baseline_arousal>0.0</baseline_arousal>
  <familiar_media>
    <item>Film C</item>
  </familiar_media>
</questionnaire>
""".strip(),
        encoding="utf-8",
    )

    blueprint_json = tmp_path / "blueprint.json"
    blueprint_json.write_text(
        json.dumps(
            {
                "id": "exp-cli-1",
                "phases": [
                    {
                        "phase_order": 1,
                        "target_valence": 0.1,
                        "target_arousal": 0.5,
                        "min_duration_ms": 60000,
                        "max_duration_ms": 120000,
                        "allowed_modalities": ["video"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        (
            "Film,Arousal mean,Model_C (Dominant),Duration_seconds\n"
            "Film A,4.0,0.10,60\n"
            "Film B,4.1,0.11,60\n"
            "Film C,3.9,0.09,60\n"
        ),
        encoding="utf-8",
    )

    output_json = tmp_path / "session_output.json"
    output_csv = tmp_path / "session_output.csv"

    result = prepare_session(
        participant_file=participant_xml,
        blueprint_file=blueprint_json,
        dataset_csv=dataset_csv,
        output_json=output_json,
        output_csv=output_csv,
        output_xml=None,
        output_yaml=None,
        model_id="Model_C",
        default_clip_duration_seconds=60,
    )

    assert result["participant_id"] == "P-CLI-1"
    assert result["experiment_id"] == "exp-cli-1"
    assert len(result["phases"]) == 1
    assert output_json.exists()
    assert output_csv.exists()

    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["participant_id"] == "P-CLI-1"


def test_export_blueprint_files_from_db(tmp_path, db_session):
    # Setup test data in the test database
    bp = models.ExperimentBlueprint(
        id="exp-cli-test", name="CLI Test", description="Test Description"
    )
    db_session.add(bp)

    phase = models.BlueprintPhase(
        blueprint_id="exp-cli-test",
        phase_order=1,
        target_valence=0.5,
        target_arousal=0.5,
        min_duration_ms=1000,
        max_duration_ms=2000,
        allowed_modalities=["video"],
    )
    db_session.add(phase)
    db_session.commit()

    output_dir = tmp_path / "exports"

    # Create a factory that behaves like SessionLocal
    class SessionFactory:
        def __init__(self, session):
            self.session = session

        def __call__(self):
            return self.session

    db_session_factory = SessionFactory(db_session)

    # Mock _load_db_dependencies to return our test db_session factory
    with patch("src.prepare_session_cli._load_db_dependencies") as mock_load:
        mock_load.return_value = (db_session_factory, models)
        result = export_blueprint_files(
            output_dir=output_dir, blueprint_id="exp-cli-test", blueprint_name=None
        )

    assert "exp-cli-test" in result["json"]
    assert output_dir.exists()
    assert (output_dir / "exp-cli-test.json").exists()

    # Verify content
    exported = json.loads((output_dir / "exp-cli-test.json").read_text())
    assert exported["id"] == "exp-cli-test"
    assert len(exported["phases"]) == 1
    assert exported["phases"][0]["target_valence"] == 0.5


def test_prepare_session_from_xml_blueprint(tmp_path):
    participant_xml = tmp_path / "participant.xml"
    participant_xml.write_text(
        """
<questionnaire>
  <participant_id>P-CLI-XML</participant_id>
  <baseline_valence>0.0</baseline_valence>
  <baseline_arousal>0.0</baseline_arousal>
  <familiar_media></familiar_media>
</questionnaire>
""".strip(),
        encoding="utf-8",
    )

    blueprint_xml = tmp_path / "blueprint.xml"
    blueprint_xml.write_text(
        """
<blueprint>
  <id>exp-cli-xml</id>
  <phases>
    <phase>
      <phase_order>1</phase_order>
      <target_valence>0.1</target_valence>
      <target_arousal>0.0</target_arousal>
      <min_duration_ms>60000</min_duration_ms>
      <max_duration_ms>120000</max_duration_ms>
      <allowed_modalities>
        <modality>video</modality>
      </allowed_modalities>
    </phase>
  </phases>
</blueprint>
""".strip(),
        encoding="utf-8",
    )

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        ("Film,Arousal mean,Model_C (Dominant),Duration_seconds\n" "Film A,4.0,0.10,60\n"),
        encoding="utf-8",
    )

    output_json = tmp_path / "session_output.json"
    output_csv = tmp_path / "session_output.csv"

    result = prepare_session(
        participant_file=participant_xml,
        blueprint_file=blueprint_xml,
        dataset_csv=dataset_csv,
        output_json=output_json,
        output_csv=output_csv,
        output_xml=None,
        output_yaml=None,
        model_id="Model_C",
        default_clip_duration_seconds=60,
    )

    assert result["participant_id"] == "P-CLI-XML"
    assert result["experiment_id"] == "exp-cli-xml"
    assert len(result["phases"]) == 1
    assert output_json.exists()
    assert output_csv.exists()


def test_prepare_session_with_psychopy_and_eprime(tmp_path):
    participant_json = tmp_path / "participant.json"
    participant_json.write_text(
        json.dumps(
            {
                "participant_id": "P-TEST-123",
                "familiar_media": [],
            }
        ),
        encoding="utf-8",
    )

    blueprint_json = tmp_path / "blueprint.json"
    blueprint_json.write_text(
        json.dumps(
            {
                "id": "exp-test-123",
                "phases": [
                    {
                        "phase_order": 1,
                        "target_valence": 0.5,
                        "target_arousal": 0.5,
                        "min_duration_ms": 60000,
                        "max_duration_ms": 60000,
                        "allowed_modalities": ["video"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "Film,Arousal mean,Model_C (Dominant),Duration_seconds\nFilm X,4.0,0.50,60\n",
        encoding="utf-8",
    )

    output_psychopy = tmp_path / "psychopy.csv"
    output_eprime = tmp_path / "eprime.txt"

    prepare_session(
        participant_file=participant_json,
        blueprint_file=blueprint_json,
        dataset_csv=dataset_csv,
        output_json=None,
        output_csv=None,
        output_xml=None,
        output_yaml=None,
        model_id="Model_C",
        default_clip_duration_seconds=60,
        output_psychopy=output_psychopy,
        output_eprime=output_eprime,
    )

    assert output_psychopy.exists()
    assert output_eprime.exists()

    psychopy_text = output_psychopy.read_text(encoding="utf-8")
    assert "Trial,Phase,Film,Code,Target_Valence,Target_Arousal" in psychopy_text
    assert "Film X" in psychopy_text

    eprime_text = output_eprime.read_text(encoding="utf-8")
    assert "Weight\tNested\tProcedure\ttrial_id\tphase_id\tstimulus_id" in eprime_text
    assert "Film X" in eprime_text


def test_prepare_session_preserves_blueprint_constraints_and_export_targets(tmp_path):
    participant_json = tmp_path / "participant.json"
    participant_json.write_text(
        json.dumps({"participant_id": "P-CONSTRAINTS", "familiar_media": []}),
        encoding="utf-8",
    )

    blueprint_json = tmp_path / "blueprint.json"
    blueprint_json.write_text(
        json.dumps(
            {
                "id": "exp-constraints",
                "valence_model": "Model_C",
                "phases": [
                    {
                        "phase_order": 1,
                        "target_valence": 0.8,
                        "target_arousal": 0.75,
                        "min_duration_ms": 60000,
                        "max_duration_ms": 120000,
                        "allowed_modalities": ["audio"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        (
            "Film,Arousal mean,Model_C (Dominant),Duration_seconds,Modality\n"
            "Video Candidate,0.75,0.8,60,video\n"
            "Audio Candidate,0.75,0.8,60,audio\n"
        ),
        encoding="utf-8",
    )

    output_psychopy = tmp_path / "psychopy.csv"
    result = prepare_session(
        participant_file=participant_json,
        blueprint_file=blueprint_json,
        dataset_csv=dataset_csv,
        output_json=None,
        output_csv=None,
        output_xml=None,
        output_yaml=None,
        model_id="Model_C",
        default_clip_duration_seconds=60,
        output_psychopy=output_psychopy,
    )

    phase = result["phases"][0]
    assert phase["target_valence"] == 0.8
    assert phase["target_arousal"] == 0.75
    assert phase["target_duration_seconds"] == 120
    assert phase["selected_films"][0]["film"] == "Audio Candidate"

    psychopy_text = output_psychopy.read_text(encoding="utf-8")
    assert "Audio Candidate" in psychopy_text
    assert "Video Candidate" not in psychopy_text
    assert ",0.8,0.75," in psychopy_text

    selected = phase["selected_films"][0]
    assert selected["uri"] == "csv://Audio Candidate"
    assert selected["modality"] == "audio"
