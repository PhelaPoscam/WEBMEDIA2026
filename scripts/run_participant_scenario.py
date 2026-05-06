import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure isolated scenario DB and disable auth for this scripted demo.
os.environ["SECURITY_ENABLED"] = "false"
os.environ["DATABASE_URL"] = "sqlite:///./scratch/participant_scenario.db"

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.db import Base, engine  # noqa: E402


def assert_ok(response, action: str) -> None:
    if response.status_code != 200:
        raise RuntimeError(f"{action} failed: {response.status_code} -> {response.text}")


def seed_stimuli(client: TestClient) -> None:
    stimuli = [
        {
            "id": "movie_001",
            "uri": "https://example.org/clips/movie_001.mp4",
            "modality": "video",
            "content_tags": ["calm", "nature"],
            "normative_valence": 0.0,
            "normative_arousal": 0.0,
            "duration_ms": 60000,
        },
        {
            "id": "movie_002",
            "uri": "https://example.org/clips/movie_002.mp4",
            "modality": "video",
            "content_tags": ["pleasant"],
            "normative_valence": 0.7,
            "normative_arousal": 0.2,
            "duration_ms": 60000,
        },
        {
            "id": "movie_003",
            "uri": "https://example.org/clips/movie_003.mp4",
            "modality": "video",
            "content_tags": ["neutral"],
            "normative_valence": 0.1,
            "normative_arousal": 0.1,
            "duration_ms": 60000,
        },
        {
            "id": "movie_004",
            "uri": "https://example.org/clips/movie_004.mp4",
            "modality": "video",
            "content_tags": ["pleasant", "music"],
            "normative_valence": 0.8,
            "normative_arousal": 0.3,
            "duration_ms": 60000,
        },
        {
            "id": "movie_005",
            "uri": "https://example.org/clips/movie_005.mp4",
            "modality": "video",
            "content_tags": ["stress", "action"],
            "normative_valence": -0.8,
            "normative_arousal": 0.8,
            "duration_ms": 60000,
        },
        {
            "id": "movie_006",
            "uri": "https://example.org/clips/movie_006.mp4",
            "modality": "video",
            "content_tags": ["stress", "snakes"],
            "normative_valence": -0.7,
            "normative_arousal": 0.9,
            "duration_ms": 60000,
        },
        {
            "id": "movie_007",
            "uri": "https://example.org/clips/movie_007.mp4",
            "modality": "video",
            "content_tags": ["neutral", "dialogue"],
            "normative_valence": 0.0,
            "normative_arousal": 0.2,
            "duration_ms": 60000,
        },
        {
            "id": "movie_008",
            "uri": "https://example.org/clips/movie_008.mp4",
            "modality": "video",
            "content_tags": ["pleasant", "comedy"],
            "normative_valence": 0.9,
            "normative_arousal": 0.4,
            "duration_ms": 60000,
        },
        {
            "id": "movie_009",
            "uri": "https://example.org/clips/movie_009.mp4",
            "modality": "video",
            "content_tags": ["stress", "dark"],
            "normative_valence": -0.9,
            "normative_arousal": 0.7,
            "duration_ms": 60000,
        },
    ]

    for stim in stimuli:
        response = client.post("/api/stimulus", json=stim)
        assert_ok(response, f"seed stimulus {stim['id']}")


def create_blueprint(client: TestClient) -> str:
    blueprint = {
        "id": "bp-participant-sim-001",
        "name": "Participant Simulation Blueprint",
        "description": "3-phase session for demo scenario",
        "phases": [
            {
                "phase_order": 1,
                "target_valence": 0.0,
                "target_arousal": 0.1,
                "min_duration_ms": 120000,
                "max_duration_ms": 180000,
                "allowed_modalities": ["video"],
            },
            {
                "phase_order": 2,
                "target_valence": 0.8,
                "target_arousal": 0.3,
                "min_duration_ms": 120000,
                "max_duration_ms": 180000,
                "allowed_modalities": ["video"],
            },
            {
                "phase_order": 3,
                "target_valence": -0.8,
                "target_arousal": 0.8,
                "min_duration_ms": 120000,
                "max_duration_ms": 180000,
                "allowed_modalities": ["video"],
            },
        ],
    }
    response = client.post("/api/blueprints", json=blueprint)
    assert_ok(response, "create blueprint")
    return blueprint["id"]


def run_scenario() -> dict:
    scenario_db = Path("scratch/participant_scenario.db")
    if scenario_db.exists():
        scenario_db.unlink()

    get_settings.cache_clear()
    Base.metadata.create_all(bind=engine)

    client = TestClient(app)

    assert_ok(client.get("/health"), "health check")
    assert_ok(client.get("/api/health"), "api health check")

    seed_stimuli(client)
    blueprint_id = create_blueprint(client)

    participant_questionnaire = {
        "participant_id": "P-SCENARIO-001",
        "baseline_valence": 0.2,
        "baseline_arousal": 0.1,
        "familiar_media": ["movie_001", "movie_003"],
        "phobias": ["snakes"],
        "preferences": {
            "movie_008": 5,
            "movie_005": 1,
        },
    }

    files = {
        "file": (
            "participant_questionnaire.json",
            json.dumps(participant_questionnaire),
            "application/json",
        )
    }
    upload_response = client.post(
        f"/api/participant/upload-questionnaire?blueprint_id={blueprint_id}",
        files=files,
    )
    assert_ok(upload_response, "upload questionnaire and auto-generate playlist")

    return {
        "questionnaire_submitted": participant_questionnaire,
        "assembly_output": upload_response.json(),
    }


if __name__ == "__main__":
    output = run_scenario()
    print(json.dumps(output, indent=2))
