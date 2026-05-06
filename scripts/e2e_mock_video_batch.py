import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Configure env before importing app/database modules.
os.environ["SECURITY_ENABLED"] = "false"
os.environ["DATABASE_URL"] = "sqlite:///./scratch/mock_batch_test.db"

from fastapi.testclient import TestClient
from main import app

from src.config import get_settings
from src.db import Base, engine


def make_constraint(experiment_id: str, target_valence: float, target_arousal: float):
    return {
        "experiment_id": experiment_id,
        "phases": [
            {
                "phase_id": 1,
                "target_valence": target_valence,
                "target_arousal": target_arousal,
                "duration_range_ms": [600000, 600000],
                "allowed_modalities": ["video"],
                "requires_novelty": True,
            }
        ],
    }


def build_stimuli(category: str, target_val: float, target_aro: float, count: int = 12):
    rows = []
    for idx in range(1, count + 1):
        # Small deterministic offsets keep rankings stable while still realistic.
        offset_v = ((idx % 4) - 1.5) * 0.04
        offset_a = ((idx % 3) - 1.0) * 0.05
        val = max(-1.0, min(1.0, target_val + offset_v))
        aro = max(-1.0, min(1.0, target_aro + offset_a))
        rows.append(
            {
                "id": f"{category[:3]}-vid-{idx:02d}",
                "uri": f"https://mock.local/{category}/video_{idx:02d}.mp4",
                "modality": "video",
                "content_tags": [category],
                "normative_valence": round(val, 3),
                "normative_arousal": round(aro, 3),
                "duration_ms": 60000,
            }
        )
    return rows


def assert_ok(resp, action: str):
    if resp.status_code != 200:
        raise RuntimeError(f"{action} failed: {resp.status_code} -> {resp.text}")


def main():
    out_dir = Path("scratch/mock_batch_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path("scratch/mock_batch_test.db")
    if db_path.exists():
        db_path.unlink()

    # Ensure settings cache reflects environment overrides for this process.
    get_settings.cache_clear()
    Base.metadata.create_all(bind=engine)

    client = TestClient(app)

    assert_ok(client.get("/health"), "GET /health")
    assert_ok(client.get("/ready"), "GET /ready")
    assert_ok(client.get("/api/health"), "GET /api/health")

    pool = []
    pool.extend(build_stimuli("pleasant", target_val=0.8, target_aro=0.2))
    pool.extend(build_stimuli("stress", target_val=-0.8, target_aro=0.8))
    pool.extend(build_stimuli("neutral", target_val=0.0, target_aro=0.0))

    for stim in pool:
        assert_ok(client.post("/api/stimulus", json=stim), f"POST /api/stimulus {stim['id']}")

    participants = {
        "pleasant": {
            "id": "p-mock-pleasant",
            "baseline_valence": 0.0,
            "baseline_arousal": 0.0,
            "phobias": ["stress", "neutral"],
            "familiar_media": [],
            "preferences": {},
        },
        "stress": {
            "id": "p-mock-stress",
            "baseline_valence": 0.0,
            "baseline_arousal": 0.0,
            "phobias": ["pleasant", "neutral"],
            "familiar_media": [],
            "preferences": {},
        },
        "neutral": {
            "id": "p-mock-neutral",
            "baseline_valence": 0.0,
            "baseline_arousal": 0.0,
            "phobias": ["pleasant", "stress"],
            "familiar_media": [],
            "preferences": {},
        },
    }
    for p in participants.values():
        assert_ok(client.post("/api/participant", json=p), f"POST /api/participant {p['id']}")

    scenarios = [
        (
            "pleasant",
            participants["pleasant"]["id"],
            make_constraint("exp-pleasant-10min", 0.8, 0.2),
        ),
        (
            "stress",
            participants["stress"]["id"],
            make_constraint("exp-stress-10min", -0.8, 0.8),
        ),
        (
            "neutral",
            participants["neutral"]["id"],
            make_constraint("exp-neutral-10min", 0.0, 0.0),
        ),
    ]

    summary = {
        "participant_ids": {k: v["id"] for k, v in participants.items()},
        "seeded_stimuli": len(pool),
        "results": [],
    }

    for name, participant_id, constraint in scenarios:
        gen_resp = client.post(
            f"/api/generate-playlist?participant_id={participant_id}",
            json=constraint,
        )
        assert_ok(gen_resp, f"POST /api/generate-playlist ({name})")
        playlist = gen_resp.json()

        phase = playlist["phases"][0]
        selected = phase["selected_stimuli"]
        total_duration = phase["total_duration_ms"]
        if total_duration != 600000:
            raise RuntimeError(
                f"{name} duration check failed: expected 600000, got {total_duration}"
            )

        json_resp = client.post(
            f"/api/export-playlist/json?participant_id={participant_id}",
            json=constraint,
        )
        assert_ok(json_resp, f"POST /api/export-playlist/json ({name})")

        csv_resp = client.post(
            f"/api/export-playlist/csv?participant_id={participant_id}",
            json=constraint,
        )
        assert_ok(csv_resp, f"POST /api/export-playlist/csv ({name})")

        json_path = out_dir / f"{name}_playlist.json"
        csv_path = out_dir / f"{name}_playlist.csv"
        json_path.write_text(json.dumps(json_resp.json(), indent=2), encoding="utf-8")
        csv_path.write_text(csv_resp.text, encoding="utf-8")

        summary["results"].append(
            {
                "set": name,
                "participant_id": participant_id,
                "selected_count": len(selected),
                "total_duration_ms": total_duration,
                "first_ids": [s["stimulus_id"] for s in selected[:3]],
                "json_output": str(json_path),
                "csv_output": str(csv_path),
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("E2E mock batch run completed successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
