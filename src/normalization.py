"""Normalization utilities for participant and blueprint payloads.

Shared between the CLI and the PlaylistGenerator to avoid
duplicating field-mapping logic.
"""

import csv
from pathlib import Path
from typing import Any


def resolve_phase_target_seconds(phase: dict[str, Any]) -> int:
    if "target_duration_seconds" in phase:
        return int(phase["target_duration_seconds"])

    if "target_duration_ms" in phase:
        return max(1, int(round(float(phase["target_duration_ms"]) / 1000.0)))

    if "duration_range_ms" in phase and isinstance(phase["duration_range_ms"], list):
        max_ms = phase["duration_range_ms"][1]
        return max(1, int(round(float(max_ms) / 1000.0)))

    if "max_duration_ms" in phase:
        return max(1, int(round(float(phase["max_duration_ms"]) / 1000.0)))

    if "min_duration_ms" in phase:
        return max(1, int(round(float(phase["min_duration_ms"]) / 1000.0)))

    raise ValueError(
        "Each phase must include one of: "
        "target_duration_seconds, target_duration_ms, "
        "duration_range_ms, max_duration_ms, or min_duration_ms"
    )


def resolve_phase_min_seconds(phase: dict[str, Any], target_seconds: int) -> int:
    if "min_duration_seconds" in phase:
        return max(1, int(phase["min_duration_seconds"]))

    if "min_duration_ms" in phase:
        return max(1, int(round(float(phase["min_duration_ms"]) / 1000.0)))

    if "duration_range_ms" in phase and isinstance(phase["duration_range_ms"], list):
        min_ms = phase["duration_range_ms"][0]
        return max(1, int(round(float(min_ms) / 1000.0)))

    return target_seconds


def normalize_participant_payload(payload: dict[str, Any]) -> dict[str, Any]:
    participant_id = payload.get("participant_id") or payload.get("id")
    if not participant_id:
        raise ValueError("Participant payload must include 'participant_id' or 'id'")

    familiar = payload.get("familiar_films")
    if familiar is None:
        familiar = payload.get("familiar_media", [])

    if not isinstance(familiar, list):
        familiar = []

    normalized = dict(payload)
    normalized["participant_id"] = str(participant_id)
    normalized["familiar_films"] = [str(item) for item in familiar]
    return normalized


def normalize_blueprint_payload(
    payload: dict[str, Any],
    model_id: str,
    default_clip_duration_seconds: int,
) -> dict[str, Any]:
    experiment_id = payload.get("experiment_id") or payload.get("id")
    if not experiment_id:
        raise ValueError("Blueprint payload must include 'experiment_id' or 'id'")

    phases = payload.get("phases", [])
    if not isinstance(phases, list) or not phases:
        raise ValueError("Blueprint payload must include at least one phase")

    normalized_phases: list[dict[str, Any]] = []
    for idx, phase in enumerate(phases, start=1):
        phase_id = phase.get("phase_id") or phase.get("phase_order") or f"phase-{idx}"
        target_duration_seconds = resolve_phase_target_seconds(phase)
        normalized_phase = {
            "phase_id": str(phase_id),
            "target_valence": float(phase["target_valence"]),
            "target_arousal": float(phase["target_arousal"]),
            "target_duration_seconds": target_duration_seconds,
            "min_duration_seconds": resolve_phase_min_seconds(phase, target_duration_seconds),
        }
        if "allowed_modalities" in phase:
            normalized_phase["allowed_modalities"] = coerce_modalities(
                phase.get("allowed_modalities")
            )
        normalized_phases.append(normalized_phase)

    return {
        "experiment_id": str(experiment_id),
        "valence_model": payload.get("valence_model", model_id),
        "default_clip_duration_seconds": int(
            payload.get(
                "default_clip_duration_seconds",
                default_clip_duration_seconds,
            )
        ),
        "duration_column_candidates": payload.get("duration_column_candidates", []),
        "phases": normalized_phases,
    }


def coerce_modalities(raw_value: Any) -> list[str]:
    import json

    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]

    text = str(raw_value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass

    return [segment.strip() for segment in text.split(",") if segment.strip()]


def write_phase_csv(playlist: dict[str, Any], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "participant_id",
                "experiment_id",
                "phase_id",
                "film",
                "duration_seconds",
                "target_duration_seconds",
                "valence",
                "arousal",
                "distance",
            ]
        )

        participant_id = playlist.get("participant_id", "")
        experiment_id = playlist.get("experiment_id", "")

        for phase in playlist.get("phases", []):
            phase_id = phase.get("phase_id", "")
            target_duration = phase.get("target_duration_seconds", "")
            for item in phase.get("selected_films", []):
                writer.writerow(
                    [
                        participant_id,
                        experiment_id,
                        phase_id,
                        item.get("film", ""),
                        item.get("duration_seconds", ""),
                        target_duration,
                        item.get("valence", ""),
                        item.get("arousal", ""),
                        item.get("distance", ""),
                    ]
                )
