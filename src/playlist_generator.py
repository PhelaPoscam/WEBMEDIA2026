from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.affective_matching import (
    AffectiveMatchingEngine,
    MatchingStrategy,
    select_stimuli,
)
from src.data_source import (
    DEFAULT_DURATION_COLUMN_CANDIDATES,
    MODEL_TO_COLUMN,
    CsvStimulusSource,
    StimulusSource,
)
from src.data_source import load_master_csv as _load_master_csv

load_master_csv = _load_master_csv


@dataclass
class PhaseSelectionResult:
    phase_id: str
    target_valence: float
    target_arousal: float
    target_duration_seconds: int
    achieved_duration_seconds: int
    duration_gap_seconds: int
    selected_count: int
    status: str
    selected_films: list[dict]


class InsufficientStimuliError(Exception):
    def __init__(self, phase_id: str, missing_duration_ms: int):
        self.phase_id = phase_id
        self.missing_duration_ms = missing_duration_ms
        super().__init__(
            f"Insufficient stimuli for phase {phase_id}. Missing {missing_duration_ms}ms."
        )


class PlaylistGenerator:
    """Unified playlist generator supporting both CSV and DB stimulus sources."""

    def __init__(
        self,
        stimulus_source: StimulusSource | pd.DataFrame,
        matching_strategy: str = "euclidean",
        match_arousal: bool = True,
    ):
        if isinstance(stimulus_source, pd.DataFrame):
            self._dataframe_source = stimulus_source.copy()
            self._source = CsvStimulusSource(self._dataframe_source)
            self.df = stimulus_source.copy()
        else:
            self._dataframe_source = None
            self._source = stimulus_source
            self.df = pd.DataFrame()

        self.matching_strategy = matching_strategy
        self.match_arousal = match_arousal
        try:
            self.strategy = MatchingStrategy(matching_strategy)
            self.distance_metric = AffectiveMatchingEngine.get_metric(self.strategy)
        except ValueError as e:
            raise ValueError(f"Invalid matching strategy: {e}")

    @property
    def source_name(self) -> str:
        return self._source.source_name

    @staticmethod
    def _normalize_familiar_set(familiar_films: list[str]) -> set[str]:
        return {str(name).strip().casefold() for name in familiar_films if str(name).strip()}

    def generate_playlist(
        self,
        participant_payload: dict,
        experiment_blueprint: dict,
    ) -> dict:
        participant_id = participant_payload.get("participant_id", "unknown_participant")
        experiment_id = experiment_blueprint.get("experiment_id", "unknown_experiment")

        model_key = experiment_blueprint.get("valence_model", "Model_C")
        valence_column = self._resolve_valence_column_display(model_key)

        phases = experiment_blueprint.get("phases", [])
        if not phases:
            raise ValueError("Experiment blueprint must include at least one phase.")

        default_clip_duration_seconds = int(
            experiment_blueprint.get("default_clip_duration_seconds", 60)
        )
        if default_clip_duration_seconds <= 0:
            raise ValueError("default_clip_duration_seconds must be > 0.")

        duration_column_candidates = experiment_blueprint.get("duration_column_candidates")
        if not duration_column_candidates:
            duration_column_candidates = DEFAULT_DURATION_COLUMN_CANDIDATES
        if self._dataframe_source is not None:
            self._source = CsvStimulusSource(
                self._dataframe_source,
                valence_model_key=model_key,
                duration_column_candidates=duration_column_candidates,
                default_clip_duration_seconds=default_clip_duration_seconds,
            )

        familiar_set = self._normalize_familiar_set(participant_payload.get("familiar_films", []))
        phobias = [p.strip().lower() for p in participant_payload.get("phobias", []) if p.strip()]

        exclusions_log: list[dict] = []

        all_stimuli = list(self._source.iter_all())

        working_pool: list = []
        for s in all_stimuli:
            stim_id = str(s.id)

            if stim_id.lower() in familiar_set:
                exclusions_log.append({"stimulus_id": stim_id, "reason": "familiar_media"})
                continue

            if phobias:
                tags = [t.strip().lower() for t in (s.content_tags or [])]
                matching_phobias = [t for t in tags if t in phobias]
                if matching_phobias:
                    exclusions_log.append(
                        {
                            "stimulus_id": stim_id,
                            "reason": (
                                f"phobia_constraint "
                                f"(matches tags: {', '.join(matching_phobias)})"
                            ),
                        }
                    )
                    continue

            try:
                float(s.normative_valence)
                float(s.normative_arousal)
            except (ValueError, TypeError):
                exclusions_log.append(
                    {"stimulus_id": stim_id, "reason": "invalid_affective_metadata"}
                )
                continue

            duration_ms = int(s.duration_ms) if s.duration_ms else 0
            if duration_ms <= 0:
                exclusions_log.append({"stimulus_id": stim_id, "reason": "invalid_duration"})
                continue

            working_pool.append(s)

        output: dict[str, Any] = {
            "participant_id": participant_id,
            "experiment_id": experiment_id,
            "valence_model": model_key,
            "valence_column": valence_column,
            "matching_strategy": self.matching_strategy,
            "input_pool_size": int(len(all_stimuli)),
            "post_familiarity_pool_size": int(len(working_pool)),
            "phases": [],
            "unassigned_pool_size": 0,
        }

        transparency_report: dict[str, Any] = {
            "run_metadata": {
                "timestamp": datetime.now().isoformat(),
                "matching_strategy": self.matching_strategy,
                "valence_model": model_key,
                "valence_column": valence_column,
                "input_pool_size": len(all_stimuli),
            },
            "exclusions": [],
            "phases": [],
        }

        used_ids: set[str] = set()

        for phase in phases:
            phase_id = str(phase.get("phase_id", "unknown_phase"))
            target_duration_seconds = int(phase["target_duration_seconds"])
            min_duration_seconds = int(phase.get("min_duration_seconds", target_duration_seconds))

            if target_duration_seconds <= 0:
                raise ValueError(f"phase '{phase_id}' has invalid target_duration_seconds <= 0.")
            if min_duration_seconds <= 0 or min_duration_seconds > target_duration_seconds:
                min_duration_seconds = target_duration_seconds

            if phase_id.upper().startswith("REST"):
                phase_result = PhaseSelectionResult(
                    phase_id=phase_id,
                    target_valence=0.0,
                    target_arousal=0.0,
                    target_duration_seconds=target_duration_seconds,
                    achieved_duration_seconds=target_duration_seconds,
                    duration_gap_seconds=0,
                    selected_count=1,
                    status="exact_duration_match",
                    selected_films=[
                        {
                            "film": "REST_PERIOD",
                            "arousal": 0.0,
                            "valence": 0.0,
                            "distance": 0.0,
                            "duration_seconds": target_duration_seconds,
                        }
                    ],
                )
                output["phases"].append(phase_result.__dict__)

                phase_audit = {
                    "phase_id": phase_id,
                    "target_valence": 0.0,
                    "target_arousal": 0.0,
                    "achieved_valence": 0.0,
                    "achieved_arousal": 0.0,
                    "achieved_duration_seconds": target_duration_seconds,
                    "target_duration_seconds": target_duration_seconds,
                    "duration_gap_seconds": 0,
                    "candidate_pool_size": 0,
                    "selections": [
                        {
                            "stimulus_id": "REST_PERIOD",
                            "distance": 0.0,
                            "duration_ms": target_duration_seconds * 1000,
                            "selection_reason": "rest_period_static_assignment",
                        }
                    ],
                    "rejected_candidates": [],
                }
                transparency_report["phases"].append(phase_audit)
                continue

            target_valence = float(phase["target_valence"])
            target_arousal = float(phase["target_arousal"])

            allowed_modalities_raw = phase.get("allowed_modalities")
            allowed_set: set[str] | None = None
            if allowed_modalities_raw:
                allowed_set = {m.strip().lower() for m in allowed_modalities_raw if m.strip()}

            phase_pool = []
            for s in working_pool:
                if s.id in used_ids:
                    continue
                if allowed_set:
                    modality = getattr(s, "modality", "video")
                    if str(modality).strip().lower() not in allowed_set:
                        continue
                phase_pool.append(s)

            min_duration_ms = min_duration_seconds * 1000
            target_duration_ms = target_duration_seconds * 1000
            selected_in_phase, achieved_ms, phase_audit_data = select_stimuli(
                pool=phase_pool,
                target_valence=target_valence,
                target_arousal=target_arousal,
                min_duration_ms=min_duration_ms,
                max_duration_ms=target_duration_ms,
                distance_metric=self.distance_metric,
                match_arousal=self.match_arousal,
            )

            selected_rows = []
            for item in selected_in_phase:
                selected_rows.append(
                    {
                        "film": item["stimulus_id"],
                        "stimulus_id": item["stimulus_id"],
                        "stimulus_code": item.get("stimulus_code", "N/A"),
                        "uri": item.get("uri", ""),
                        "modality": item.get("modality", "unknown"),
                        "arousal": float(item["raw_arousal"]),
                        "valence": float(item["raw_valence"]),
                        "distance": round(float(item["distance"]), 6),
                        "duration_seconds": int(item["duration_ms"]) // 1000,
                        "duration_ms": int(item["duration_ms"]),
                    }
                )

            selected_film_ids = {item["stimulus_id"] for item in selected_in_phase}
            used_ids |= selected_film_ids

            achieved = achieved_ms // 1000
            gap = abs(target_duration_seconds - achieved)
            if achieved == 0:
                status = "no_valid_clips"
            elif achieved < target_duration_seconds:
                status = "partial_insufficient_pool"
            elif gap == 0:
                status = "exact_duration_match"
            else:
                status = "duration_overshoot"

            phase_result = PhaseSelectionResult(
                phase_id=phase_id,
                target_valence=target_valence,
                target_arousal=target_arousal,
                target_duration_seconds=target_duration_seconds,
                achieved_duration_seconds=achieved,
                duration_gap_seconds=gap,
                selected_count=len(selected_rows),
                status=status,
                selected_films=selected_rows,
            )
            output["phases"].append(phase_result.__dict__)

            if selected_in_phase:
                avg_valence = sum(item["eff_valence"] for item in selected_in_phase) / len(
                    selected_in_phase
                )
                avg_arousal = sum(item["eff_arousal"] for item in selected_in_phase) / len(
                    selected_in_phase
                )
            else:
                avg_valence = 0.0
                avg_arousal = 0.0

            phase_audit = {
                "phase_id": phase_id,
                "target_valence": target_valence,
                "target_arousal": target_arousal,
                "achieved_valence": avg_valence,
                "achieved_arousal": avg_arousal,
                "achieved_duration_seconds": achieved,
                "target_duration_seconds": target_duration_seconds,
                "duration_gap_seconds": gap if status == "duration_overshoot" else 0,
                "candidate_pool_size": phase_audit_data["candidate_pool_size"],
                "selections": phase_audit_data["selections"],
                "rejected_candidates": phase_audit_data["rejected_candidates"],
            }
            transparency_report["phases"].append(phase_audit)

        transparency_report["exclusions"] = exclusions_log
        output["transparency_report"] = transparency_report
        output["unassigned_pool_size"] = int(sum(1 for s in working_pool if s.id not in used_ids))
        return output

    def _resolve_valence_column_display(self, model_key: str) -> str:
        if model_key in MODEL_TO_COLUMN:
            return MODEL_TO_COLUMN[model_key]
        return model_key


MOCK_PARTICIPANT_PAYLOAD = {
    "participant_id": "P-0001",
    "age": 27,
    "gender": "female",
    "familiar_films": [
        "Ghost",
        "Forrest Gump",
        "The Exorcist",
        "E.T.",
        "The visitors",
    ],
}

MOCK_EXPERIMENT_BLUEPRINT = {
    "experiment_id": "EXP-MACRO-001",
    "valence_model": "Model_C",
    "default_clip_duration_seconds": 60,
    "duration_column_candidates": DEFAULT_DURATION_COLUMN_CANDIDATES,
    "phases": [
        {
            "phase_id": "Phase_1_Neutral",
            "target_valence": 0.05,
            "target_arousal": 4.0,
            "target_duration_seconds": 300,
        },
        {
            "phase_id": "Phase_2_Pleasant",
            "target_valence": 0.85,
            "target_arousal": 4.8,
            "target_duration_seconds": 420,
        },
        {
            "phase_id": "Phase_3_Stressful",
            "target_valence": -0.45,
            "target_arousal": 4.7,
            "target_duration_seconds": 360,
        },
    ],
}
