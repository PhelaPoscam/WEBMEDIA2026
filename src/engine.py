from sqlalchemy.orm import Session

from src.affective_matching import (
    AffectiveMatchingEngine,
    MatchingStrategy,
    compute_effective_affect,
)
from src.config import get_settings
from src.data_source import DbStimulusSource
from src.models import ParticipantProfile
from src.playlist_generator import PlaylistGenerator
from src.schemas import experiment_constraint_to_blueprint_dict


class InsufficientStimuliError(Exception):
    def __init__(self, phase_id: str, missing_duration_ms: int):
        self.phase_id = phase_id
        self.missing_duration_ms = missing_duration_ms
        super().__init__(
            f"Insufficient stimuli for phase {phase_id}. Missing {missing_duration_ms}ms."
        )


def _db_stimulus_to_dict(item: dict) -> dict:
    return {
        "stimulus_id": item.get("film", ""),
        "stimulus_code": item.get("film", "N/A"),
        "uri": f"csv://{item.get('film', '')}",
        "distance": item.get("distance", 0.0),
        "eff_valence": item.get("valence", 0.0),
        "eff_arousal": item.get("arousal", 0.0),
        "raw_valence": item.get("valence", 0.0),
        "raw_arousal": item.get("arousal", 0.0),
        "duration_ms": item.get("duration_seconds", 0) * 1000,
    }


def _try_parse_int(s: str) -> str | int:
    try:
        return int(s)
    except ValueError:
        return s


def _normalize_transparency_phases(phases: list[dict]) -> list[dict]:
    result = []
    for p in phases:
        p = dict(p)
        p["phase_id"] = _try_parse_int(str(p.get("phase_id", "")))
        result.append(p)
    return result


class MatchEngine:
    AFFECT_MIN = 1.0
    AFFECT_MAX = 9.0
    LIKERT_MIN = 1.0
    LIKERT_MAX = 5.0
    VALENCE_SHIFT_STRENGTH = 0.6
    AROUSAL_SHIFT_STRENGTH = 0.3

    def __init__(self, db: Session, participant_id: str):
        self.db = db
        self.participant = (
            db.query(ParticipantProfile).filter(ParticipantProfile.id == participant_id).first()
        )
        if not self.participant:
            raise ValueError(f"Participant {participant_id} not found.")

        settings = get_settings()
        strategy = MatchingStrategy(settings.affective_matching_strategy)
        self.distance_metric = AffectiveMatchingEngine.get_metric(strategy)

        self._settings = settings

    def _get_effective_affect(
        self,
        normative_val: float,
        normative_aro: float,
        preference_score: float | None,
        match_arousal: bool = True,
    ) -> tuple[float, float]:
        return compute_effective_affect(
            normative_val=normative_val,
            normative_aro=normative_aro,
            preference_score=preference_score,
            match_arousal=match_arousal,
            valence_shift_strength=self.VALENCE_SHIFT_STRENGTH,
            arousal_shift_strength=self.AROUSAL_SHIFT_STRENGTH,
            likert_min=self.LIKERT_MIN,
            likert_max=self.LIKERT_MAX,
        )

    def generate_playlist(
        self,
        constraint,
        randomize: bool = False,
        model_id: str = "Model_C",
        match_arousal: bool = True,
    ) -> dict:
        from datetime import datetime

        bp_dict = experiment_constraint_to_blueprint_dict(constraint, model_id)

        participant_payload = {
            "participant_id": self.participant.id,
            "familiar_films": self.participant.familiar_media,
            "phobias": self.participant.phobias,
        }

        generator = PlaylistGenerator(
            stimulus_source=DbStimulusSource(self.db),
            matching_strategy=self._settings.affective_matching_strategy,
            match_arousal=match_arousal,
        )

        raw_result = generator.generate_playlist(
            participant_payload=participant_payload,
            experiment_blueprint=bp_dict,
        )

        # Check for insufficient stimuli
        for i, csv_phase in enumerate(raw_result.get("phases", [])):
            achieved = csv_phase.get("achieved_duration_seconds", 0)
            if achieved == 0:
                raise InsufficientStimuliError(
                    phase_id=csv_phase.get("phase_id", "unknown"),
                    missing_duration_ms=csv_phase.get("target_duration_seconds", 0) * 1000,
                )
            constraint_phase = constraint.phases[i]
            min_required_s = constraint_phase.min_duration_ms // 1000
            if achieved < min_required_s:
                raise InsufficientStimuliError(
                    phase_id=csv_phase.get("phase_id", "unknown"),
                    missing_duration_ms=(min_required_s - achieved) * 1000,
                )

        result: dict = {
            "participant_id": raw_result["participant_id"],
            "experiment_id": raw_result["experiment_id"],
            "valence_model": model_id,
            "valence_column": "normative_valence",
            "unassigned_pool_size": raw_result.get("unassigned_pool_size", 0),
            "phases": [],
            "transparency_report": {},
        }

        for csv_phase in raw_result.get("phases", []):
            selected_stimuli = [
                _db_stimulus_to_dict(f) for f in csv_phase.get("selected_films", [])
            ]
            total_duration_ms = sum(item["duration_ms"] for item in selected_stimuli)

            result["phases"].append(
                {
                    "phase_id": csv_phase["phase_id"],
                    "target_valence": csv_phase.get("target_valence", 0.0),
                    "target_arousal": csv_phase.get("target_arousal", 0.0),
                    "selected_stimuli": selected_stimuli,
                    "total_duration_ms": total_duration_ms,
                }
            )

        raw_report = raw_result.get("transparency_report", {})
        result["transparency_report"] = {
            "run_metadata": {
                "timestamp": raw_report.get("run_metadata", {}).get(
                    "timestamp", datetime.now().isoformat()
                ),
                "matching_strategy": raw_report.get("run_metadata", {}).get(
                    "matching_strategy", ""
                ),
                "valence_model": model_id,
                "valence_column": "normative_valence",
                "input_pool_size": raw_report.get("run_metadata", {}).get("input_pool_size", 0),
            },
            "exclusions": raw_report.get("exclusions", []),
            "phases": _normalize_transparency_phases(raw_report.get("phases", [])),
        }

        return result
