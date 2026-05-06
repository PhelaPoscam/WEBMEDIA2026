from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParticipantProfileBase(BaseModel):
    id: str = Field(min_length=1)
    baseline_valence: float = Field(ge=-1.0, le=1.0)
    baseline_arousal: float = Field(ge=-1.0, le=1.0)
    phobias: List[str] = Field(default_factory=list)
    familiar_media: List[str] = Field(default_factory=list)
    preferences: dict = Field(default_factory=dict)
    is_excluded: bool = False
    exclusion_reason: str | None = Field(default=None, max_length=500)


class ParticipantProfileCreate(ParticipantProfileBase):
    pass


class ParticipantAdminUpdate(BaseModel):
    is_excluded: bool = False
    exclusion_reason: str | None = Field(default=None, max_length=500)


class ParticipantProfile(ParticipantProfileBase):
    model_config = ConfigDict(from_attributes=True)


class StimulusMetadataBase(BaseModel):
    id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    content_tags: List[str] = Field(default_factory=list)
    normative_valence: float = Field(ge=-1.0, le=1.0)
    normative_arousal: float = Field(ge=-1.0, le=1.0)
    duration_ms: int = Field(gt=0)


class StimulusMetadataCreate(StimulusMetadataBase):
    pass


class StimulusMetadata(StimulusMetadataBase):
    model_config = ConfigDict(from_attributes=True)


class ExperimentPhase(BaseModel):
    """Phase definition for the DB-backed MatchEngine.

    Use ``ExperimentConstraint`` for database queries and the
    ``experiment_constraint_to_blueprint_dict`` adapter to convert
    to the canonical ``MacroExperimentBlueprint`` dict format expected
    by the unified ``PlaylistGenerator``.
    """

    phase_id: str | int
    target_valence: float = Field(ge=-1.0, le=1.0)
    target_arousal: float = Field(ge=-1.0, le=1.0)
    duration_range_ms: List[int] = Field(
        min_length=2,
        max_length=2,
    )  # [min, max]
    allowed_modalities: List[str] = Field(min_length=1)
    requires_novelty: bool = True

    @property
    def min_duration_ms(self) -> int:
        return self.duration_range_ms[0]

    @property
    def max_duration_ms(self) -> int:
        return self.duration_range_ms[1]

    @model_validator(mode="after")
    def validate_duration_range(self):
        if self.duration_range_ms[0] > self.duration_range_ms[1]:
            raise ValueError("duration_range_ms must be [min, max] " "with min <= max")
        return self


class ExperimentConstraint(BaseModel):
    experiment_id: str
    phases: List[ExperimentPhase] = Field(min_length=1)


class PlaylistResponse(BaseModel):
    experiment_id: str
    participant_id: str
    phases: List[dict]  # Will contain phase_id and list of stimuli URIs


class BlueprintPhaseBase(BaseModel):
    phase_order: int = Field(ge=0)
    target_valence: float = Field(ge=-1.0, le=1.0)
    target_arousal: float = Field(ge=-1.0, le=1.0)
    min_duration_ms: int = Field(gt=0)
    max_duration_ms: int = Field(gt=0)
    allowed_modalities: List[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_duration_bounds(self):
        if self.min_duration_ms > self.max_duration_ms:
            raise ValueError("min_duration_ms must be less than or equal to max_duration_ms")
        return self


class BlueprintPhaseCreate(BlueprintPhaseBase):
    pass


class BlueprintPhase(BlueprintPhaseBase):
    id: int
    blueprint_id: str
    model_config = ConfigDict(from_attributes=True)


class ExperimentBlueprintBase(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str | None = None


class ExperimentBlueprintCreate(ExperimentBlueprintBase):
    phases: List[BlueprintPhaseBase] = Field(min_length=1)


class ExperimentBlueprint(ExperimentBlueprintBase):
    phases: List[BlueprintPhase]
    model_config = ConfigDict(from_attributes=True)


class BlueprintImportResult(BaseModel):
    status: str
    overwritten: bool
    blueprint: ExperimentBlueprint


class MacroParticipantPayload(BaseModel):
    participant_id: str = Field(min_length=1)
    familiar_films: List[str] = Field(default_factory=list)
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "participant_id": "P-0001",
                "familiar_films": [
                    "Ghost",
                    "Forrest Gump",
                    "The Exorcist",
                ],
            }
        },
    )


class MacroPhase(BaseModel):
    phase_id: str = Field(min_length=1)
    target_valence: float
    target_arousal: float
    target_duration_seconds: int = Field(gt=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "phase_id": "Phase_1_Neutral",
                "target_valence": 0.05,
                "target_arousal": 4.0,
                "target_duration_seconds": 300,
            }
        }
    )


class MacroExperimentBlueprint(BaseModel):
    """Canonical blueprint schema for the unified PlaylistGenerator.

    Both the CSV-based CLI (``prepare-session``) and the DB-backed
    ``MatchEngine`` (via the adapter) produce payloads conforming to
    this schema.
    """

    experiment_id: str = Field(min_length=1)
    valence_model: str = Field(min_length=1)
    phases: List[MacroPhase] = Field(min_length=1)
    default_clip_duration_seconds: int = Field(default=60, gt=0)
    duration_column_candidates: List[str] = Field(default_factory=list)
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "experiment_id": "EXP-MACRO-001",
                "valence_model": "Model_C",
                "default_clip_duration_seconds": 60,
                "duration_column_candidates": [
                    "Duration_seconds",
                    "duration_ms",
                ],
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
                ],
            }
        },
    )


class MacroPipelineRequest(BaseModel):
    participant_payload: MacroParticipantPayload
    experiment_blueprint: MacroExperimentBlueprint
    matching_strategy: str = Field(
        default="euclidean",
        description=(
            "Affective matching strategy: euclidean, manhattan, "
            "chebyshev, quadratic, or mahalanobis"
        ),
    )
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "participant_payload": {
                    "participant_id": "P-0001",
                    "familiar_films": [
                        "Ghost",
                        "Forrest Gump",
                        "The Exorcist",
                    ],
                },
                "experiment_blueprint": {
                    "experiment_id": "EXP-MACRO-001",
                    "valence_model": "Model_C",
                    "default_clip_duration_seconds": 60,
                    "duration_column_candidates": [
                        "Duration_seconds",
                        "duration_ms",
                    ],
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
                },
                "matching_strategy": "euclidean",
            }
        },
    )


class MacroPipelineResponse(BaseModel):
    participant_id: str
    experiment_id: str
    valence_model: str
    valence_column: str
    input_pool_size: int
    post_familiarity_pool_size: int
    phases: List[dict[str, Any]]
    unassigned_pool_size: int


def experiment_constraint_to_blueprint_dict(
    constraint: "ExperimentConstraint",
    valence_model: str = "Model_C",
) -> dict[str, Any]:
    """Convert a DB-layer ``ExperimentConstraint`` into the canonical dict
    expected by ``PlaylistGenerator.generate_playlist``."""
    return {
        "experiment_id": constraint.experiment_id,
        "valence_model": valence_model,
        "phases": [
            {
                "phase_id": str(phase.phase_id),
                "target_valence": phase.target_valence,
                "target_arousal": phase.target_arousal,
                "target_duration_seconds": phase.duration_range_ms[1] // 1000,
                "min_duration_seconds": phase.min_duration_ms // 1000,
                "allowed_modalities": phase.allowed_modalities,
            }
            for phase in constraint.phases
        ],
    }
