from sqlalchemy import JSON, Boolean, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from src.db import Base


class ParticipantProfile(Base):
    __tablename__ = "participant_profiles"

    id = Column(String, primary_key=True, index=True)
    baseline_valence = Column(Float, default=0.0)
    baseline_arousal = Column(Float, default=0.0)
    phobias = Column(JSON, default=list)  # List of strings
    familiar_media = Column(JSON, default=list)  # List of stimulus IDs
    preferences = Column(JSON, default=dict)  # Dict of stimulus_id: preference_score
    is_excluded = Column(Boolean, default=False)
    exclusion_reason = Column(String, default=None)


class StimulusMetadata(Base):
    __tablename__ = "stimuli_metadata"

    id = Column(String, primary_key=True, index=True)
    uri = Column(String, index=True)
    modality = Column(String, index=True)
    content_tags = Column(JSON, default=list)
    normative_valence = Column(Float, default=0.0)
    normative_arousal = Column(Float, default=0.0)
    duration_ms = Column(Integer)


class ExperimentBlueprint(Base):
    __tablename__ = "experiment_blueprints"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)

    phases = relationship(
        "BlueprintPhase", back_populates="blueprint", cascade="all, delete-orphan"
    )


class BlueprintPhase(Base):
    __tablename__ = "blueprint_phases"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    blueprint_id = Column(String, ForeignKey("experiment_blueprints.id"), index=True)
    phase_order = Column(Integer, default=0)

    target_valence = Column(Float)
    target_arousal = Column(Float)
    min_duration_ms = Column(Integer)
    max_duration_ms = Column(Integer)
    allowed_modalities = Column(JSON, default=list)  # List of strings

    blueprint = relationship("ExperimentBlueprint", back_populates="phases")
