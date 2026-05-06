# Backward-compat re-export shim.
# New imports should use src.playlist_generator directly.

__all__ = [
    "PlaylistGenerator",
    "PhaseSelectionResult",
    "InsufficientStimuliError",
    "load_master_csv",
    "MOCK_PARTICIPANT_PAYLOAD",
    "MOCK_EXPERIMENT_BLUEPRINT",
]

from src.playlist_generator import (  # noqa: E402,F401
    MOCK_EXPERIMENT_BLUEPRINT,
    MOCK_PARTICIPANT_PAYLOAD,
    InsufficientStimuliError,
    PhaseSelectionResult,
    PlaylistGenerator,
    load_master_csv,
)
