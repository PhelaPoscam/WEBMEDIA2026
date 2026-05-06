import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.macro_pipeline import (  # noqa: E402
    MOCK_EXPERIMENT_BLUEPRINT,
    MOCK_PARTICIPANT_PAYLOAD,
    PlaylistGenerator,
    load_master_csv,
)

if __name__ == "__main__":
    csv_path = ROOT / "dataset" / "All_Models_Normalized_Comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Master stimuli CSV not found: {csv_path}")

    master_df = load_master_csv(csv_path)
    generator = PlaylistGenerator(master_df)
    playlist = generator.generate_playlist(
        participant_payload=MOCK_PARTICIPANT_PAYLOAD,
        experiment_blueprint=MOCK_EXPERIMENT_BLUEPRINT,
    )

    print(json.dumps(playlist, indent=2))
