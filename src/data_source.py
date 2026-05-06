"""Stimulus data-source abstraction for CSV and database backends.

Both the CSV-based macro pipeline and the DB-backed MatchEngine
provide stimulus data through this protocol so the unified
PlaylistGenerator doesn't care where its stimuli come from.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol

import pandas as pd

from src.affective_matching import UnifiedStimulus

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

MODEL_TO_COLUMN = {
    "Model_A": "Model_A (PANAS)",
    "Model_B": "Model_B (Averages)",
    "Model_C": "Model_C (Dominant)",
    "Model_D": "Model_D (PCA)",
}

DEFAULT_DURATION_COLUMN_CANDIDATES = [
    "Duration_ms",
    "duration_ms",
    "Duration_seconds",
    "duration_seconds",
    "Duration (s)",
    "duration_s",
]


def parse_csv_tags(val: Any) -> list[str]:
    if not val or pd.isna(val):
        return []
    val_str = str(val).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            import json

            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed]
        except Exception:
            pass
    return [x.strip().lower() for x in val_str.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class StimulusSource(Protocol):
    """Protocol for stimulus data backends (CSV DataFrame, DB session, etc.)."""

    def iter_all(self) -> Iterator[UnifiedStimulus]: ...

    @property
    def source_name(self) -> str: ...


# ---------------------------------------------------------------------------
# CSV backend
# ---------------------------------------------------------------------------


class CsvStimulusSource:
    """Reads stimuli from a pandas DataFrame backed by the master CSV catalog."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        valence_model_key: str = "Model_C",
        duration_column_candidates: list[str] | None = None,
        default_clip_duration_seconds: int = 60,
    ):
        self._df = dataframe.copy()
        self._valence_model_key = valence_model_key
        self._duration_candidates = duration_column_candidates or DEFAULT_DURATION_COLUMN_CANDIDATES
        self._default_duration_s = default_clip_duration_seconds

        required_base_cols = {"Film", "Arousal mean"}
        missing = required_base_cols - set(self._df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        self._df["Film"] = self._df["Film"].astype(str).str.strip()
        self._df["Arousal mean"] = pd.to_numeric(self._df["Arousal mean"], errors="coerce")

        for _, col in MODEL_TO_COLUMN.items():
            if col in self._df.columns:
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")

        self._tag_column: str | None = self._discover_column(
            {"content_tags", "content_tag", "tags", "tag"}
        )
        self._modality_column: str | None = self._discover_column({"modality", "type"})

    source_name = "csv"

    # --- resolvers ----------------------------------------------------------

    def _resolve_valence_column(self, model_key: str) -> str:
        if model_key in MODEL_TO_COLUMN:
            known_col = MODEL_TO_COLUMN[model_key]
            if known_col in self._df.columns:
                return known_col

        candidates = [c for c in self._df.columns if c.startswith(model_key)]
        if candidates:
            return candidates[0]

        valid = ", ".join(MODEL_TO_COLUMN)
        raise ValueError(
            f"Unsupported valence model '{model_key}'. "
            f"Expected one of: {valid}, or ensure the CSV has a column "
            f"starting with '{model_key}'."
        )

    def _resolve_duration_column(self) -> tuple[str, str]:
        for col in self._duration_candidates:
            if col in self._df.columns:
                unit = "milliseconds" if col.lower().endswith("_ms") else "seconds"
                return col, unit
        raise ValueError(
            "Could not find a duration column in the dataset. "
            "Ensure the CSV contains 'Duration_ms' or 'Duration_seconds'.",
        )

    def _discover_column(self, names: set[str]) -> str | None:
        for col in self._df.columns:
            if col.lower() in names:
                return col
        return None

    # --- iteration ----------------------------------------------------------

    def iter_all(self) -> Iterator[UnifiedStimulus]:
        df = self._df
        val_col = self._resolve_valence_column(self._valence_model_key)
        dur_col, dur_unit = self._resolve_duration_column()
        dur_is_ms = dur_unit == "milliseconds"
        tag_col = self._tag_column
        mod_col = self._modality_column

        for _, row in df.iterrows():
            try:
                film_id = str(row["Film"])
                arousal_val = float(row["Arousal mean"])
                valence_val = float(row[val_col])

                duration_raw = pd.to_numeric(row[dur_col], errors="coerce")
                if pd.isna(duration_raw) or duration_raw <= 0:
                    duration_s = self._default_duration_s
                elif dur_is_ms:
                    duration_s = int(round(float(duration_raw) / 1000.0))
                else:
                    duration_s = int(round(float(duration_raw)))

                if duration_s <= 0:
                    duration_s = self._default_duration_s

                tags = parse_csv_tags(row[tag_col]) if tag_col else []
                modality = str(row[mod_col]).strip().lower() if mod_col else "video"

                yield _CsvStimulus(
                    stimulus_id=film_id,
                    film_name=film_id,
                    valence=valence_val,
                    arousal=arousal_val,
                    duration_ms=int(duration_s) * 1000,
                    content_tags=tags,
                    modality=modality,
                )
            except (ValueError, TypeError, KeyError):
                continue


class _CsvStimulus:
    """Lightweight UnifiedStimulus-compatible wrapper for a CSV row."""

    __slots__ = (
        "id",
        "code",
        "uri",
        "modality",
        "content_tags",
        "normative_valence",
        "normative_arousal",
        "duration_ms",
        "film",
    )

    def __init__(
        self,
        stimulus_id: str,
        film_name: str,
        valence: float,
        arousal: float,
        duration_ms: int,
        content_tags: list[str],
        modality: str = "video",
    ):
        self.id = stimulus_id
        self.code = film_name
        self.uri = f"csv://{film_name}"
        self.modality = modality
        self.content_tags = content_tags
        self.normative_valence = valence
        self.normative_arousal = arousal
        self.duration_ms = duration_ms
        self.film = film_name


# ---------------------------------------------------------------------------
# Database backend
# ---------------------------------------------------------------------------


def load_master_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")


class DbStimulusSource:
    """Reads stimuli from the StimulusMetadata SQLAlchemy table.

    If the table is empty, falls back to the master CSV catalog
    (matching the original MatchEngine._load_csv_stimuli behavior).
    """

    source_name = "db"

    def __init__(
        self,
        session: "Session",
        valence_model_key: str = "Model_C",
    ):
        self._session = session
        self._valence_model_key = valence_model_key

    def iter_all(self) -> Iterator[UnifiedStimulus]:
        from src.models import StimulusMetadata

        db_stimuli = self._session.query(StimulusMetadata).all()
        if db_stimuli:
            yield from db_stimuli
            return

        yield from self._load_csv_fallback()

    def _load_csv_fallback(self) -> Iterator[UnifiedStimulus]:
        csv_path = Path("dataset/All_Models_Normalized_Comparison.csv")
        if not csv_path.exists():
            return

        try:
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        except Exception:
            return

        valence_col = MODEL_TO_COLUMN.get(self._valence_model_key, MODEL_TO_COLUMN["Model_C"])
        if (
            "Film" not in df.columns
            or "Arousal mean" not in df.columns
            or valence_col not in df.columns
        ):
            return

        for _, row in df.iterrows():
            try:
                film_id = str(row["Film"])
                valence = float(row[valence_col])
                arousal = float(row["Arousal mean"])

                duration_ms = 60000
                for cand in (
                    "Duration_ms",
                    "duration_ms",
                    "Duration_seconds",
                    "duration_seconds",
                ):
                    if cand in row:
                        val = float(row[cand])
                        if cand.lower().endswith("_ms"):
                            duration_ms = int(val)
                        else:
                            duration_ms = int(val * 1000)
                        break

                yield _DbFallbackStimulus(
                    stimulus_id=film_id,
                    valence=valence,
                    arousal=arousal,
                    duration_ms=duration_ms,
                )
            except (ValueError, TypeError, KeyError):
                continue


class _DbFallbackStimulus:
    """Lightweight UnifiedStimulus-compatible for DB CSV fallback."""

    __slots__ = (
        "id",
        "code",
        "uri",
        "modality",
        "content_tags",
        "normative_valence",
        "normative_arousal",
        "duration_ms",
        "film",
    )

    def __init__(self, stimulus_id: str, valence: float, arousal: float, duration_ms: int):
        self.id = stimulus_id
        self.code = stimulus_id
        self.uri = "csv_catalog"
        self.modality = "video"
        self.content_tags: list[str] = []
        self.normative_valence = valence
        self.normative_arousal = arousal
        self.duration_ms = duration_ms
        self.film = stimulus_id
