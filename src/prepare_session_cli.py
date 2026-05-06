import argparse
import json
from pathlib import Path

from src._yaml import get_yaml
from src.exporter import (
    generate_blueprint_json,
    generate_blueprint_xml,
    generate_blueprint_yaml,
    generate_eprime_tab,
    generate_playlist_json,
    generate_playlist_xml,
    generate_playlist_yaml,
    generate_psychopy_csv,
)
from src.macro_pipeline import PlaylistGenerator, load_master_csv
from src.normalization import (
    coerce_modalities,
    normalize_blueprint_payload,
    normalize_participant_payload,
    write_phase_csv,
)
from src.parsers import get_blueprint_parser, get_parser

yaml = get_yaml()


def _load_db_dependencies():
    from src import models as loaded_models
    from src.db import SessionLocal as loaded_session_local

    return loaded_session_local, loaded_models


def _load_structured_file(path: Path, is_blueprint: bool = False) -> dict:
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8-sig")

    if is_blueprint:
        parser = get_blueprint_parser(path.name)
        parsed = parser.parse(raw_text)
        return parsed.model_dump()

    if suffix == ".xml":
        parser = get_parser(path.name)
        profile = parser.parse(raw_text)
        return profile.model_dump()

    if suffix == ".json":
        return json.loads(raw_text)

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("YAML support requires PyYAML to be installed")
        parsed = yaml.safe_load(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected mapping in {path}")
        return parsed

    raise ValueError(f"Unsupported file extension for structured input: {path.suffix}")


def prepare_session(
    participant_file: Path,
    blueprint_file: Path,
    dataset_csv: Path,
    output_json: Path | None,
    output_csv: Path | None,
    output_xml: Path | None,
    output_yaml: Path | None,
    model_id: str,
    default_clip_duration_seconds: int,
    matching_strategy: str = "euclidean",
    output_report: Path | None = None,
    output_psychopy: Path | None = None,
    output_eprime: Path | None = None,
) -> dict:
    participant_raw = _load_structured_file(participant_file, is_blueprint=False)
    blueprint_raw = _load_structured_file(blueprint_file, is_blueprint=True)

    participant_payload = normalize_participant_payload(participant_raw)
    blueprint_payload = normalize_blueprint_payload(
        blueprint_raw,
        model_id=model_id,
        default_clip_duration_seconds=default_clip_duration_seconds,
    )

    dataframe = load_master_csv(dataset_csv)
    generator = PlaylistGenerator(dataframe, matching_strategy=matching_strategy)
    playlist = generator.generate_playlist(
        participant_payload=participant_payload,
        experiment_blueprint=blueprint_payload,
    )

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(generate_playlist_json(playlist), encoding="utf-8")

    if output_csv is not None:
        write_phase_csv(playlist, output_csv)

    if output_xml is not None:
        output_xml.parent.mkdir(parents=True, exist_ok=True)
        output_xml.write_text(generate_playlist_xml(playlist), encoding="utf-8")

    if output_yaml is not None:
        output_yaml.parent.mkdir(parents=True, exist_ok=True)
        output_yaml.write_text(generate_playlist_yaml(playlist), encoding="utf-8")

    if output_report is not None:
        from src.exporter import generate_transparency_report

        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(generate_transparency_report(playlist), encoding="utf-8")

    if output_psychopy is not None:
        output_psychopy.parent.mkdir(parents=True, exist_ok=True)
        output_psychopy.write_text(generate_psychopy_csv(playlist), encoding="utf-8")

    if output_eprime is not None:
        output_eprime.parent.mkdir(parents=True, exist_ok=True)
        output_eprime.write_text(generate_eprime_tab(playlist), encoding="utf-8")

    return playlist


def export_blueprint_files(
    output_dir: Path,
    blueprint_id: str | None,
    blueprint_name: str | None,
) -> dict[str, str]:
    if not blueprint_id and not blueprint_name:
        raise ValueError("Provide --blueprint-id or --blueprint-name")

    session_local, db_models = _load_db_dependencies()

    with session_local() as db:
        if blueprint_id:
            bp = (
                db.query(db_models.ExperimentBlueprint)
                .filter(db_models.ExperimentBlueprint.id == blueprint_id)
                .first()
            )
        else:
            bp = (
                db.query(db_models.ExperimentBlueprint)
                .filter(db_models.ExperimentBlueprint.name == blueprint_name)
                .first()
            )

        if not bp:
            query_value = blueprint_id or blueprint_name
            raise ValueError(f"Blueprint not found: {query_value}")

        resolved_id = bp.id
        resolved_name = bp.name
        resolved_description = bp.description

        phases = (
            db.query(db_models.BlueprintPhase)
            .filter(db_models.BlueprintPhase.blueprint_id == resolved_id)
            .order_by(db_models.BlueprintPhase.phase_order)
            .all()
        )

    payload = {
        "id": resolved_id,
        "name": resolved_name,
        "description": resolved_description,
        "phases": [
            {
                "phase_order": phase.phase_order,
                "target_valence": phase.target_valence,
                "target_arousal": phase.target_arousal,
                "min_duration_ms": phase.min_duration_ms,
                "max_duration_ms": phase.max_duration_ms,
                "allowed_modalities": coerce_modalities(phase.allowed_modalities),
            }
            for phase in phases
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "json": output_dir / f"{resolved_id}.json",
        "xml": output_dir / f"{resolved_id}.xml",
        "yaml": output_dir / f"{resolved_id}.yaml",
    }
    outputs["json"].write_text(generate_blueprint_json(payload), encoding="utf-8")
    outputs["xml"].write_text(generate_blueprint_xml(payload), encoding="utf-8")
    outputs["yaml"].write_text(generate_blueprint_yaml(payload), encoding="utf-8")
    return {fmt: str(path) for fmt, path in outputs.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a full macro session from participant input, "
            "blueprint, and "
            "dataset CSV without running the web server."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-session",
        help="Generate a macro session output package",
    )
    prepare.add_argument(
        "--participant-file",
        required=True,
        type=Path,
        help="Path to participant file (.json/.yaml/.yml/.xml)",
    )
    prepare.add_argument(
        "--blueprint-file",
        required=True,
        type=Path,
        help="Path to blueprint file (.json/.yaml/.yml)",
    )
    prepare.add_argument(
        "--dataset-csv",
        required=True,
        type=Path,
        help=("Path to dataset CSV " "(e.g., dataset/All_Models_Normalized_Comparison.csv)"),
    )
    prepare.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for full session JSON",
    )
    prepare.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output path for flattened phase CSV",
    )
    prepare.add_argument(
        "--output-xml",
        type=Path,
        default=None,
        help="Optional output path for full session XML",
    )
    prepare.add_argument(
        "--output-yaml",
        type=Path,
        default=None,
        help="Optional output path for full session YAML",
    )
    prepare.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional output path for human-readable Markdown transparency report",
    )
    prepare.add_argument(
        "--output-psychopy",
        type=Path,
        default=None,
        help="Optional output path for PsychoPy-compatible loop CSV",
    )
    prepare.add_argument(
        "--output-eprime",
        type=Path,
        default=None,
        help="Optional output path for E-Prime-compatible tab-delimited file",
    )
    prepare.add_argument(
        "--model-id",
        default="Model_C",
        help="Fallback valence model when blueprint omits valence_model",
    )
    prepare.add_argument(
        "--default-clip-duration-seconds",
        type=int,
        default=60,
        help="Fallback clip duration in seconds",
    )
    prepare.add_argument(
        "--matching-strategy",
        default="euclidean",
        choices=["euclidean", "manhattan", "chebyshev", "quadratic", "mahalanobis"],
        help="Affective matching strategy for stimulus selection",
    )

    export_bp = subparsers.add_parser(
        "export-blueprint",
        help="Export a blueprint from the database to files",
    )
    export_bp.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save the exported files",
    )
    export_bp.add_argument(
        "--blueprint-id",
        default=None,
        help="Blueprint ID to export",
    )
    export_bp.add_argument(
        "--blueprint-name",
        default=None,
        help="Blueprint name to export (exact match)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-session":
        playlist = prepare_session(
            participant_file=args.participant_file,
            blueprint_file=args.blueprint_file,
            dataset_csv=args.dataset_csv,
            output_json=args.output_json,
            output_csv=args.output_csv,
            output_xml=args.output_xml,
            output_yaml=args.output_yaml,
            model_id=args.model_id,
            default_clip_duration_seconds=args.default_clip_duration_seconds,
            matching_strategy=args.matching_strategy,
            output_report=args.output_report,
            output_psychopy=args.output_psychopy,
            output_eprime=args.output_eprime,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "participant_id": playlist.get("participant_id"),
                    "experiment_id": playlist.get("experiment_id"),
                    "phase_count": len(playlist.get("phases", [])),
                    "output_json": str(args.output_json) if args.output_json else None,
                    "output_csv": (str(args.output_csv) if args.output_csv else None),
                    "output_xml": (str(args.output_xml) if args.output_xml else None),
                    "output_yaml": (str(args.output_yaml) if args.output_yaml else None),
                    "output_report": (str(args.output_report) if args.output_report else None),
                    "output_psychopy": (
                        str(args.output_psychopy) if args.output_psychopy else None
                    ),
                    "output_eprime": (str(args.output_eprime) if args.output_eprime else None),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "export-blueprint":
        outputs = export_blueprint_files(
            output_dir=args.output_dir,
            blueprint_id=args.blueprint_id,
            blueprint_name=args.blueprint_name,
        )
        print(json.dumps({"status": "ok", "outputs": outputs}, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
