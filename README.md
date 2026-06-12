# WebMedia 2026

**ASA** (Affective Stimulus Assembly) framework

This version has been prepared and published for **WebMedia 2026**.

## Key Features

- **File-first framework**: Designed for research automation, batch processing, and integration with different affective datasets.
- **Affective matching engine**: Distance metrics for stimulus selection in valence-arousal space.
- **Flexible exporters**: CSV, JSON, XML, and YAML output for downstream research tools.
- **Subject-tailored logic**: Filters familiar films from participant-specific playlists.
- **Dataset agnostic**: Processes compatible experimental datasets when they provide stimulus identifiers, affective metadata, and duration information.

## Framework Model

ASA is organized around a reusable assembly contract rather than a single fixed dataset. A run combines:

- a participant profile with familiarity and safety constraints;
- an experimental blueprint with phase-level affective targets, durations, and modality constraints;
- an affective stimulus dataset with identifiers, valence/arousal metadata, and durations.

The bundled CSV is the reference dataset, but the framework is intended to support other affective datasets through column mapping and schema normalization. Current CSV inputs should include at minimum `Film`, `Arousal mean`, one supported valence column such as `Model_C (Dominant)`, and a duration column such as `Duration_ms` or `Duration_seconds`.

## Installation

Create and activate a project-local virtual environment before installing dependencies. This keeps ASA from accidentally running inside another application's bundled Python, such as PsychoPy's Python.

### Windows PowerShell

```powershell
git clone https://github.com/PhelaPoscam/WEBMEDIA2026.git
cd WEBMEDIA2026

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -c "import sys; print(sys.executable)"
```

The final command should print a path inside `.venv`.

### macOS/Linux

```bash
git clone https://github.com/PhelaPoscam/WEBMEDIA2026.git
cd WEBMEDIA2026

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -c "import sys; print(sys.executable)"
```

The final command should print a path inside `.venv`.

## Usage

### Generating a Session Playlist

Use the built-in CLI to generate a participant-specific playlist using the stress-detection blueprint:

```powershell
python -m src.prepare_session_cli prepare-session `
    --participant-file "demo/inputs/profiles/subject_001.json" `
    --blueprint-file "demo/inputs/blueprints/stress_detection.json" `
    --dataset-csv "dataset/All_Models_Normalized_Comparison.csv" `
    --matching-strategy "euclidean" `
    --output-json "demo/outputs/research_session/playlist_stress.json" `
    --output-csv "demo/outputs/research_session/playlist_stress.csv" `
    --output-report "demo/outputs/research_session/transparency_report_stress.md" `
    --output-psychopy "demo/outputs/research_session/playlist_stress_psychopy.csv" `
    --output-eprime "demo/outputs/research_session/playlist_stress_eprime.txt"
```

### Configurable Strategies

For the serverless CLI, pass the matching strategy explicitly:

```powershell
python -m src.prepare_session_cli prepare-session `
    --participant-file "demo/inputs/profiles/subject_001.json" `
    --blueprint-file "demo/inputs/blueprints/stress_detection.json" `
    --dataset-csv "dataset/All_Models_Normalized_Comparison.csv" `
    --matching-strategy "quadratic" `
    --output-json "demo/outputs/research_session/playlist_stress_quadratic.json"
```

Available strategies: `euclidean`, `manhattan`, `chebyshev`, `quadratic`, `mahalanobis`.

The `.env` setting `AFFECTIVE_MATCHING_STRATEGY` is used by the database-backed matching engine, not by the CSV CLI when `--matching-strategy` is provided.

### Optional Database Blueprint Export

`prepare-session` is fully CSV/file based. The `export-blueprint` command is optional and uses the local SQLAlchemy models/database:

```powershell
python -m src.prepare_session_cli export-blueprint `
    --blueprint-id "example-blueprint-id" `
    --output-dir "exports/blueprints"
```

## Testing

Activate the virtual environment first, then run:

```bash
python -m pytest tests
```