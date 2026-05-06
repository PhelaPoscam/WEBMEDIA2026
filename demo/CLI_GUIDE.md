# ASA CLI Demonstration Guide

This guide demonstrates how to use the **ASA (Affective Stimulus Assembly)** Command Line Interface (CLI) to generate experimental sessions.

## Overview

The CLI tool `src/prepare_session_cli.py` allows researchers to:
1. **Prepare Sessions**: Match participant profiles against experimental blueprints using a dataset.
2. **Export Blueprints**: Pull blueprint configurations from the database into JSON/YAML/XML files.

---

## 1. Preparing a Session

To generate a session playlist, you need three inputs:
1. **Participant Profile**: A JSON/YAML/XML file containing participant details and familiar media.
2. **Session Blueprint**: A JSON/YAML/XML file defining the target emotional states (valence/arousal) and durations for each phase.
3. **Dataset CSV**: The master list of stimuli with their affective ratings.

### Example Command

Run the following command from the project root:

```powershell
python -m src.prepare_session_cli prepare-session `
    --participant-file "demo/inputs/profiles/subject_001.json" `
    --blueprint-file "demo/inputs/blueprints/stress_detection.json" `
    --dataset-csv "dataset/All_Models_Normalized_Comparison.csv" `
    --matching-strategy "euclidean" `
    --output-json "demo/outputs/research_session/playlist_stress.json" `
    --output-csv "demo/outputs/research_session/playlist_stress.csv" `
    --output-report "demo/outputs/research_session/transparency_report_stress.md"
```

### What this does:
- Loads the participant's "familiar films" to ensure they aren't repeated.
- Reads the `stress_detection` blueprint which defines the neutral baseline, stress induction, and pleasant recovery emotional arc.
- Queries the `All_Models_Normalized_Comparison.csv` dataset.
- Calculates the optimal stimuli to reach the target valence/arousal for each phase using the **Euclidean** distance metric.
- Saves the resulting playlist and transparency report to `demo/outputs/research_session/`.

---

## 2. Automated Demo Script

We have provided a PowerShell script to run a full demonstration of the session preparation:

1. Open PowerShell in the project root.
2. Run:
   ```powershell
   powershell -ExecutionPolicy Bypass -File ./demo/run_demo.ps1
   ```

This will generate a sample playlist in `demo/outputs/research_session/`.
