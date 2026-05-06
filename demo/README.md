# ASA Demo Materials

This directory contains resources for testing and demonstrating the **Affective Stimulus Assembly (ASA)**.

## Contents

- **[CLI_GUIDE.md](./CLI_GUIDE.md)**: A detailed guide on how to use the Command Line Interface.
- **[run_demo.ps1](./run_demo.ps1)**: A PowerShell script that runs a sample research-grade session preparation.
- **inputs/profiles/**: Sample research participant profile.
- **inputs/blueprints/**: Sample experimental induction blueprint.
- **outputs/**: Directory where generated playlists are stored.

## Quick Start (CLI)

To see the system in action, run the following in your terminal:

```powershell
powershell -ExecutionPolicy Bypass -File ./demo/run_demo.ps1
```

This will match a research participant against a standard induction blueprint and generate a full session playlist in `demo/outputs/research_session/`.
