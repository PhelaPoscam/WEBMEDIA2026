# TODO

## Academic Readiness and Novelty

ASA is positioned as a reproducible, file-first framework for affective stimulus assembly. It compiles participant profiles, experimental blueprints, and affective datasets into auditable experiment-ready playlists. These items track work that strengthens the framework claim without requiring a standalone human-subjects evaluation before the broader experiments.

### Correctness and Transparency

- Keep generated phase payloads, presentation exports, XML/JSON/YAML exports, and transparency reports aligned around the same target affective coordinates.
- Preserve blueprint constraints end to end, including minimum duration, maximum/target duration, allowed modalities, matching strategy, and model selection.
- Report excluded stimuli and reasons, including familiarity, phobia/content sensitivity, invalid affective metadata, invalid duration, and phase-level modality constraints where appropriate.
- Add aggregate metrics to transparency reports: affective target error, duration error, selected-count summary, and per-phase pool reduction.

### Framework Generality

- Document the dataset contract so external datasets can be mapped into the framework without code changes.
- Support explicit column-mapping configuration for arbitrary dataset schemas, not only ASA's current CSV column names.
- Add examples using at least one non-ASA dataset schema to demonstrate that the framework is dataset-agnostic in practice.

### Methodological Comparisons

- Add non-human-subject benchmark scripts comparing Euclidean, Manhattan, Chebyshev, Quadratic, and Mahalanobis strategies across the same blueprint.
- Report strategy overlap, average affective distance, duration deviation, and remaining-pool statistics.
- Compare against simple baselines such as random selection, closest-single-clip selection, and manual/static playlist fixtures.

### Future Evaluation

- Evaluate the framework alongside the planned experiments rather than as a standalone study for now.
- Later collect researcher-facing evidence about reproducibility, transparency, setup time, and interpretability.
