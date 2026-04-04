# Brain Surgery Project

Sparse Autoencoder-based mechanistic interpretability project for small LLMs,
with end-to-end run-scoped training, verification, and executive reporting.

## Research Index

- [Theory Framework](docs/THEORY.md)
- [Q1: Model Wrapper and Hooks](docs/Q1_model_and_hooks.md)
- [Q2: Data Collection and Storage](docs/Q2_data_collection_and_storage.md)
- [Q3: SAE Training](docs/Q3_sae_training.md)
- [Q4: Feature Interpretation](docs/Q4_feature_interpretation.md)
- [Q5: Feature Labeling and Cluster Validation](docs/Q5_feature_labeling_bonus.md)
- [Q6: Intervention and Counterfactuals](docs/Q6_intervention_and_counterfactuals.md)

## Installation

Requirements:

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

Setup:

```bash
uv sync
pre-commit install
uv run pytest tests/
```

Model weights (offline local directory expected at `models/qwen2.5-0.5b`):

```bash
uv run hf download Qwen/Qwen2.5-0.5B --local-dir ./models/qwen2.5-0.5b
```

## Usage Guide

The standard workflow is a 3-step run pipeline.

1. Train SAE (smoke test or full run)

```bash
uv run python scripts/train_university.py --smoke-test --run-id run_YYYYMMDD_HHMM
```

For full training, remove `--smoke-test` and tune training flags as needed.

2. Verify pilot (Q4-Q6) with run-scoped paths

```bash
uv run python scripts/verify_pilot.py --run-id run_YYYYMMDD_HHMM
```

Checkpoint resolution order is:

1. `--checkpoint` (if explicitly provided)

1. `results/experiments/<run_id>/checkpoints/sae_best.pt`

1. global fallback checkpoint

1. Generate executive summary report

```bash
uv run python scripts/generate_report.py --run-id run_YYYYMMDD_HHMM
```

This writes `executive_summary.json` in
`results/experiments/<run_id>/`.

## What Is Finalized

- NDJSON-driven corpus and metadata pipeline for activation capture.
- Run-scoped directories for metrics, features, experiments, and checkpoints.
- Spherical K-Means workflow with dynamic elbow selection for semantic grouping.
- Metadata-driven cluster purity reporting (`cluster_report.json`).
- Q6 intervention analysis with target-vs-control specificity logic.

## Development Notes

- Lint/format/type checks run through pre-commit.
- Main scripts are designed for deterministic, run-id-based reproducibility.
- Keep large artifacts in `data/` and `results/`; avoid committing generated
  binaries unless required for release snapshots.

## 📝 License

See [LICENSE](LICENSE) for details.

______________________________________________________________________

**Course:** MCDA 5511 – Deep Learning (Winter 2026)
**Instructor:** Mr. Greg Kirczenow, Mr. Somto Muetoe, Dr. Drira, Prof. Neveditsin
**Team Members:**

1. Bhavik Kantilal Bhagat
1. Miguel Angel Palafox Gomez
1. Nikola Kriznar
1. Sridhar Vadla
