# Brain Surgery Project

Sparse Autoencoder-based mechanistic interpretability project for small LLMs,
with end-to-end run-scoped training, verification, and executive reporting.

## Table of Contents

- [Research Index](#research-index)
- [Installation](#installation)
- [Quick Start (Local Verification)](#quick-start-local-verification)
- [Usage Guide](#usage-guide)
- [Reproducing the Results (7B)](#reproducing-the-results-7b)
- [What Is Finalized](#what-is-finalized)
- [Development Notes](#development-notes)
- [References](#references)
- [License](#-license)

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
```

- Activate the environment (for mac/linux)

```bash
source .venv/bin/activate
```

or (for windows)

```
.venv/Scripts/activate
```

```bash
pre-commit install
uv run pytest tests/
```

Model weights (offline local directory expected at `models/qwen2.5-7b`):

```bash
uv run hf download Qwen/Qwen2.5-7B --local-dir ./models/qwen2.5-7b
```

## Usage Guide

**Pre-requisite:**

- Ensure the local data directory exists.

- Before training, you must extract activation tensors from the LLM. This module runs the prompt corpus through the model hooks and saves a consolidated dataset.

`mkdir -p data/activations/`

## Reproducing the Results (7B)

Use the following workflow to reproduce the final 7B pipeline.

### 1. Model Acquisition

```bash
uv run hf download Qwen/Qwen2.5-7B --local-dir ./models/qwen2.5-7b
```

### 2. Activation Extraction

Generate the soccer activation dataset from Layer 14 residual-stream hooks.

```bash
uv run python -m brain_surgery.data_gen
```

### 3. SAE Training

Train for 100 epochs with L1 penalty 0.001.

```bash
uv run python scripts/train_university.py --epoch <num_epoches> --l1 <lambda_panelty> --wandb-run-name <wandb-ai-run-name-for-traceability>
```

### 4. Verification and Reporting

Verify pilot results (Q4-Q6) and generate an executive summary.

```bash
# Verify clusters, purity, and interventions
uv run scripts/verify_pilot.py \
  --run-id <run_id> \
  --checkpoint results/experiments/<run_id>/checkpoints/sae_best.pt \
  --dataset data/activations/soccer_activations_dataset.pt
```

### 5. Report Generation

Synthesize interpretability findings from the final run ID.

```bash
uv run python scripts/generate_report.py --run-id <run_id>
```

Checkpoint resolution follows the experiment hierarchy: `results/experiments/<run_id>/checkpoints/sae_best.pt`.

### 6. Project Structure Note

- Run-scoped outputs are stored under `results/experiments/`.
- For the 7B workflow, the SAE uses a 32x expansion factor:
  $3584 \\times 32 = 114688$ latent features.

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

## References

- Nikola Kriznar:
  https://github.com/nkriznar/sparse-autoencoder-llm-interpretability
- Miguel Angel Palafox Gomez (ter-kes):
  https://github.com/ter-kes/sparse-autoencoder-llm-interpretability
- Anthropic, Scaling Monosemanticity:
  https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

## 📝 License

See [LICENSE](LICENSE) for details.

______________________________________________________________________

**Course:** MCDA 5511 – Current Practices I (Winter 2026)

**Instructors:**

1. Mr. Greg Kirczenow
1. Mr. Somto Muetoe
1. Dr. Drira
1. Prof. Neveditsin

**Team Members:**

1. Bhavik Kantilal Bhagat
1. Miguel Angel Palafox Gomez
1. Nikola Kriznar
1. Sridhar Vadla
