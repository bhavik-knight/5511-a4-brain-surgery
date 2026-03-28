# Q2 — Data Collection & Activation Storage

Implementation is in `src/brain_surgery/data_gen.py`.

## Storage Locations (Repo Convention)

- **Corpus data:** `data/corpus/`
- **Captured activations (artifacts):** `data/activations/`
  - Saved as `.pt` files via `ModelWrapper.save_activations()`.
  - Each artifact is designed to be *token-aligned* (tokens ↔ activation rows).

## Artifact Naming Convention

Activation artifacts are stored as `.pt` files under `data/activations/`.

- **Default naming:** if you don’t pass `file_stem=...`, the wrapper uses a descriptive stem like:
  - `layers_{layer_idx}_acts_batch_{batch_idx}.pt`
- **Recommended naming (when running a corpus pipeline):** pass `file_stem=...` so filenames encode the run context.
  - Suggested stem format:
    - `{corpus}_{split}_layer{layer_idx}_shard{shard}_batch{batch_idx}`
  - Example:
    - `tinystories_train_layer4_shard00_batch0123.pt`

For non-activation outputs, follow the repo’s output folders:

- `results/metrics/` for JSON/CSV metrics and plots
- `results/experiments/` for intervention/counterfactual outputs
- `results/features/` for feature reports/snippets

## Results File Naming (Recommended)

To keep handoff and grading simple, save machine-readable outputs with predictable filenames.

- **Corpus / data-collection stats** (Q2) → `results/metrics/`

  - Suggested filename:
    - `corpus_stats_{corpus}_{split}.json`
  - Example:
    - `corpus_stats_tinystories_train.json`

- **SAE training metrics** (Q3) → `results/metrics/`

  - Suggested filename:
    - `sae_metrics_{corpus}_{split}_layer{layer_idx}.json`
  - Example:
    - `sae_metrics_tinystories_train_layer4.json`

- **Intervention / counterfactual summaries** (Q6) → `results/experiments/`

  - Suggested filename:
    - `clamp_summary_{corpus}_{split}_layer{layer_idx}.json`
  - Example:
    - `clamp_summary_tinystories_valid_layer4.json`

If you run multiple experimental settings, add a short suffix like `_v1`, `_topk50`, or a date stamp (e.g., `_2026-03-28`).

## Expected Artifact Contents

Activation artifacts written by `ModelWrapper.save_activations()` include:

- `prompt`, `generated_text`
- `token_ids`, `token_strs`, `token_texts`
- `activations` (shape: `(seq_len, hidden_dim)`)

## Results/Outputs

Any corpus statistics (counts, shapes, sizes, timing) should be written under:

- `results/metrics/`

______________________________________________________________________

This document can be expanded with the final pipeline description, batching strategy, and storage format details once Q2 is implemented.
