# Q3 - Sparse Autoencoder Training

Implementation files:

- `src/brain_surgery/sae.py`
- `src/brain_surgery/trainer.py`
- `scripts/train_university.py`

## Theoretical Goal

Learn sparse latent features that reconstruct hidden activations while reducing
feature superposition.

Objective:

$$
\\mathcal{L} = \\text{MSE}(x, \\hat{x}) + \\lambda |z|\_1
$$

where reconstruction preserves information and $L_1$ regularization promotes
feature selectivity.

## Implementation

- Trains SAE on activation matrices produced by the NDJSON-driven Q2 pipeline.
- Uses run-scoped output directories (`results/experiments/<run_id>/...`) for
  checkpoints, logs, and metrics.
- Supports smart checkpointing:
  - `sae_best.pt` for best validation trajectory.
  - `sae_final.pt` and canonical checkpoint path for latest state.
  - optional auto-resume behavior for long cluster jobs.
- Supports `--smoke-test` in `train_university.py` to run a 1-epoch, 5-prompt
  subset verification before full training.

## Smart Checkpointing Rationale

- Protects long-running university cluster jobs from interruption loss.
- Enables deterministic downstream verification by pinning a run-specific best
  checkpoint.
- Makes Q4-Q6 reproducible by binding interpretation/intervention to a concrete
  model artifact.

## Current Status

Q3 is complete and produces reproducible checkpoints compatible with Q4-Q6.
