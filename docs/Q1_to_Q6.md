# Q1 to Q6 - Theoretical Objectives and Implementation Audit

This consolidated audit summarizes the theoretical objective and current
technical implementation for each assignment question.

## Q1 - Model Wrapper and Hooks

### Objective

Expose internal model computation by capturing residual-stream activations at a
chosen layer, creating the observability required for mechanistic analysis.

### Audit Summary

- Uses forward hooks on the transformer stack.
- Preserves token-aligned activations for downstream SAE training.
- Establishes reproducible activation capture for all later questions.

## Q2 - Data Collection and Storage

### Objective

Build a trustworthy measurement dataset where every activation row has token and
semantic context metadata for later interpretation and validation.

### Audit Summary

- Uses curated NDJSON prompt records with schema-level structure.
- Produces activation payloads under `data/activations/` with metadata.
- Supports category-aware analyses used in Q5/Q6 reporting.

## Q3 - SAE Training

### Objective

Train a sparse latent model that reconstructs hidden activations while reducing
feature superposition.

$$
\mathcal{L} = \mathrm{MSE}(x, \hat{x}) + \lambda \lVert z \rVert_1
$$

### Audit Summary

- Run-scoped checkpoints and logs improve reproducibility.
- Smart checkpointing (`sae_best.pt`, `sae_final.pt`) supports long runs.
- Smoke-test mode validates execution safely before full training.

## Q4 - Feature Interpretation

### Objective

Map latent dimensions to interpretable semantic evidence by ranking strongest
activations and recovering associated token context.

### Audit Summary

- Computes latent activations and top feature evidence.
- Produces structured top-feature outputs for pilot analysis.
- Feeds evidence into Q5 cluster-level reporting.

## Q5 - Clustering and Semantic Validation

### Objective

Organize latent features into semantic groups and validate whether cluster
structure aligns with metadata categories.

### Audit Summary

- Uses Spherical K-Means (L2 normalization before K-Means).
- Selects cluster count with dynamic elbow analysis and visual elbow plot.
- Reports cluster purity and theme tags (`Tactical`, `Historical`, `Clubs`).

## Q6 - Intervention and Counterfactuals

### Objective

Test causal hypotheses about learned features by intervening on latent
activations and measuring downstream behavioral change.

### Causal Scrubbing Theory

Q6 follows a causal-scrubbing style intervention: a targeted internal variable
(latent feature activation) is replaced/clamped, then output behavior is
compared against baseline and control edits. If output changes are specific to
the targeted feature and not reproduced by controls, this is evidence that the
feature is causally involved in the behavior.

### Audit Summary

- Applies encode-clamp-decode intervention during generation.
- Compares baseline, target-clamp, and control-clamp token log-probability deltas.
- Uses specificity scoring to separate targeted causal effect from generic shifts.
