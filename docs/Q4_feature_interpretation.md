# Q4 - Feature Interpretation

Implementation file: `src/brain_surgery/interpret.py`

## Theoretical Goal

Map latent SAE dimensions to interpretable semantic evidence by ranking the
token contexts that maximally activate each feature.

Interpretation operator:

$$
\mathrm{Feature}\ j \Rightarrow \operatorname*{TopK}_{i}(z_{i,j})
$$

where the highest-activation rows provide candidate meanings for feature $j$.

## Implementation

- Loads run-specific checkpoint + activation dataset.
- Computes latent matrix and ranks top activating token rows per feature.
- Exports top feature tables for pilot auditing and reporting.

## Results

- Interpretation now feeds directly into Spherical K-Means preprocessing used in
  Q5 (L2-normalized feature profiles).
- Top-activation evidence is used as semantic support when assigning cluster
  themes and validating cluster purity by metadata category.
- Output artifacts are integrated into run-scoped reports rather than global,
  ambiguous paths.

## Code Entry Points

- Module: `src/brain_surgery/interpret.py`
- Latent computation: `SAEInterpreter.compute_latents(...)`
- Feature ranking: `SAEInterpreter.rank_features_by_max_activation(...)`
- Evidence lookup: `SAEInterpreter.get_top_examples_for_feature(...)`
