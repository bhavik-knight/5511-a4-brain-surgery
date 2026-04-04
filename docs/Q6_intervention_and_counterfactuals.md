# Q6 - Intervention and Counterfactuals

Implementation file: `src/brain_surgery/intervention.py`

## Theoretical Goal

Test causal influence of latent features by forcing feature activation during
generation and comparing next-token probabilities with controlled baselines.

Intervention rule:

$$
z_j \\leftarrow \\alpha \\cdot \\max(z_j)
$$

where $j$ is a target feature and $\\alpha$ is a clamp multiplier.

## Implementation

- Supports checkpoint-aware SAE loading in `SAEIntervention` for robust run
  portability.
- Hooks into generation-time hidden states, applies encode-clamp-decode
  transformation, and writes modified activations back into the forward pass.
- Compares baseline logits with target-feature and control-feature clamps.

## Results

- Q6 reporting uses a specificity logic:

$$
\\Delta\_{\\text{target}} = p\_{\\text{target}} - p\_{\\text{baseline}}, \\quad
\\Delta\_{\\text{control}} = p\_{\\text{control}} - p\_{\\text{baseline}}
$$

$$
\\mathrm{Specificity\\ Score} = \\Delta\_{\\text{target}} - \\Delta\_{\\text{control}}
$$

- Positive specificity suggests the target feature has a stronger causal effect
  than an unrelated control feature.
- Intervention outputs are stored per run in
  `results/experiments/<run_id>/intervention_results.csv` and summarized in the
  executive report pipeline.

## Code Entry Points

- Intervention core: `src/brain_surgery/intervention.py` -> `SAEIntervention`
- Next-token scoring: `src/brain_surgery/intervention.py` -> `compare_next_token_logprobs(...)`
- Q6 phase runner: `scripts/verify_pilot.py` -> `run_phase_q6(...)`
