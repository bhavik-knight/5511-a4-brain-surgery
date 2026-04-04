# Q1 - Model Wrapper and Hooks

Implementation file: `src/brain_surgery/model_wrapper.py`

## Theoretical Goal

Expose internal transformer computations as analyzable signals by hooking the
residual stream at a middle layer. This creates the bridge between black-box
language generation and mechanistic analysis.

Let $h\_\\ell$ denote hidden states at layer $\\ell$. Q1 operationalizes:

$$
ext{text prompt} \\rightarrow h\_\\ell \\rightarrow \\text{token-aligned activation rows}
$$

These rows become the substrate for SAE training (Q3), feature interpretation
(Q4/Q5), and interventions (Q6).

## Implementation

- Uses native PyTorch forward hooks (`register_forward_hook`) on the selected
  transformer block (default mid-layer for Qwen-2.5-0.5B).
- Loads model/tokenizer offline from local weights for reproducible execution.
- Captures token-aligned activations and preserves prompt/token metadata needed
  by the NDJSON-driven downstream pipeline.
- Persists outputs in run-consistent project directories so later stages can
  reuse artifacts without path ambiguity.

## Why This Matters

- Residual stream activations provide a stable representation for SAE fitting.
- Hooking during autoregressive generation captures evolving context effects.
- Strict token-to-row alignment is required for causal and semantic audits.

## Current Status

Q1 is complete and provides the activation interface consumed by Q2-Q6.
