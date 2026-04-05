# Q1 - Model Wrapper and Hooks

Implementation file: `src/brain_surgery/model_wrapper.py`

## Qwen-2.5 Base Architecture (Pilot vs Golden)

The project uses dense, decoder-only Qwen-2.5 checkpoints with grouped-query
attention (GQA). Pilot experiments run on Qwen-2.5-0.5B, while high-fidelity
analysis targets Qwen-2.5-7B.

| Feature                           |   Qwen-2.5-0.5B (Pilots) | Qwen-2.5-7B (Golden Run) |
| --------------------------------- | -----------------------: | -----------------------: |
| Architecture                      | Decoder-only Transformer | Decoder-only Transformer |
| Layers                            |                       24 |                       28 |
| Hidden Dim ($d_{\mathrm{model}}$) |                      896 |                     3584 |
| Attention Heads                   |    14 Query / 2 KV (GQA) |    28 Query / 4 KV (GQA) |
| Activation Function               |                   SwiGLU |                   SwiGLU |
| Normalization                     |                  RMSNorm |                  RMSNorm |
| Positional Encoding               |                     RoPE |                     RoPE |

### Why Both Models Matter

- Qwen-2.5-0.5B is used for efficient hyperparameter calibration
  (learning rate and $L_1$ sweeps).
- Qwen-2.5-7B is the target model for high-fidelity feature extraction on A100.
- Shared architectural motifs support methodological transfer from pilot runs
  to large-scale runs.

## Theoretical Goal

Expose internal transformer computations as analyzable signals by hooking the
residual stream at a middle layer. This creates the bridge between black-box
language generation and mechanistic analysis.

Let $h_{\ell}$ denote hidden states at layer $\ell$. Q1 operationalizes:

$$
\mathrm{prompt} \rightarrow h_{\ell} \rightarrow \mathrm{token\ alignment\ rows}
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

## Code Entry Points

- Module: `src/brain_surgery/model_wrapper.py`
- Core generation hook path: `ModelWrapper.generate_with_activations(...)`
- Typical script-level launcher: `python -m brain_surgery.model_wrapper`
