# Q1 — Model Wrapper & Forward Hooks

## Goal

Capture transformer internal activations during generation so we can:

- train Sparse Autoencoders (SAEs) on a consistent activation stream
- later map learned SAE features back to text snippets
- run interventions (clamping/ablation) against the same representation

In this project, the implementation lives in `src/brain_surgery/model_wrapper.py`.

## Implementation Notes (Offline + Native Hooks)

- **Offline model loading:** The model and tokenizer are loaded from a local directory (default: `./models/qwen2.5-0.5b`) using `AutoModelForCausalLM.from_pretrained(..., local_files_only=True)` and `AutoTokenizer.from_pretrained(..., local_files_only=True)`.
- **Standard PyTorch hook architecture:** Activation capture is implemented using `nn.Module.register_forward_hook` on the selected transformer block. This is the native mechanism that provides access to residual-stream activations for SAE training and downstream interventions.

______________________________________________________________________

## What We Capture (Residual Stream Activations)

Transformers compute token representations layer-by-layer. A common “highway” representation is the **residual stream**: the running hidden state that gets updated by attention and MLP blocks with skip connections.

For a given layer $\\ell$, you can think of the residual stream (informally) as:

$$
\\mathbf{r}_{\\ell+1} = \\mathbf{r}_\\ell + \\Delta^{\\text{attn}}_\\ell(\\mathbf{r}_\\ell) + \\Delta^{\\text{mlp}}_\\ell(\\mathbf{r}_\\ell)
$$

Hooking the residual stream provides a relatively stable, information-rich activation space for SAE training.

______________________________________________________________________

## Why Hook a Middle Layer?

Empirically and conceptually, **middle layers** often provide a good tradeoff:

- **Early layers**: closer to token/lexical features; may be too “local” or syntactic
- **Late layers**: more specialized toward the model’s final prediction objective; features can become more entangled with the output head and task-specific circuitry
- **Middle layers**: frequently contain a blend of semantics + syntax with enough abstraction to form reusable features while still being relatively interpretable

For Qwen-2.5-0.5B (24 layers), we default to a mid-layer (e.g., layer 12).

______________________________________________________________________

## Why Residual Stream (and not only MLP outputs)?

The residual stream is the shared substrate across blocks. Capturing residual activations aligns well with a common interpretability workflow:

- train an SAE on a single activation stream
- interpret SAE features as directions/components in that stream
- intervene by editing that same stream

This is consistent with the general framing used in mechanistic interpretability work, including Anthropic’s *Scaling Monosemanticity* write-up (see references).

______________________________________________________________________

## Token-to-Activation Mapping

To interpret features later, every saved activation needs to be aligned with a token/text snippet.

Our saved artifact includes:

- `token_ids`: token IDs produced by the tokenizer
- `token_strs`: tokenizer token strings (may include BPE markers)
- `token_texts`: per-token decoded text snippet (1 token at a time)
- `activations`: a 2D tensor shaped `(seq_len, hidden_dim)`

Alignment invariant:

- `token_texts[i]` corresponds to `activations[i, :]`

This enables later workflows like “find top-activating tokens for feature $k$”.

______________________________________________________________________

## Limitations / Caveats

1. **Hook placement is architecture-dependent**

   - Different HF model classes expose layers differently (e.g., `model.model.layers` vs `transformer.h`). We handle common cases, but the exact module to hook can vary across models.

1. **Generation calls the model multiple times**

   - During autoregressive generation, the model is invoked step-by-step. After the first step, many models run with `seq_len=1` using KV-cache.
   - If you only store the latest hook output, you’ll end up with a tiny tensor (often `(1, 1, hidden_dim)`). We therefore **accumulate** per-step activations and concatenate them.

1. **Residual stream is still a mixture**

   - Even if “monosemantic” features exist, most real networks contain mixed features and polysemantic directions.
   - SAEs improve interpretability, but they are not guaranteed to produce perfectly disentangled units.

1. **Tokenization artifacts**

   - `token_strs` and `token_texts` depend on tokenizer behavior (BPE, special tokens). “Text snippet per token” is useful for alignment, but it’s not always human-readable.

1. **Memory and storage footprint**

   - Saving full-sequence hidden states across many prompts can create large artifacts.
   - We save in `.pt` and automatically update `.gitignore` for artifacts > 50MB to avoid repo bloat.

______________________________________________________________________

## References

- Anthropic — *Scaling Monosemanticity* (Transformer Circuits):
  - https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
