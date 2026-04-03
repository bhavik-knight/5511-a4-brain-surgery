# Developer's Guide: Structural Audit for a4-brain-surgery

This guide maps each assignment requirement (Q1-Q6) to the current codebase and explains the full tensor lifecycle.

## 0. Quick Commands (Pure uv + Cluster CLI)

Environment and install:

- `uv sync`
  - Creates/updates `.venv` from `pyproject.toml` and lock state, and installs the local package in editable mode through uv project settings.

Minimum viable command:

- `uv run python scripts/train_university.py`
  - Uses built-in defaults for dataset/checkpoint and training hyperparameters.

Run training (headless cluster style):

- `uv run python scripts/train_university.py --dataset data/activations/soccer_activations_dataset.pt --checkpoint models/sae_checkpoint.pt --epochs 100 --batch-size 256 --lr 1e-4 --l1 2e-3 --patience 10 --resume --wandb-project a4-brain-surgery --wandb-run-name run_a --tensorboard-dir runs`

Required vs Optional (`scripts/train_university.py`):

- Required:
  - None. The script runs with defaults when your dataset exists at `data/activations/soccer_activations_dataset.pt`.
- Optional:
  - All flags below are optional overrides.

Argument reference:

- `--dataset`: Input `.pt` file containing `activation_matrix`.
- `--checkpoint`: Checkpoint path for load/save.
- `--epochs`: Number of training epochs.
- `--batch-size`: Mini-batch size.
- `--lr`: Adam learning rate.
- `--l1`: SAE sparsity coefficient.
- `--patience`: Early-stopping patience in epochs.
- `--resume`: Resume from checkpoint when present.
- `--no-wandb`: Disable WandB and rely on TensorBoard logging.
- `--wandb-project`: WandB project override.
- `--wandb-run-name`: Optional run label.
- `--tensorboard-dir`: TensorBoard event log directory.

Run tests:

- `uv run pytest tests`
  - Runs the full assignment/unit test suite.

## 1. The Life of a Tensor (Data Flow)

1. Prompt source and corpus intent:

- Assignment intent points to `data/corpus/`, but the active 118-prompt production list is currently embedded in `src/brain_surgery/data_gen.py` (`load_corpus()`).

2. Model hook capture:

- Model and layer/device defaults are configured in `src/brain_surgery/model_wrapper.py`.
- The hook is registered with `register_forward_hook(...)`.
- Hook callback captures residual activations and appends per-step tensors to `_activation_steps`.

3. Dataset assembly:

- Batch/prompt loop and activation collection happen in `src/brain_surgery/data_gen.py` (`generate_dataset()`).
- Token/activation alignment metadata is built with token text/str/id + prompt/layer fields.
- Consolidated dataset tensor is saved as `data/activations/soccer_activations_dataset.pt`.

4. Latent projection (SAE):

- Encoder projects activations to 3584 sparse latents in `src/brain_surgery/sae.py` (`encode()`).

5. Discovery/interpretation:

- Latents for all tokens are computed in `src/brain_surgery/interpret.py` (`compute_latents()`).
- Feature ranking and top examples are done via `rank_features_by_max_activation()` and `get_top_examples_for_feature()`.

## 2. File-to-Question Mapping

### Q1: Hooks

24-layer logic:

- Midpoint defaulting and bounds checks are in `ModelWrapper.__init__` in `src/brain_surgery/model_wrapper.py`.
- Dynamic layer count property is `total_layers`.

Hook type:

- Current implementation uses `register_forward_hook(...)`.
- There is no `forward_pre_hook` path in the current file.

### Q2: Dataset and token-activation matching

Alignment mechanism:

- Uses token metadata emitted by ModelWrapper and truncates with `seq_len = min(...)` to shared token/activation length.
- Per-token metadata rows include `prompt_id`, `prompt_text`, `token_id`, `token_text`, `token_str`, generated text, and hook layer info.

Persisted dataset:

- Final save includes `activation_matrix`, `metadata`, prompt count, token count, and hidden dim in `soccer_activations_dataset.pt`.

### Q3: SAE architecture

Tied weights:

- Decoder uses encoder transpose: `z @ self.encoder_weight.T + self.decoder_bias` in `src/brain_surgery/sae.py`.

L1 loss:

- Implemented in `compute_loss(...)` as:
  - `reconstruction_loss = MSE(...)`
  - `l1_loss = mean(abs(z))`
  - `total_loss = reconstruction_loss + l1_lambda * l1_loss`

Trainer objective and optimizer:

- Adam optimizer in `src/brain_surgery/trainer.py`.
- Training loop uses SAE `compute_loss(...)` outputs.
- Dead-neuron metric computed from zero-activation latent units each epoch, with explicit epoch-1 print.

### Q4 & Q5: Interpretability

Feature-to-token discovery:

- `SAEInterpreter` loads checkpoint + dataset metadata, computes latents, and retrieves top activating tokens per feature.

Concept grouping:

- `src/brain_surgery/clustering.py` runs KMeans across latent feature profiles.
- It extracts representative features/tokens per cluster to produce human-readable conceptual summaries.

### Q6: Intervention

Clamp mechanics:

- `register_prompt_clamp_hook(...)` installs a hook on the middle transformer block.
- Hook flow: hidden states -> SAE encode -> set chosen latent index to clamped value -> SAE decode -> return modified hidden states.

Generation with intervention:

- `generate_with_clamped_feature(...)` runs text generation under active hook and returns diagnostics.

Next-token probability change:

- `compare_next_token_logprobs(...)` computes baseline vs clamped next-token log-probabilities for candidate soccer tokens.

## 3. The Headless Layer (Cluster-Friendly Execution)

Environment bootstrap:

- `scripts/train_university.py` imports `load_dotenv` and calls `load_dotenv()` at startup.

CLI/no-manual mode:

- Argparse flags control epochs, L1, LR, resume, WandB usage, and TensorBoard directory.

Env-driven WandB:

- Trainer reads:
  - `WANDB_API_KEY` (login)
  - `WANDB_PROJECT` (project override)
  - `WANDB_ENTITY` (entity/org)

Template for server setup:

- `.env.example` provides the required keys.

OOM crash logging:

- `train_university.py` catches CUDA/runtime OOM paths and appends stack traces to `crash_report.log`.

## 4. Safety Mechanisms (VRAM + Dtype)

VRAM protections:

- `ModelWrapper` defaults `activation_device="cpu"` and moves captured activations off-GPU.
- On CUDA, model loads in FP16 with `device_map="auto"` for memory efficiency.

Dtype safety:

- SAE normalizes input tensor dtype/device in `to_device_and_dtype(...)`.
- SAE loss compares reconstruction against dtype-normalized input.
- Intervention hook casts modified hidden states back to original model activation dtype before returning.

## 5. Relevant Test Coverage

Core suite entrypoint:

- Run all assignment tests with `pytest tests` from the `a4-brain-surgery` directory.

What is covered now:

- `tests/test_model_wrapper.py`: midpoint default layer selection for 24-layer models, activation shape checks, and loaded-state checks.
- `tests/test_data_gen.py`: 118-prompt corpus count and token/activation alignment behavior (`seq_len = min(...)`).
- `tests/test_sae.py`: 896 -> 3584 projection shape, tied-weight decode behavior, and L1 contribution in total loss.
- `tests/test_intervention.py`: clamp-hook activation modification, dtype preservation, and next-token log-prob delta verification.
- `tests/test_env.py`: dotenv loading for `WANDB_API_KEY` in headless workflows.
- `tests/test_activation_capture.py`: activation capture save/load and token-to-activation mapping checks.

Why this matters:

- These tests directly validate Q1-Q6 critical paths while remaining lightweight enough for local and cluster preflight checks.

## 6. Important Audit Note

Your assignment wording says prompts start in `data/corpus/`, but current implementation hardcodes the 118 prompts in `data_gen.py` (`load_corpus()`).

This works functionally. If strict traceability is needed, move prompts into a file under `data/corpus/` and make `load_corpus()` read from disk.
