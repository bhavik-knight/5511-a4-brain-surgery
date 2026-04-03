# Brain Surgery Project

A graduate-level deep learning project implementing Sparse Autoencoders (SAEs) to extract interpretable features from small language models, following Anthropic's *[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)* paper.

## 📋 Project Overview

This project investigates mechanistic interpretability by training Sparse Autoencoders to decompose activations from small LLMs (e.g., Qwen-2.5-0.5B) into interpretable features. We then intervene in the forward pass to validate the causal importance of discovered features.

**Key learning outcomes:**

- Forward hooks and activation capture from transformer residual streams
- SAE training with sparsity constraints (L₁ regularization)
- Feature interpretation and analysis via backwards mapping
- Counterfactual experiments with activation clamping

## Documentation

| Assignment Question | Documentation                                                                                  | Source Code                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Q1                  | [./docs/Q1_model_and_hooks.md](./docs/Q1_model_and_hooks.md)                                   | [./src/brain_surgery/model_wrapper.py](./src/brain_surgery/model_wrapper.py) |
| Q2                  | [./docs/Q2_data_collection_and_storage.md](./docs/Q2_data_collection_and_storage.md)           | [./src/brain_surgery/data_gen.py](./src/brain_surgery/data_gen.py)           |
| Q3                  | [./docs/Q3_sae_training.md](./docs/Q3_sae_training.md)                                         | [./src/brain_surgery/sae.py](./src/brain_surgery/sae.py)                     |
| Q4                  | [./docs/Q4_feature_interpretation.md](./docs/Q4_feature_interpretation.md)                     | [./src/brain_surgery/interpret.py](./src/brain_surgery/interpret.py)         |
| Q5                  | [./docs/Q5_feature_labeling_bonus.md](./docs/Q5_feature_labeling_bonus.md)                     | [./src/brain_surgery/interpret.py](./src/brain_surgery/interpret.py)         |
| Q6                  | [./docs/Q6_intervention_and_counterfactuals.md](./docs/Q6_intervention_and_counterfactuals.md) | [./src/brain_surgery/intervention.py](./src/brain_surgery/intervention.py)   |

## 🛠️ Installation

### Requirements

- **Python:** 3.12+
- **Hardware:** NVIDIA GPU recommended (RTX 4070 or equivalent, 12GB+ VRAM)
- **Package Manager:** [uv](https://docs.astral.sh/uv/)

### Setup

1. **Clone and navigate to the project:**

   ```bash
   git clone https://github.com/bhavik-knight/5511-a4-brain-surgery/ a4-brain-surgery
   cd a4-brain-surgery
   ```

1. **Install dependencies using uv:**

   ```bash
   uv sync
   ```

1. **Install pre-commit hooks (optional but recommended):**

   ```bash
   pre-commit install
   ```

1. **Verify the environment:**

   ```bash
   uv run python --version
   uv run pytest tests/
   ```

## Offline Model Setup (Hugging Face)

This project loads Qwen **directly via `transformers`** from a local directory (`./models/qwen2.5-0.5b`) so we can register native PyTorch forward hooks.

### 24-Layer Qwen-2.5-0.5B Setup (Recommended)

- Qwen-2.5-0.5B has **24 transformer layers** and **896 hidden dimension**.
- The default activation layer is the midpoint: **layer 12** (0-indexed).
- Verified activation shape from `ModelWrapper` should be `(N, 896)`.

1. Download once (requires internet):

```bash
uv run hf download Qwen/Qwen2.5-0.5B --local-dir ./models/qwen2.5-0.5b
```

If you are not using `uv`, install the CLI with `pip install huggingface-hub` and run:

```bash
hf download Qwen/Qwen2.5-0.5B --local-dir ./models/qwen2.5-0.5b
```

If you prefer the official Hugging Face CLI, use:

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ./models/qwen2.5-0.5b
```

1. Run offline:

- The code uses `local_files_only=True` when calling `from_pretrained(...)`.
- `brain_surgery.main` sets `TRANSFORMERS_OFFLINE=1` so accidental network calls fail fast.

## Quick Start (Q1 Verification)

Run the Q1 smoke test to verify forward hooks + activation saving locally:

Prerequisite: you must have a local model directory under `models/` (see “Offline Model Setup (Hugging Face)” above).

```bash
uv run python -m brain_surgery.model_wrapper
```

## Why PyTorch?

This project uses **native PyTorch + Hugging Face Transformers** (not an external LLM runtime) because we need **low-level forward hooks** (`nn.Module.register_forward_hook`) to capture internal residual-stream activations. Those activations are required for SAE training, feature interpretation, and intervention experiments.

## Local Development

For long sessions on an NVIDIA RTX 4070 (12GB VRAM), you generally want to **load the model once** and keep the Python process alive.

Run the project entrypoint in interactive mode:

```bash
uv run python -i -m brain_surgery.main
```

## How to Run (Pure uv)

Minimum project flow:

```bash
uv sync
uv run pytest tests/
uv run python -m brain_surgery.model_wrapper
uv run python scripts/train_university.py
```

5-epoch pilot with explicit overrides:

```bash
uv run python scripts/train_university.py --epochs 5 --batch-size 256 --lr 1e-4 --l1 2e-3 --patience 10 --dataset data/activations/soccer_activations_dataset.pt --checkpoint models/sae_checkpoint.pt
```

Developer guide:

- See `DEVELOPER_GUDIE.md` for architecture mapping, command rationale, and test coverage notes.

Then keep the interactive session open for as long as needed (e.g., 4 hours). As long as the process stays alive and you don’t re-instantiate the model repeatedly, the weights remain resident in VRAM.

### Memory Management

- Avoid launching multiple Python processes that each load the model (VRAM will fill quickly).
- Reuse a single `ModelWrapper` instance inside loops.
- When you’re done capturing activations, call `unregister_hooks()` to release hook handles and clear cached activations.

## Data, Outputs, and Storage Locations

This repo keeps all large and/or generated artifacts in a few predictable folders:

- **Model weights (offline):** `./models/`
  - Default target: `./models/qwen2.5-0.5b/`
  - Downloaded once, then loaded with `local_files_only=True`.
- **Input data (corpus):** `./data/corpus/`
- **Captured activations:** `./data/activations/` (saved as `.pt` artifacts)
- **Assignment PDF:** `./data/assignment-4-brain-surgery.pdf`
- **Results/outputs:** `./results/`
  - `./results/metrics/` — training curves + evaluation metrics
  - `./results/features/` — extracted features + human/auto interpretations
  - `./results/experiments/` — intervention/counterfactual experiment outputs

## 📁 Project Structure

```text
a4-brain-surgery/
├── docs/
│   └── Q1_model_and_hooks.md        # Q1 write-up and rationale
├── models/                           # Local Hugging Face model directories (offline)
├── src/
│   └── brain_surgery/
│       ├── __init__.py
│       ├── model_wrapper.py          # Q1: ModelWrapper with hooks
│       ├── data_gen.py               # Q2: Corpus + activation collection
│       ├── sae.py                    # Q3: SAE architecture + training
│       ├── interpret.py              # Q4/Q5: Feature interpretation
│       ├── intervention.py           # Q6: Clamping & counterfactual exps
│       ├── main.py                   # Interactive entrypoint (keep model in VRAM)
│       └── utils.py                  # Shared configs + paths
├── tests/
│   ├── test_model_wrapper.py
│   ├── test_activation_capture.py
│   ├── test_sae.py
│   └── test_intervention.py
├── data/
│   ├── assignment-4-brain-surgery.pdf
│   ├── corpus/                       # TinyStories or WikiText data
│   └── activations/                  # Cached activations
├── results/
│   ├── features/                     # Extracted features & interpretations
│   ├── metrics/                      # Training curves, evaluation metrics
│   └── experiments/                  # Counterfactual experiment results
├── pyproject.toml                    # Dependencies & project config
├── pyrightconfig.json                # Pyright/Pylance type checking config
├── uv.lock                           # Locked dependency versions for uv
├── .pre-commit-config.yaml           # Pre-commit hooks
├── CONTRIBUTING.md                   # Contribution guidelines
├── WORK_DISTRIBUTION.md              # Team roles & responsibilities
└── README.md                         # This file
```

## 📚 Assignment Questions & Implementation Status

### **Q1: Forward Hooks & Activation Capture**

Implement a `ModelWrapper` class that registers forward hooks on the residual stream of a middle transformer layer.

**File:** `src/brain_surgery/model_wrapper.py`

- [x] Load a small LLM (Qwen-2.5-0.5B) using Hugging Face Transformers
- [x] Register hooks on layer N's residual stream
- [x] Implement `generate_with_activations()` to capture hidden states during generation
- [x] Document hook placement rationale and approach limitations

**Key Classes:**

- `ModelWrapper(model_name: str, layer_idx: int)`: Wraps the model with hooks
- `generate_with_activations(prompt: str, max_tokens: int) -> tuple[str, Tensor]`: Returns generated text and corresponding activations

______________________________________________________________________

### **Q2: Data Collection & Activation Storage**

Build a data pipeline to load a corpus and save computed activations with token-to-activation mappings.

**File:** `src/brain_surgery/data_gen.py`

- [ ] Create a data loader for a corpus (TinyStories or WikiText)
- [ ] Process batches through the model and capture activations
- [ ] Save activations to disk with metadata (tokens, positions, layer info)
- [ ] Maintain a token-to-activation index for later analysis

**Key Classes:**

- `CorpusDataLoader(corpus_name: str, batch_size: int)`: Yields tokenized batches
- `ActivationCollector`: Saves and indexes activations

**Output:** Summary statistics on dataset (corpus size, token distribution, activation shapes)

______________________________________________________________________

### **Q3: Sparse Autoencoder Training**

Implement a SAE with an Encoder-Decoder architecture and tied weights. Train with L₁ regularization.

**File:** `src/brain_surgery/sae.py`

- [ ] Implement `SparseAutoencoder(input_dim: int, hidden_dim: int)` with tied encoder/decoder weights
- [ ] Add L₁ loss term: `loss = mse_loss + λ * L1_norm(latents)`
- [ ] Implement training utilities: early stopping, learning rate scheduling, metric tracking
- [ ] Log training metrics (reconstruction loss, sparsity, loss curves)

**Key Hyperparameters to Document:**

- Encoder hidden dimension (expansion factor)
- L₁ coefficient (λ)
- Learning rate schedule
- Early stopping patience

**Key Concepts to Explain:**

- Difference between "activation space" and "feature space"
- Why tied weights help with interpretability
- Role of sparsity in preventing feature collapse

______________________________________________________________________

### **Q4: Feature Interpretation & Analysis**

Implement backwards mapping logic and manually identify 3–5 distinct interpretable features.

**File:** `src/brain_surgery/interpret.py`

- [ ] Implement backwards mapping: Feature → Activation → Token
- [ ] Identify top activating combinations (feature + token context)
- [ ] Manually inspect and label 3–5 features (e.g., "Python code," "Emotional tone," "Mathematical expressions")
- [ ] Collect text snippets demonstrating each feature

**Key Classes:**

- `FeatureInterpreter`: Maps latent dimensions to text patterns
- Feature report with examples and activation distributions

______________________________________________________________________

### **Q5: Bonus – LLM-based Feature Labeling**

Query a larger LLM (via API) to automatically label discovered features.

**File:** `src/brain_surgery/interpret.py`

- [ ] Query a larger model (e.g., Claude, GPT-4) with context snippets
- [ ] Ask the model to suggest semantic labels for feature activations
- [ ] Build a correlation matrix / clustering of the feature space
- [ ] Visualize feature relationships

______________________________________________________________________

### **Q6: Intervention & Counterfactual Experiments**

Implement activation clamping to validate feature importance via causality.

**File:** `src/brain_surgery/intervention.py`

- [ ] Implement a `CampingInterventionHook`: Replace activations with clamped values
- [ ] Run baseline generation (no intervention)
- [ ] Run clamped generations (fix specific features)
- [ ] Compare outputs and measure differences (BLEU, perplexity, semantic similarity)
- [ ] Document causal effects of features

**Experiments to Run:**

1. Baseline: Generate freely
1. Clamp feature X to zero: Does output change semantically?
1. Clamp multiple features: Do effects interact?
1. Clamp to random values: Ablation control

______________________________________________________________________

## 🎯 Key Concepts & Definitions

### Activation Space vs. Feature Space

- **Activation Space:** Raw hidden states from the transformer's residual stream (high-dimensional, entangled)
- **Feature Space:** Interpretable dimensions learned by the SAE (sparse, disentangled)

### Sparsity

L₁ regularization forces most latent dimensions to zero, promoting interpretability. Only a small subset of features activate per example.

### Tied Weights

The encoder and decoder share weights (up to transpose), reducing parameters and regularizing the learned feature space for better interpretability.

## 🚀 Getting Started

1. **Setup & Verify:**

   ```bash
   uv sync
   pre-commit install
   uv run pytest tests/
   ```

1. **Run Q1 – Model Wrapping:**

   ```bash
   uv run python -m brain_surgery.model_wrapper
   ```

1. **Q2–Q6:**

   - These modules are scaffolded (`src/brain_surgery/data_gen.py`, `sae.py`, `interpret.py`, `intervention.py`) but the runnable pipelines/CLIs are implemented as each question is completed.
   - Output conventions are documented in `docs/` and summarized in the “Data, Outputs, and Storage Locations” section above.

## 📊 Results & Outputs

All results are saved under `results/` using this convention:

- **Training metrics:** Loss curves, sparsity plots
- **Features:** Identified features with example text snippets
- **Experiments:** Counterfactual results showing causal effects
- **Visualizations:** Feature correlation matrices, activation distributions

## 📖 References

- **Paper:** Anthropic's [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- **Dependencies:**
  - PyTorch: Deep learning framework
  - Hugging Face Transformers: Pre-trained LLMs
  - pydantic: Data validation
  - ruff: Linting & formatting
  - mypy: Static type checking

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for code standards, branching strategy, and pull request guidelines.

See [WORK_DISTRIBUTION.md](WORK_DISTRIBUTION.md) for team roles and task assignments.

## 📝 License

See [LICENSE](LICENSE) for details.

______________________________________________________________________

**Course:** MCDA 5511 – Deep Learning (Winter 2026)
**Instructor:** Mr. Greg Kirczenow, Mr. Somto Muetoe, Dr. Drira, Prof. Neveditsin
**Team Members:**

1. Bhavik Kantilal Bhagat
1. Miguel Angel Palafox Gomez
1. Nikola Kriznar
1. Sridhar Vadla
