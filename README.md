# Brain Surgery: Mechanistic Interpretability with SAEs

A graduate-level deep learning project implementing Sparse Autoencoders (SAEs) to extract interpretable features from small language models, following Anthropic's *[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)* paper.

## 📋 Project Overview

This project investigates mechanistic interpretability by training Sparse Autoencoders to decompose activations from small LLMs (e.g., Qwen-2.5-0.5B) into interpretable features. We then intervene in the forward pass to validate the causal importance of discovered features.

**Key learning outcomes:**
- Forward hooks and activation capture from transformer residual streams
- SAE training with sparsity constraints (L₁ regularization)
- Feature interpretation and analysis via backwards mapping
- Counterfactual experiments with activation clamping

## 🛠️ Installation

### Requirements
- **Python:** 3.12+
- **Hardware:** NVIDIA GPU recommended (RTX 4070 or equivalent, 12GB+ VRAM)
- **Package Manager:** [uv](https://docs.astral.sh/uv/)

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd a4-brain-surgery
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

3. **Install pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

4. **Verify the environment:**
   ```bash
   uv run python --version
   uv run pytest tests/
   ```

## 📁 Project Structure

```
a4-brain-surgery/
├── src/
│   └── brain_surgery/
│       ├── __init__.py
│       ├── model_wrapper.py          # Q1: ModelWrapper with hooks
│       ├── activation_capture.py     # Q2: Data loader & activation saving
│       ├── sparse_autoencoder.py     # Q3: SAE architecture & training
│       ├── feature_interpretation.py # Q4: Backwards mapping & analysis
│       ├── feature_labeling.py       # Q5: bonus: LLM-based feature labels
│       └── intervention.py           # Q6: Clamping & counterfactual exps
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
├── ruff.toml                         # Linting & formatting rules
├── mypy.ini                          # Type checking config
├── .pre-commit-config.yaml           # Pre-commit hooks
├── CONTRIBUTING.md                   # Contribution guidelines
├── WORK_DISTRIBUTION.md              # Team roles & responsibilities
└── README.md                         # This file
```

## 📚 Assignment Questions & Implementation Status

### **Q1: Forward Hooks & Activation Capture**
Implement a `ModelWrapper` class that registers forward hooks on the residual stream of a middle transformer layer.

**File:** `src/brain_surgery/model_wrapper.py`

- [ ] Load a small LLM (Qwen-2.5-0.5B) using Hugging Face Transformers
- [ ] Register hooks on layer N's residual stream
- [ ] Implement `generate_with_activations()` to capture hidden states during generation
- [ ] Document hook placement rationale and approach limitations

**Key Classes:**
- `ModelWrapper(model_name: str, layer_idx: int)`: Wraps the model with hooks
- `generate_with_activations(prompt: str, max_tokens: int) -> tuple[str, Tensor]`: Returns generated text and corresponding activations

---

### **Q2: Data Collection & Activation Storage**
Build a data pipeline to load a corpus and save computed activations with token-to-activation mappings.

**File:** `src/brain_surgery/activation_capture.py`

- [ ] Create a data loader for a corpus (TinyStories or WikiText)
- [ ] Process batches through the model and capture activations
- [ ] Save activations to disk with metadata (tokens, positions, layer info)
- [ ] Maintain a token-to-activation index for later analysis

**Key Classes:**
- `CorpusDataLoader(corpus_name: str, batch_size: int)`: Yields tokenized batches
- `ActivationCollector`: Saves and indexes activations

**Output:** Summary statistics on dataset (corpus size, token distribution, activation shapes)

---

### **Q3: Sparse Autoencoder Training**
Implement a SAE with an Encoder-Decoder architecture and tied weights. Train with L₁ regularization.

**File:** `src/brain_surgery/sparse_autoencoder.py`

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

---

### **Q4: Feature Interpretation & Analysis**
Implement backwards mapping logic and manually identify 3–5 distinct interpretable features.

**File:** `src/brain_surgery/feature_interpretation.py`

- [ ] Implement backwards mapping: Feature → Activation → Token
- [ ] Identify top activating combinations (feature + token context)
- [ ] Manually inspect and label 3–5 features (e.g., "Python code," "Emotional tone," "Mathematical expressions")
- [ ] Collect text snippets demonstrating each feature

**Key Classes:**
- `FeatureInterpreter`: Maps latent dimensions to text patterns
- Feature report with examples and activation distributions

---

### **Q5: Bonus – LLM-based Feature Labeling**
Query a larger LLM (via API) to automatically label discovered features.

**File:** `src/brain_surgery/feature_labeling.py`

- [ ] Query a larger model (e.g., Claude, GPT-4) with context snippets
- [ ] Ask the model to suggest semantic labels for feature activations
- [ ] Build a correlation matrix / clustering of the feature space
- [ ] Visualize feature relationships

---

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
2. Clamp feature X to zero: Does output change semantically?
3. Clamp multiple features: Do effects interact?
4. Clamp to random values: Ablation control

---

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

2. **Run Q1 – Model Wrapping:**
   ```bash
   uv run python src/brain_surgery/model_wrapper.py --model qwen-2.5-0.5b --layer 8
   ```

3. **Run Q2 – Data Collection:**
   ```bash
   uv run python src/brain_surgery/activation_capture.py --corpus tinystories --batch_size 32
   ```

4. **Run Q3 – SAE Training:**
   ```bash
   uv run python src/brain_surgery/sparse_autoencoder.py --input_dim 512 --hidden_dim 4096 --l1_coef 0.001
   ```

5. **Run Q4 – Feature Analysis:**
   ```bash
   uv run python src/brain_surgery/feature_interpretation.py --num_features 5
   ```

6. **Run Q6 – Intervention:**
   ```bash
   uv run python src/brain_surgery/intervention.py --features_to_clamp 0,1,2 --num_trials 50
   ```

## 📊 Results & Outputs

All results are saved to the `results/` directory:
- **Training metrics:** Loss curves, sparsity plots
- **Features:** Identified features with example text snippets
- **Experiments:** Counterfactual results showing causal effects
- **Visualizations:** Feature correlation matrices, activation distributions

Example outputs:
```
results/
├── features/
│   ├── feature_0_python_code.txt
│   ├── feature_1_emotional_tone.txt
│   └── ...
├── metrics/
│   ├── sae_training_loss.png
│   └── sparsity_over_epochs.png
└── experiments/
    └── clamping_effects_summary.json
```

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

---

**Course:** MCDA 5511 – Deep Learning (Winter 2026)  
**Instructor:** Greg Kirczenow  
**Team Members:** TBD
