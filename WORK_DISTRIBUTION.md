# Work Distribution – Brain Surgery: Mechanistic Interpretability with SAEs

A 4-member team assignment with clearly defined roles and responsibilities. Each member owns 1–2 assignment questions and contributes to documentation.

______________________________________________________________________

## 👥 Team Structure

| Role                             | Member   | Assignment Questions | Focus Area                                 |
| -------------------------------- | -------- | -------------------- | ------------------------------------------ |
| **The Architect**                | Member 1 | Q1 & Repo Setup      | Model wrapping, hooks, initial pipeline    |
| **The Data Engineer**            | Member 2 | Q2 & Q6              | Data flow, serialization, intervention     |
| **The Deep Learning Researcher** | Member 3 | Q3                   | SAE architecture, training, optimization   |
| **The Interpretability Analyst** | Member 4 | Q4 & Q5 (Bonus)      | Feature analysis, LLM labeling, clustering |

______________________________________________________________________

## ✅ Completed

- [x] **2026-03-28** — **Member 1: Subtask 1 (Q1 + Repo Setup)** completed (ModelWrapper + hooks, `.pt` activation saving, and Q1 documentation).

## Member 1: The Architect 👷 (Q1 & Repo Setup)

**Primary Focus:** Model wrapping, forward hooks, initial PyTorch pipeline architecture.

**Subtask 1 (Q1) Status:** **COMPLETED** ✅ *(2026-03-28)*

### Assigned Tasks

#### Repository Initialization

- [x] Initialize repository with `uv init` *(2026-03-28)*
- [x] Set up a professional `pyproject.toml` with: *(2026-03-28)*
  - Project metadata
  - Dependencies: `torch`, `transformers`, `pydantic`, `pytest`
  - Development dependencies: `mypy`, `ruff`, `pre-commit`
- [x] Configure dev tools: `pyproject.toml` (ruff+mypy), `pyrightconfig.json`, `.pre-commit-config.yaml` *(2026-03-28)*
- [x] Create standard `.gitignore` excluding cache, models, results/ *(2026-03-28)*
- [x] Create `src/brain_surgery/` package structure *(2026-03-28)*

#### Q1: Model Wrapping & Forward Hooks

- [x] Implement `ModelWrapper` class using `transformers` library *(2026-03-28)*
  - Load Qwen-2.5-0.5B (or Qwen2-0.5B)
  - Accept `model_name` and `layer_idx` parameters
  - Register forward hooks on the specified layer's residual stream
- [x] Implement `generate_with_activations(prompt, max_tokens)` method *(2026-03-28)*
  - Returns tuple: `(generated_text, activations_tensor)`
  - Activations shape: `(seq_len, hidden_dim)`
- [x] Document: *(2026-03-28)*
  - Why register on residual stream vs. MLP output?
  - Limitations of hook-based approach (computational overhead, synchronization)
  - Choice of layer (middle layers recommended for balance)
  - Example usage with sample prompts

#### README Documentation

- [x] Write "Project Overview" section *(2026-03-28)*
- [x] Create "Installation" subsection with `uv sync` example *(2026-03-28)*
- [x] Document project structure tree *(2026-03-28)*
- [x] Provide "Getting Started" guidance (Q1 execution example) *(2026-03-28)*

### Deliverables

- `src/brain_surgery/model_wrapper.py` (fully typed, docstrings)
- `pyproject.toml` (tested dependencies)
- `pyproject.toml`, `pyrightconfig.json`, `.pre-commit-config.yaml` (working lint/type check)
- README "Installation" and "Project Structure" sections
- Example notebook or script demonstrating hook usage

**Deliverables Detail (2026-03-28):**

- `ModelWrapper` is functional (`generate_with_activations()` + `save_activations()`).
- Forward hooks verified end-to-end on **Qwen-2.5-0.5B** using an **RTX 4070**.
- Q1 documentation finalized in `docs/Q1_model_and_hooks.md`.

### Success Criteria

- [x] Code passes `ruff check` and `mypy --strict` *(2026-03-28)*
- [x] Hooks correctly capture activations for at least 3 test prompts *(2026-03-28)*
- [x] Type hints on all functions (including hooks and callbacks) *(2026-03-28)*
- [x] Google-style docstrings for `ModelWrapper` and `generate_with_activations` *(2026-03-28)*

______________________________________________________________________

## Member 2: The Data Engineer 💾 (Q2 & Q6)

**Primary Focus:** Data pipelines, activation serialization, and intervention logic.

### Assigned Tasks

#### Q2: Data Collection & Activation Storage

**Status:** Ready to Start

**Prerequisites:** Requires `ModelWrapper.save_activations()` to be called in a loop over the chosen corpus.

**Handoff Instructions:**

- Use `ModelWrapper.generate_with_activations()` to capture token-aligned activations per prompt/document.
- Use `ModelWrapper.save_activations()` inside that loop to write `.pt` artifacts under `data/activations/` (one per batch/prompt).
- Use these saved artifacts to implement the DataGenerator / activation collection pipeline for Q2.

**Technical Note:** Activations default to CPU for VRAM safety, while the model remains on GPU (when CUDA is available).

- [ ] Implement `CorpusDataLoader` class (using Q1's `ModelWrapper`)
  - Load corpus: TinyStories or WikiText (user choice)
  - Batch tokenization and padding
  - Yield batches for forward pass
- [ ] Implement `ActivationCollector` class
  - Capture activations for every example in corpus
  - Save to disk (HDF5 or pickle + metadata JSON)
  - Build token-to-activation index (maps token_id → activations)
- [ ] Compute and log corpus statistics:
  - Total tokens, unique vocabularies
  - Activation shapes and data types
  - Storage size estimates
  - Duration of collection

#### Q6: Intervention & Clamping

- [ ] Implement `CampingInterventionHook` class
  - Register a forward hook that replaces activations
  - Support clamping specific features to zero
  - Support clamping to random values (control)
- [ ] Implement `CounterfactualExperiment` runner
  - Baseline generation (no intervention)
  - Run interventions on selected features
  - Log outputs and compute similarity metrics
- [ ] Metrics computation:
  - BLEU score comparison
  - Semantic similarity (cosine distance of embeddings)
  - Perplexity shift
  - Token-level attention patterns (optional)
- [ ] Document experimental results:
  - Which features have causal effects?
  - Do feature effects interact?
  - Summary statistics and visualizations

#### README Documentation

- [ ] Write "Q2: Data Collection" section with corpus stats summary
- [ ] Write "Q6: Intervention & Experiments" section explaining clamping methodology
- [ ] Provide example commands for running data collection and interventions

### Deliverables

- `src/brain_surgery/data_gen.py` (data loader + activation collection)
- `src/brain_surgery/intervention.py` (clamping hooks + experiments)
- `results/metrics/corpus_statistics.json` (Q2 output)
- `results/experiments/clamping_effects_summary.json` (Q6 output)
- README sections for Q2 and Q6

### Success Criteria

- [ ] Data loader works with both TinyStories and WikiText
- [ ] Activations correctly saved; checksums match across loads
- [ ] Token-to-activation mapping is accurate and fast (~O(1) lookup)
- [ ] Intervention hook preserves model shape and dtype
- [ ] Counterfactual results show clear differences for causal features
- [ ] All code passes type checking and linting

______________________________________________________________________

## Member 3: The Deep Learning Researcher 🧠 (Q3)

**Primary Focus:** SAE architecture, training loop, hyperparameter tuning.

### Assigned Tasks

#### Q3: Sparse Autoencoder Architecture & Training

- [ ] Implement `SparseAutoencoder` class
  - Input: activation tensor from Q2 (shape: `[batch, hidden_dim]`)
  - Encoder: Linear layer to `hidden_dim` (expansion factor ~8×)
  - Latent representations with ReLU activation (for sparsity)
  - Decoder: Linear layer back to `hidden_dim`
  - **Tied weights:** Decoder weights = Encoder weights transposed
- [ ] Implement training loop
  - MSE reconstruction loss
  - L₁ regularization: `L1_loss = lambda * ||latents||_1`
  - Total loss: `L_total = MSE + L1_loss`
- [ ] Implement training utilities
  - Learning rate scheduling (e.g., exponential decay, cosine annealing)
  - Early stopping (monitor validation loss)
  - Metric tracking (reconstruction, sparsity, L1 norm)
  - Checkpoint saving/loading
- [ ] Hyperparameter tuning
  - Evaluate different expansion factors: \[4, 8, 16\]
  - Sweep L₁ coefficients: \[0.0001, 0.001, 0.01, 0.1\]
  - Track training curves for all configurations
  - Select best configuration based on validation metrics
- [ ] Documentation
  - Explain "activation space" vs. "feature space"
  - Justify choice of tied weights
  - Discuss sparsity role in preventing feature collapse
  - Report final hyperparameter choices with justification

#### README Documentation

- [ ] Write "Q3: Sparse Autoencoder Training" section
- [ ] Include hyperparameter tuning results with plots
- [ ] Document key concepts: activation space, feature space, tied weights, sparsity

### Deliverables

- `src/brain_surgery/sae.py` (SAE + training utilities)
- `results/metrics/sae_training_loss.png` (training curves)
- `results/metrics/hyperparameter_sweep.json` (tuning results)
- Trained SAE weights saved under `results/` (location to be finalized during Q3 implementation)
- README "Q3: Hyperparameter Tuning" section

### Success Criteria

- [ ] SAE trains without numerical instabilities
- [ ] Sparsity increases with higher λ values
- [ ] Validation loss plateaus (early stopping effective)
- [ ] Reconstruction quality reasonable (low MSE)
- [ ] Type-checked code with full docstrings
- [ ] All hyperparameter runs justified and documented

______________________________________________________________________

## Member 4: The Interpretability Analyst 📊 (Q4 & Q5 Bonus)

**Primary Focus:** Feature extraction, manual + LLM-based interpretation, visualization.

### Assigned Tasks

#### Q4: Feature Interpretation & Manual Analysis

- [ ] Implement `FeatureInterpreter` class
  - Compute feature activation statistics (mean, std, max activations)
  - Perform backwards mapping: latent dimension → activation → tokens
  - Identify top-k examples per feature (highest activations)
- [ ] Manual feature inspection
  - Identify 3–5 distinct, interpretable features
  - Example categories: "Python code," "Emotional tone," "Mathematical concepts," "Names," "URLs"
  - For each feature:
    - Collect 5–10 text snippets where it activates strongly
    - Describe the pattern in plain language
    - Rate interpretability (subjective, 1-5 scale)
- [ ] Create feature report
  - Save to `results/features/feature_report.md`
  - Include feature description, examples, activation distribution histogram
- [ ] Visualization
  - Activation distribution per feature (histograms or violin plots)
  - t-SNE or UMAP projection of feature activations (optional)

#### Q5: Bonus – LLM-based Feature Labeling

- [ ] Implement `FeatureLabelingPipeline`
  - Query a larger LLM (Claude, GPT-4, or open-source Llama)
  - Provide top examples for a feature and ask: "What does this feature represent?"
  - Collect multiple suggested labels per feature
  - Aggregate and score label suggestions
- [ ] Feature correlation analysis
  - Compute pairwise cosine distance between feature activation vectors
  - Build correlation matrix (features × features)
  - Cluster features using hierarchical clustering or k-means
- [ ] Visualization
  - Feature correlation heatmap
  - Dendrogram of feature clusters
  - Interactive plot of feature space (t-SNE/UMAP) colored by cluster

#### README Documentation

- [ ] Write "Q4: Feature Interpretation" section
  - Describe the 3–5 identified features with examples
  - Include snippets demonstrating each feature
  - Provide activation distribution plots
- [ ] Write "Q5 Bonus: Feature Clustering" section (if implemented)
  - Show correlation matrix and cluster assignments
  - Explain semantic groupings

### Deliverables

- `src/brain_surgery/interpret.py` (backwards mapping + analysis)
- (Optional) any Q5-specific code may live in `src/brain_surgery/interpret.py` or a new dedicated module if created later
- `results/features/feature_report.md` (manual analysis)
- `results/features/feature_*.txt` (example snippets per feature)
- `results/features/activation_distributions.png` (histograms)
- `results/features/feature_correlation_heatmap.png` (Q5 bonus)
- README "Q4" and "Q5" sections

### Success Criteria

- [ ] 3–5 features identified with clear semantic interpretations
- [ ] Examples provided for each feature (not cherry-picked edge cases)
- [ ] Feature report is well-written and informative
- [ ] (Bonus) LLM labeling works and produces coherent labels
- [ ] (Bonus) Feature clustering reveals meaningful groupings
- [ ] All code typed, documented, and tested

______________________________________________________________________

## 📋 Cross-Cutting Responsibilities

### All Members

- [ ] Follow **CONTRIBUTING.md** guidelines
  - Branch naming: `feat/`, `fix/`, `exp/`, `docs/`, `refactor/`
  - Commits: Conventional Commits format
  - Type hints: All functions fully typed
  - Docstrings: Google-style on all public functions/classes
- [ ] Run pre-commit hooks before pushing
  - `uv run ruff check --fix .`
  - `uv run ruff format .`
  - `uv run mypy .`
  - `uv run pytest tests/`
- [ ] Write tests for your code
  - Unit tests in `tests/` directory
  - Test coverage requirement: ≥80%
- [ ] Code review: Review PRs before merging
  - Ensure type safety
  - Check docstrings and examples
  - Verify experimental reproducibility (logging, seeds)

### Documentation (Distributed)

- **Member 1:** Installation, Project Structure, Q1 explanation
- **Member 2:** Q2 data stats, Q6 intervention methodology
- **Member 3:** Q3 hyperparameter tuning, concepts
- **Member 4:** Q4 feature analysis, Q5 clustering (bonus)
- **All:** Contribute to "Getting Started" and "Troubleshooting" sections

______________________________________________________________________

## 🗓️ Timeline & Milestones

| Week    | Milestone                                 | Owner(s)      | Status          |
| ------- | ----------------------------------------- | ------------- | --------------- |
| **1–2** | Repo setup, Q1 model wrapper              | Member 1      | ✅ (2026-03-28) |
| **2–3** | Q2 data collection pipeline               | Member 2      | ⏳              |
| **3–4** | Q3 SAE training + hyperparameter tuning   | Member 3      | ⏳              |
| **4–5** | Q4 feature interpretation                 | Member 4      | ⏳              |
| **5–6** | Q5 bonus (LLM labeling) & Q6 intervention | Members 4 & 2 | ⏳              |
| **6–7** | Write-up & documentation, final review    | All           | ⏳              |
| **7**   | Final submission                          | All           | ⏳              |

______________________________________________________________________

## 📊 Integration Points

```
Q1 (Model Wrapper)
    ↓
Q2 (Data Collection) ← uses Q1
    ↓
Q3 (SAE Training) ← uses Q2
    ↓
Q4 (Feature Analysis) ← uses Q3
    ↓
Q6 (Intervention) ← uses Q1 + Q3 + Q4
    ↓
Q5 (Bonus: LLM Labeling) ← uses Q4
```

**Key Handoffs:**

- **Q1 → Q2:** `ModelWrapper` class and `generate_with_activations()` method
- **Q2 → Q3:** Saved activations (HDF5 + token mappings)
- **Q3 → Q4:** Trained SAE checkpoint and feature representations
- **Q4 → Q6:** Identified feature indices for clamping experiments

______________________________________________________________________

## ✅ Final Checklist

Before submission, ensure:

- [ ] All code passes `ruff check`, `ruff format`, `mypy --strict`
- [ ] All tests pass: `pytest tests/ -v`
- [ ] All PRs reviewed and merged to `main`
- [ ] README.md complete with all 6 assignment questions documented
- [ ] All deliverables in `results/` directory
- [ ] Hyperparameter sweep results documented (Member 3)
- [ ] Feature analysis with 3–5 examples provided (Member 4)
- [ ] Counterfactual experiment results saved (Member 2)
- [ ] CONTRIBUTING.md and WORK_DISTRIBUTION.md up-to-date
- [ ] No API keys or credentials in repository

______________________________________________________________________

**Good luck, team! 🚀**
