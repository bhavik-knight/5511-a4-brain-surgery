# THEORY - Foundations and Methodological Rationale

## 1. SAE Bottleneck Logic and Feature Disentanglement

Following the sparse autoencoder (SAE) framing used in prior interpretability
work, we treat transformer hidden states as vectors $x \\in \\mathbb{R}^{d}$ and
learn a latent representation $z \\in \\mathbb{R}^{m}$ with an expansion
($m > d$) plus sparsity pressure.

Linear encoder/decoder core:

$$
z = \\operatorname{ReLU}(W_e x + b_e), \\quad \\hat{x} = W_d z + b_d
$$

where:

- $W_e \\in \\mathbb{R}^{m \\times d}$ is the encoder.
- $W_d \\in \\mathbb{R}^{d \\times m}$ is the decoder.
- $\\hat{x}$ reconstructs the original hidden activation.

Training objective:

$$
\\mathcal{L} = \\frac{1}{n}\\sum\_{i=1}^{n}\\lVert x_i - \\hat{x}\_i \\rVert_2^2

- \\lambda \\cdot \\frac{1}{n}\\sum\_{i=1}^{n}\\lVert z_i \\rVert_1
  $$

Interpretability role of the $L_1$ penalty:

- The reconstruction term preserves signal fidelity.
- The $L_1$ term drives most latent coordinates toward zero.
- Sparse activations reduce feature superposition and improve disentanglement.

In practical terms, the SAE acts as an interpretability bottleneck: only a
small subset of latent units activate per token event, making semantic analysis
and intervention more tractable.

## 2. Activation Capture via Hooks

The project captures internal activations by registering forward hooks on a
middle transformer layer. This operationalizes the measurement pipeline:

$$
\\text{prompt} \\rightarrow h\_\\ell \\rightarrow \\text{token-aligned activation rows}
$$

where $h\_\\ell$ denotes hidden states at layer $\\ell$.

Why this matters:

- Hooked activations are the dataset used to train the SAE.
- Token-to-row alignment enables feature-to-text interpretation.
- The same representation can be edited during intervention experiments.

## 3. Why We Pivoted to Spherical K-Means

Earlier clustering attempts with standard Euclidean workflows (including
DBSCAN tuning in this codebase) showed instability in high-dimensional feature
space. The core issue is the curse of dimensionality:

- Distances concentrate in high dimensions.
- Raw vector norms dominate Euclidean geometry.
- Density thresholds (DBSCAN) become brittle across runs.

To address this, we use a spherical variant of K-Means by L2-normalizing each
feature vector before clustering:

$$
\\tilde{f}\_i = \\frac{f_i}{\\lVert f_i \\rVert_2}
$$

Then K-Means groups by directional similarity (cosine-like behavior) rather
than raw magnitude. This is more appropriate for semantic feature neighborhoods
in SAE latent analysis.

## 4. Elbow Method and Dynamic Selection

For candidate values of $k$, we compute inertia/SSE:

$$
\\operatorname{SSE}(k) = \\sum\_{c=1}^{k}\\sum\_{f_i \\in C_c}\\lVert f_i - \\mu_c \\rVert_2^2
$$

We then select $k$ using a dynamic slowdown heuristic on the SSE improvement
rate, and visualize the decision with an elbow plot (`results/metrics/elbow_plot.png`).
This supports methodological transparency in Q5.

## 5. Mechanistic Interpretability and Causal Testing

The "brain surgery" framing combines interpretation with intervention:

1. Learn sparse latent features from internal activations.
1. Map features to high-activation token/context evidence.
1. Clamp a selected feature during generation.
1. Compare baseline vs intervened token probabilities.

A consistent targeted change under intervention provides causal evidence that
the feature participates in the model behavior being tested.

## 6. Positioning of This Project

This repository can be viewed as a third-generation refinement:

- Foundational SAE interpretability implementation roots.
- Refined theory and reporting integration from prior project evolution.
- Additional university-specific metadata auditing and spherical clustering
  validation for assignment-grade reproducibility.

## References

- Anthropic. Scaling Monosemanticity.
  https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- Nikola Kriznar. sparse-autoencoder-llm-interpretability.
  https://github.com/nkriznar/sparse-autoencoder-llm-interpretability
- Miguel Angel Palafox Gomez (ter-kes). sparse-autoencoder-llm-interpretability.
  https://github.com/ter-kes/sparse-autoencoder-llm-interpretability
