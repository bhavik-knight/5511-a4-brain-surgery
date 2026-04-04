# THEORY - Mechanistic Interpretability Framework

## 1. SAE Architecture and Objective

We model hidden states from a transformer layer as vectors $x \\in \\mathbb{R}^{d}$ and
train a Sparse Autoencoder (SAE) to map them into a larger latent space
$z \\in \\mathbb{R}^{m}$ where $m > d$.

Encoder and decoder equations:

$$
z = \\operatorname{ReLU}(W_e x + b_e), \\quad \\hat{x} = W_d z + b_d
$$

where:

- $W_e \\in \\mathbb{R}^{m \\times d}$ is the encoder weight matrix.
- $W_d \\in \\mathbb{R}^{d \\times m}$ is the decoder weight matrix.
- $\\hat{x}$ is the reconstruction of the original activation vector.

The training objective balances fidelity and interpretability:

$$
\\mathcal{L} = \\frac{1}{n}\\sum\_{i=1}^{n} |x_i - \\hat{x}\_i|\_2^2

- \\lambda \\cdot \\frac{1}{n}\\sum\_{i=1}^{n} |z_i|\_1
  $$

* The first term preserves information (reconstruction quality).
* The $L_1$ term pushes most latent coordinates toward zero (sparsity).
* $\\lambda$ controls the tradeoff between reconstruction and sparsity.

Interpretability rationale:

- Dense activations are often polysemantic (multiple concepts mixed together).
- Sparse latent features are easier to map to token-level semantics.
- Expansion ($m>d$) plus sparsity enables more separable semantic directions.

## 2. Activation Capturing via Forward Hooks

To analyze internal computation, we intercept activations from an internal
transformer block using a forward hook. In PyTorch terms, this is
`register_forward_hook` on the selected residual-stream layer.

Conceptually:

1. A prompt is tokenized and sent through the model.
1. At the target layer $\\ell$, we read hidden states $h\_\\ell$.
1. We align each token position with its activation row.
1. We persist activation rows together with metadata for downstream analysis.

Why middle-layer hooks:

- Early layers bias toward lexical/syntactic signals.
- Late layers are strongly task-logit coupled.
- Middle layers typically provide a balanced semantic representation.

## 3. Clustering Theory for High-Dimensional Features

After SAE training, each latent feature is represented by a profile vector over
token rows. Feature discovery requires grouping semantically related profiles.

### Why Spherical K-Means

In high-dimensional spaces, Euclidean distance magnitude can dominate geometry.
We mitigate this by L2-normalizing feature vectors before clustering:

$$
\\tilde{f}\_i = \\frac{f_i}{|f_i|\_2}
$$

Then standard K-Means on normalized vectors approximates cosine-similarity
grouping (spherical behavior), which is better suited for semantic orientation
rather than raw norm magnitude.

### Why Elbow Selection

For candidate cluster counts $k$, we compute SSE/inertia:

$$
\\operatorname{SSE}(k) = \\sum\_{c=1}^{k}\\sum\_{f_i \\in C_c}|f_i - \\mu_c|\_2^2
$$

The elbow criterion selects the $k$ where marginal SSE improvement starts to
flatten. In this project, we use a dynamic rate-of-change heuristic that
identifies where improvement slowdown becomes significant.

## 4. Mechanistic Interpretability and "Brain Surgery"

Mechanistic interpretability asks not only what the model outputs, but which
internal circuits cause those outputs.

Our "brain surgery" workflow:

1. Learn sparse latent features from internal activations.
1. Associate features with semantic evidence (top tokens/contexts).
1. Intervene causally by clamping a target feature at inference time.
1. Compare output probabilities against controls.

If changing one feature consistently changes targeted behavior while controls do
not, that is evidence of causal relevance rather than correlation.

This bridges descriptive interpretation (feature labeling) and interventional
validation (counterfactual effect measurement).

## 5. Practical Limits

- A single-layer SAE gives a partial model of the full network computation.
- Sparse features can still retain residual polysemanticity.
- Interpretations are bounded by corpus coverage and metadata quality.
- Causal claims are local to the intervention setting and prompt distribution.

These limits motivate rigorous metadata audits, control interventions, and
run-scoped experiment reproducibility.
