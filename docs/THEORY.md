# THEORY - Foundations and Methodological Rationale

## 1. SAE Bottleneck Logic and Feature Disentanglement

Following the sparse autoencoder framing, we treat transformer hidden states as
vectors $x \in \mathbb{R}^{d}$ and learn latent features
$z \in \mathbb{R}^{m}$ with expansion ($m > d$) and sparsity pressure.

Linear encoder/decoder core:

$$
z = \operatorname{ReLU}(W_e x + b_e), \quad \hat{x} = W_d z + b_d
$$

where:

- $W_e \in \mathbb{R}^{m \times d}$ is the encoder matrix.
- $W_d \in \mathbb{R}^{d \times m}$ is the decoder matrix.
- $\hat{x}$ is the reconstruction of the input activation.

Training objective:

$$
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\lVert x_i - \hat{x}_i \rVert_2^2 + \lambda \cdot \frac{1}{n}\sum_{i=1}^{n}\lVert z_i \rVert_1
$$

Why the $L_1$ penalty matters:

- The reconstruction term preserves information.
- The $L_1$ term drives most latent units toward zero.
- Sparse activations reduce feature superposition and improve disentanglement.

## 2. Activation Capture via Hooks

We capture internal activations by registering forward hooks on a middle
transformer layer. The measurement pipeline is:

$$
\mathrm{prompt} \rightarrow h_{\ell} \rightarrow \mathrm{token\ activation\ rows}
$$

where $h_{\ell}$ denotes hidden states at layer $\ell$.

## 3. Why We Pivoted to Spherical K-Means

Standard Euclidean clustering and DBSCAN-style density thresholds were unstable
in high-dimensional feature spaces, a classic curse-of-dimensionality issue:

- distance concentration,
- norm-dominated geometry,
- brittle density hyperparameters.

So we switched to Spherical K-Means by normalizing features before clustering:

$$
\widetilde{f}_i = \frac{f_i}{\lVert f_i \rVert_2}
$$

This emphasizes directional similarity (cosine-like structure), which is better
aligned with semantic feature grouping.

## 4. Elbow Method and Dynamic Selection

For candidate values of $k$, we compute inertia/SSE:

$$
\operatorname{SSE}(k) = \sum_{c=1}^{k}\sum_{f_i \in C_c}\lVert f_i - \mu_c \rVert_2^2
$$

We select $k$ by a dynamic slowdown heuristic on SSE improvement and export an
elbow plot for visual justification.

## 5. Mechanistic Interpretability and Causal Testing

The "brain surgery" workflow combines interpretation and intervention:

1. Learn sparse latent features from hidden activations.
1. Map latent features to top-activating token contexts.
1. Clamp selected features during generation.
1. Compare baseline vs target/control intervention outcomes.

If targeted clamps produce specific behavioral changes that controls do not,
this supports a causal interpretation.

## References

- Anthropic. Scaling Monosemanticity.
  https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- Nikola Kriznar. sparse-autoencoder-llm-interpretability.
  https://github.com/nkriznar/sparse-autoencoder-llm-interpretability
- Miguel Angel Palafox Gomez (ter-kes). sparse-autoencoder-llm-interpretability.
  https://github.com/ter-kes/sparse-autoencoder-llm-interpretability
