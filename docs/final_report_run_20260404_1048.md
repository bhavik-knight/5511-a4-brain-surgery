# Formal Report: a4-brain-surgery (run_20260404_1048)

## Executive Summary

This report documents the final 7B-scale run of the a4-brain-surgery project
and presents a mechanistic interpretability narrative from architecture to
causal intervention. The core finding is that sparse latent features learned
with an SAE at 7B scale remain highly interpretable and causally actionable
for soccer-domain generation.

## Q1 and Q3: Architecture and Training

### Base Model (Qwen-2.5-7B)

- Model family: Qwen-2.5, decoder-only Transformer.
- Depth: 28 layers.
- Hidden size: $d_{\mathrm{model}} = 3584$.
- Residual stream extraction layer: Layer 14.

Layer 14 is selected as a semantic mid-layer where contextual composition is
already substantial, while late-token commitment is still incomplete. This
position yields representations that are rich enough for concept discovery yet
early enough to preserve broad latent structure for intervention.

### SAE Design

- Expansion factor: 32x.
- Latent dimensionality: $3584 \times 32 = 114688$ features.
- Encoder nonlinearity: ReLU.

Precision strategy on A100:

- Activations processed in bfloat16 for memory-efficient throughput.
- SAE weights maintained in float32 for optimization stability.

This mixed-precision strategy preserves numerical robustness while enabling
114k-feature analysis within hardware limits.

### Training Outcome

- Sparsity penalty: $L_1 = 0.01$.
- Early stopping: triggered at Epoch 26.
- Best checkpoint: Epoch 16.
- Dead-neuron fraction: 0%.

Interpretation: the run achieved an effective sparsity-reconstruction balance,
with strong specialization and no evidence of latent collapse.

______________________________________________________________________

## Q4: Interpretability Results ("Sniper" Features)

The final census identified sharp, semantically selective features that fire on
narrow soccer concepts and entities.

| Feature ID | Semantic Interpretation                            | Cluster |
| ---------: | -------------------------------------------------- | ------: |
|       6442 | Ronaldo identity                                   |       3 |
|       6265 | Tactical high-press strategy                       |       5 |
|       4635 | VAR/referee rule enforcement                       |       4 |
|       4973 | Goalkeeper match role                              |       3 |
|      12134 | Statistical tracking (assists/chances)             |     N/A |
|       5127 | Set-piece and penalty-area logic                   |     N/A |
|      13018 | Regional league pattern (La Liga/inverted wingers) |     N/A |

These features satisfy the practical "sniper" criterion: sparse activation with
high semantic specificity and low conceptual bleed across unrelated contexts.

______________________________________________________________________

## Q5: Semantic Clustering (Spherical K-Means, $k=6$)

Spherical K-Means with $k=6$ produced a coherent partition of latent semantics.
The most important separation patterns are:

- Cluster 2: Instructional noise and format language (for example, describe-style prompts).
- Cluster 3: Match action and entity-driven event dynamics.
- Cluster 4: Institutional knowledge (rules, officiating, regulatory semantics).
- Cluster 5: Tactical physicality (pressing intensity, structure, tempo pressure).

This separation demonstrates that the SAE disentangles both procedural prompt
scaffolding and genuine football-domain world knowledge in distinct latent
subspaces.

______________________________________________________________________

## Q6: Mechanistic Intervention and Causal Verification

Causal tests were performed by clamping selected latent features and measuring
token log-probability shifts in generated continuations.

- Clamping Feature 6442 (Ronaldo identity) steered entity-related outputs.
- Clamping Feature 8378 (goal feature) steered goal-related probability mass.

The intervention effect confirms directional control: these latents are not
passive correlates, but active control variables in the model's generation
process.

Tokenizer split artifact handling:

- Sub-token fragmentation (for example, "ald" as a Ronaldo fragment) is expected
  in BPE tokenization.
- The SAE still mapped these sub-token patterns to stable higher-level identity
  semantics, demonstrating robust concept binding across fragmented token forms.

______________________________________________________________________

## Scaling Note: 7B Engineering Constraints

At 7B scale and 114,688 latents, census computation requires careful memory
engineering. The key mechanism is chunked matrix multiplication in feature
census logic, which streams latent blocks to keep peak memory under the 80GB
VRAM budget of A100 hardware.

This chunking strategy is the critical enabler for end-to-end interpretability
at full latent width.

______________________________________________________________________

## Supporting Artifacts

- Elbow plot: [results/metrics/elbow_plot.png](results/metrics/elbow_plot.png)
- Global feature census snapshot: [results/features/run_20260404_0547/global_feature_census.csv](results/features/run_20260404_0547/global_feature_census.csv)

Note: run_20260404_1048 conclusions in this report follow the final structured
result summary. The local workspace currently exposes the latest census artifact
under run_20260404_0547.
