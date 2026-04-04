# Q2 - Data Collection and Storage

Implementation file: `src/brain_surgery/data_gen.py`

## Dataset Definition (Soccer Activations)

The SAE dataset is derived from internal model activations, not raw text.

- Corpus size: 170 curated soccer prompts.
- Domain coverage: player biographies (Messi, Ronaldo), team tactics,
  league history, and match rules.
- Capture location: Layer 12 (0.5B Model) / 14 (7B Model) residual stream.
- Storage tensor: $137557 \\times d\_{model}$ in
  `data/activations/soccer_activations_dataset.pt`.

### Layer Rationale

Layer 12/14 is treated as a semantic sweet spot: syntax is already integrated,
while token-level commitment is not yet fully finalized. This improves the
quality of downstream feature interpretation and intervention studies.

## Theoretical Goal

Create a trustworthy measurement dataset linking each token event to its hidden
activation and semantic context. Mechanistic interpretability quality is bounded
by the quality and traceability of this mapping.

## Implementation

- Uses a curated NDJSON corpus (`data/corpus/soccer_prompts.ndjson`)
  with structured fields such as category, topic, and prompt text.
- Validates every NDJSON record and rejects malformed lines early.
- Runs prompts through the hooked model to collect per-token activation rows.
- Stores activation matrix and metadata together in `.pt` payloads under
  `data/activations/`.
- Preserves category-rich metadata needed for later purity auditing and
  category-level analysis in Q5.

## NDJSON-Driven Pipeline Value

- Explicit schema prevents silent data drift.
- Line-oriented records support easy dataset extension and auditing.
- Rich metadata makes semantic interpretation falsifiable instead of anecdotal.

## Current Status

Q2 is complete, and the dataset format now supports downstream clustering
validation and intervention analysis.

## Code Entry Points

- Module: `src/brain_surgery/data_gen.py`
- Dataset builder: `DataGenerator.generate_dataset(...)`
- Corpus source: `data/corpus/soccer_prompts.ndjson`
