# Q2 - Data Collection and Storage

Implementation file: `src/brain_surgery/data_gen.py`

## Theoretical Goal

Create a trustworthy measurement dataset linking each token event to its hidden
activation and semantic context. Mechanistic interpretability quality is bounded
by the quality and traceability of this mapping.

## Implementation

- Uses a curated NDJSON corpus (`data/corpus/curated_soccer_prompts_1100.ndjson`)
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
