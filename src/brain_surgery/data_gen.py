"""Data generation for mechanistic interpretability of brain surgery models.

This module implements Q2 activation collection by running a fixed prompt
corpus through a ModelWrapper and saving token-aligned activations.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

import torch
from torch import Tensor

from .utils import ACTIVATIONS_DIR, CORPUS_DIR, ensure_dir_exists

type MetadataValue = int | str | list[str] | None
type MetadataRow = dict[str, MetadataValue]


class PromptRecord(TypedDict):
    """Structured input prompt record loaded from NDJSON."""

    category: str
    subcategory: str
    topic: str
    tags: list[str]
    era: str | None
    region: str | None
    prompt: str


class ActivationWrapper(Protocol):
    """Protocol for wrappers that can generate text and save activations."""

    @property
    def layer_idx(self) -> int | None:
        """Transformer layer index used for hook capture."""
        ...

    @property
    def _last_token_texts(self) -> list[str] | None:
        """Most recent decoded token text list from generation."""
        ...

    @property
    def _last_token_strs(self) -> list[str] | None:
        """Most recent token strings aligned with token ids."""
        ...

    @property
    def _last_output_ids(self) -> Tensor | None:
        """Most recent generated token ids tensor."""
        ...

    @property
    def _last_generated_text(self) -> str | None:
        """Most recent generated text string."""
        ...

    def generate_with_activations(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> tuple[str, dict[str, Tensor]]:
        """Generate text and return activation payloads."""
        ...

    def save_activations(
        self,
        *,
        batch_idx: int,
        file_stem: str,
        save_dir: Path,
    ) -> Path:
        """Save activation artifact for a prompt batch item."""
        ...


@dataclass(frozen=True)
class DatasetSummary:
    """Summary statistics for a generated activation dataset.

    Args:
            num_prompts: Total prompts processed.
            total_tokens: Total tokens captured across all prompts.
            average_seq_len: Average tokens per prompt.
            activation_shape: Final concatenated activation matrix shape.
    """

    num_prompts: int
    total_tokens: int
    average_seq_len: float
    activation_shape: tuple[int, int]


class DataGenerator:
    """Generate activation datasets from a prompt corpus.

    This class uses a ModelWrapper to capture residual-stream activations for
    each prompt, saves per-prompt artifacts via save_activations(), and builds
    a consolidated dataset tensor with token-level metadata.
    """

    def __init__(
        self,
        wrapper: ActivationWrapper,
        *,
        batch_size: int = 4,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> None:
        """Initialize the data generator.

        Args:
                wrapper: ModelWrapper instance with hooks registered.
                batch_size: Prompts processed per batch. Kept small for VRAM safety.
                max_new_tokens: Max new tokens to generate per prompt.
                temperature: Sampling temperature for generation.
                top_p: Nucleus sampling threshold for generation.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.wrapper = wrapper
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def load_corpus(self) -> list[PromptRecord]:
        """Load the structured soccer-domain prompt corpus from NDJSON.

        Returns:
                A list of prompt records with schema metadata.

        Raises:
                FileNotFoundError: If the corpus file is missing.
                ValueError: If the corpus file exists but contains no prompts.
        """
        corpus_path = CORPUS_DIR / "curated_soccer_prompts_1100.ndjson"
        if not corpus_path.exists():
            raise FileNotFoundError(
                "Missing corpus file at "
                f"{corpus_path}. Create it as NDJSON with fields "
                "prompt, category, subcategory, topic, tags, era, and region."
            )

        records: list[PromptRecord] = []
        for line_no, raw_line in enumerate(
            corpus_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid NDJSON object at {corpus_path}:{line_no}")

            prompt_obj = data.get("prompt")
            if not isinstance(prompt_obj, str) or not prompt_obj.strip():
                raise ValueError(f"Missing/invalid prompt at {corpus_path}:{line_no}")

            tags_obj = data.get("tags")
            tags: list[str]
            if isinstance(tags_obj, list):
                tags = [tag for tag in tags_obj if isinstance(tag, str)]
            else:
                tags = []

            category_obj = data.get("category")
            subcategory_obj = data.get("subcategory")
            topic_obj = data.get("topic")
            era_obj = data.get("era")
            region_obj = data.get("region")

            records.append(
                {
                    "category": category_obj
                    if isinstance(category_obj, str)
                    else "General",
                    "subcategory": (
                        subcategory_obj
                        if isinstance(subcategory_obj, str)
                        else "Legacy"
                    ),
                    "topic": topic_obj if isinstance(topic_obj, str) else "Mixed",
                    "tags": tags,
                    "era": era_obj if isinstance(era_obj, str) else None,
                    "region": region_obj if isinstance(region_obj, str) else None,
                    "prompt": prompt_obj.strip(),
                }
            )

        if not records:
            raise ValueError(
                f"Corpus file exists but is empty: {corpus_path}. "
                "Populate it with NDJSON prompt records."
            )

        return records

    def generate_dataset(
        self,
        *,
        prompt_limit: int | None = None,
    ) -> tuple[Tensor, list[MetadataRow], DatasetSummary]:
        """Generate and save the activation dataset.

        Args:
                prompt_limit: Optional limit on number of prompts for smoke tests.

        Returns:
                A tuple with activation matrix, metadata list, and summary statistics.
        """
        prompt_records = self.load_corpus()
        if prompt_limit is not None:
            if prompt_limit <= 0:
                raise ValueError(
                    f"prompt_limit must be positive when provided, got {prompt_limit}"
                )
            prompt_records = prompt_records[:prompt_limit]

        ensure_dir_exists(ACTIVATIONS_DIR)

        activation_chunks: list[Tensor] = []
        metadata: list[MetadataRow] = []
        total_tokens = 0
        seq_lens: list[int] = []

        for batch_start in range(0, len(prompt_records), self.batch_size):
            batch = prompt_records[batch_start : batch_start + self.batch_size]
            for idx, prompt_record in enumerate(batch):
                prompt_id = batch_start + idx
                prompt = prompt_record["prompt"]
                _, activations = self.wrapper.generate_with_activations(
                    prompt=prompt,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

                layer_acts = activations.get("layer")
                if layer_acts is None:
                    raise RuntimeError("Missing layer activations after generation")

                if layer_acts.dim() == 3:
                    if layer_acts.shape[0] != 1:
                        raise ValueError(
                            "Expected batch=1 activations, got "
                            f"{tuple(layer_acts.shape)}"
                        )
                    acts_2d = layer_acts[0]
                else:
                    acts_2d = layer_acts

                acts_2d = acts_2d.detach().cpu()
                activation_chunks.append(acts_2d)

                # Use internal token metadata from the wrapper for alignment.
                token_texts = self.wrapper._last_token_texts or []  # noqa: SLF001
                token_strs = self.wrapper._last_token_strs or []  # noqa: SLF001
                token_ids = []
                if self.wrapper._last_output_ids is not None:  # noqa: SLF001
                    token_ids = (
                        self.wrapper._last_output_ids[0].tolist()  # noqa: SLF001
                    )
                generated_text = self.wrapper._last_generated_text or ""  # noqa: SLF001

                seq_len = min(len(token_texts), acts_2d.shape[0])
                seq_lens.append(seq_len)
                total_tokens += seq_len
                hook_layer_index = (
                    self.wrapper.layer_idx if self.wrapper.layer_idx is not None else -1
                )
                hook_layer_name = f"model.model.layers[{hook_layer_index}]"

                for tok_idx in range(seq_len):
                    metadata.append(
                        {
                            "prompt_id": prompt_id,
                            "prompt_text": prompt,
                            "token_index": tok_idx,
                            "token_id": token_ids[tok_idx]
                            if tok_idx < len(token_ids)
                            else -1,
                            "token_text": token_texts[tok_idx],
                            "token_str": token_strs[tok_idx]
                            if tok_idx < len(token_strs)
                            else "",
                            "generated_text": generated_text,
                            "hook_layer_index": hook_layer_index,
                            "hook_layer_name": hook_layer_name,
                            "category": prompt_record["category"],
                            "subcategory": prompt_record["subcategory"],
                            "topic": prompt_record["topic"],
                            "tags": prompt_record["tags"],
                            "era": prompt_record["era"],
                            "region": prompt_record["region"],
                        }
                    )

                # Save per-prompt artifact using ModelWrapper helper.
                self.wrapper.save_activations(
                    save_dir=ACTIVATIONS_DIR,
                    batch_idx=prompt_id,
                    file_stem=f"soccer_prompt_{prompt_id:03d}",
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        activation_matrix = torch.cat(activation_chunks, dim=0)
        dataset_path = ACTIVATIONS_DIR / "soccer_activations_dataset.pt"
        torch.save(
            {
                "activation_matrix": activation_matrix,
                "metadata": metadata,
                "num_prompts": len(prompt_records),
                "total_tokens": total_tokens,
                "hidden_dim": activation_matrix.shape[1],
            },
            dataset_path,
        )

        average_seq_len = total_tokens / len(prompt_records)
        summary = DatasetSummary(
            num_prompts=len(prompt_records),
            total_tokens=total_tokens,
            average_seq_len=average_seq_len,
            activation_shape=(
                int(activation_matrix.shape[0]),
                int(activation_matrix.shape[1]),
            ),
        )

        return activation_matrix, metadata, summary


if __name__ == "__main__":
    from .model_wrapper import ModelWrapper
    from .utils import DEFAULT_MODEL_NAME

    print(f"Loading model: {DEFAULT_MODEL_NAME}...")
    # Initialize wrapper (defaults to Layer 14 and BFloat16 on A100)
    model_wrapper = ModelWrapper(model_name=DEFAULT_MODEL_NAME)

    print("Capturing activations from prompt corpus...")
    data_generator = DataGenerator(model_wrapper)

    # Generate and save the consolidated dataset
    _, _, dataset_summary = data_generator.generate_dataset()

    expected_path = ACTIVATIONS_DIR / "soccer_activations_dataset.pt"
    print(f"Step 1 Complete: Dataset saved to {expected_path}")
    print(
        f"Captured {dataset_summary.total_tokens} tokens across "
        f"{dataset_summary.num_prompts} prompts."
    )
