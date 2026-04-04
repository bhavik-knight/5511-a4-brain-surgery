"""Unit tests for DataGenerator corpus loading and token-activation alignment."""

from pathlib import Path

import pytest
import torch

from brain_surgery.data_gen import DataGenerator


class MockWrapper:
    """Mock ModelWrapper for deterministic DataGenerator tests."""

    def __init__(self) -> None:
        self.layer_idx = 12
        self._last_token_texts: list[str] | None = None
        self._last_token_strs: list[str] | None = None
        self._last_output_ids: torch.Tensor | None = None
        self._last_generated_text: str | None = None
        self.saved: list[tuple[int, str]] = []

    def generate_with_activations(
        self,
        prompt: str,
        **kwargs: object,
    ) -> tuple[str, dict[str, torch.Tensor]]:
        self._last_token_texts = ["tok0", "tok1", "tok2", "tok3"]
        self._last_token_strs = ["tok0", "tok1", "tok2", "tok3"]
        self._last_output_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        self._last_generated_text = f"{prompt} generated"

        # Use fewer activation rows than token_texts to force min() alignment.
        acts = torch.randn(1, 3, 896)
        return self._last_generated_text, {"layer": acts}

    def save_activations(
        self, *, batch_idx: int, file_stem: str, save_dir: Path
    ) -> Path:
        self.saved.append((batch_idx, file_stem))
        out = save_dir / f"{file_stem}.pt"
        torch.save({"ok": True}, out)
        return out


def test_load_corpus_returns_118_prompts() -> None:
    """Verify the prompt corpus size is exactly 118."""
    generator = DataGenerator(wrapper=MockWrapper())
    prompts = generator.load_corpus()
    assert len(prompts) == 118


def test_generate_dataset_aligns_token_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify metadata rows align with activation rows after truncation."""
    monkeypatch.setattr("brain_surgery.data_gen.ACTIVATIONS_DIR", tmp_path)

    generator = DataGenerator(wrapper=MockWrapper(), batch_size=2, max_new_tokens=5)
    activation_matrix, metadata, summary = generator.generate_dataset(prompt_limit=2)

    # Each prompt contributes min(4 token_texts, 3 activation rows) = 3 rows.
    assert activation_matrix.shape == (6, 896)
    assert isinstance(activation_matrix, torch.Tensor)
    assert len(metadata) == 6
    assert summary.total_tokens == 6

    # Check alignment fields exist and are sensible.
    first = metadata[0]
    assert first["token_text"].startswith("tok")
    assert first["hook_layer_index"] == 12
    assert first["prompt_id"] == 0


def test_generate_dataset_rejects_non_positive_prompt_limit() -> None:
    """Verify prompt_limit validation path raises ValueError."""
    generator = DataGenerator(wrapper=MockWrapper())

    with pytest.raises(ValueError):
        generator.generate_dataset(prompt_limit=0)


def test_generate_dataset_missing_layer_activations_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify generation fails fast when layer activations are missing."""

    class MissingLayerWrapper(MockWrapper):
        def generate_with_activations(
            self,
            prompt: str,
            **kwargs: object,
        ) -> tuple[str, dict[str, torch.Tensor]]:
            self._last_token_texts = ["tok0"]
            self._last_token_strs = ["tok0"]
            self._last_output_ids = torch.tensor([[1]], dtype=torch.long)
            self._last_generated_text = prompt
            return prompt, {}

    monkeypatch.setattr(
        "brain_surgery.data_gen.ACTIVATIONS_DIR",
        Path("/tmp"),
    )

    generator = DataGenerator(wrapper=MissingLayerWrapper())
    with pytest.raises(RuntimeError):
        generator.generate_dataset(prompt_limit=1)


def test_data_generator_rejects_non_positive_batch_size() -> None:
    """Verify constructor enforces positive batch size."""
    with pytest.raises(ValueError):
        DataGenerator(wrapper=MockWrapper(), batch_size=0)


def test_generate_dataset_enforces_118_prompts_when_unbounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify strict corpus-size invariant in full (non-limited) generation path."""

    class ShortCorpusGenerator(DataGenerator):
        def load_corpus(self) -> list[str]:
            return ["only", "two"]

    monkeypatch.setattr(
        "brain_surgery.data_gen.ACTIVATIONS_DIR",
        Path("/tmp"),
    )
    generator = ShortCorpusGenerator(wrapper=MockWrapper())
    with pytest.raises(ValueError):
        generator.generate_dataset(prompt_limit=None)
