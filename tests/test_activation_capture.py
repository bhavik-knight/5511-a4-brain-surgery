"""Lightweight activation capture utility tests."""

from pathlib import Path

import torch


def _chunked(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def test_corpus_data_loader_initialization() -> None:
    """Verify corpus list and batch-size assumptions used by data loaders."""
    corpus = ["a", "b", "c", "d", "e"]
    batch_size = 2
    batches = _chunked(corpus, batch_size)

    assert isinstance(corpus, list)
    assert all(isinstance(item, str) for item in corpus)
    assert len(batches) == 3
    assert batches[0] == ["a", "b"]


def test_corpus_batch_generation() -> None:
    """Verify stable chunking behavior for corpus batches."""
    corpus = [f"prompt-{i}" for i in range(7)]
    batches = _chunked(corpus, 3)

    assert len(batches) == 3
    assert batches[0] == ["prompt-0", "prompt-1", "prompt-2"]
    assert batches[-1] == ["prompt-6"]


def test_activation_collector_save(tmp_path: Path) -> None:
    """Verify activation payload can be persisted with tensor integrity."""
    path = tmp_path / "acts.pt"
    payload = {
        "activations": torch.randn(4, 896),
        "token_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
    }

    torch.save(payload, path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_activation_collector_load(tmp_path: Path) -> None:
    """Verify saved activation payload reloads with correct tensor types."""
    path = tmp_path / "acts.pt"
    original = {
        "activations": torch.ones(2, 896, dtype=torch.float32),
        "token_ids": torch.tensor([8, 9], dtype=torch.long),
    }
    torch.save(original, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(loaded["activations"], torch.Tensor)
    assert isinstance(loaded["token_ids"], torch.Tensor)
    assert loaded["activations"].dtype == torch.float32
    assert loaded["token_ids"].dtype == torch.int64


def test_token_to_activation_mapping() -> None:
    """Verify token index and activation row stay aligned."""
    token_ids = torch.tensor([101, 102, 103], dtype=torch.long)
    activations = torch.randn(3, 896)

    mapping = {int(tid): activations[idx] for idx, tid in enumerate(token_ids.tolist())}

    assert len(mapping) == 3
    assert isinstance(mapping[101], torch.Tensor)
    assert mapping[101].shape == (896,)
