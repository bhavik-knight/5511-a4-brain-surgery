"""Unit tests for SAEInterpreter feature ranking and empty-dataset behavior."""

from pathlib import Path

import pytest
import torch

from brain_surgery.interpret import SAEInterpreter
from brain_surgery.sae import SparseAutoencoder


def _write_checkpoint(path: Path, input_dim: int = 896, latent_dim: int = 64) -> None:
    model = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    torch.save(model.state_dict_for_checkpoint(), path)


def _write_dataset(
    path: Path,
    activations: torch.Tensor,
    metadata: list[dict[str, int | float | str | None]],
) -> None:
    torch.save(
        {
            "activation_matrix": activations,
            "metadata": metadata,
            "num_prompts": 1,
            "total_tokens": activations.shape[0],
            "hidden_dim": activations.shape[1] if activations.ndim == 2 else 0,
        },
        path,
    )


def test_interpreter_load_and_compute_latents(tmp_path: Path) -> None:
    """Verify interpreter loads artifacts and emits tensor latents."""
    checkpoint = tmp_path / "sae_checkpoint.pt"
    dataset = tmp_path / "dataset.pt"

    _write_checkpoint(checkpoint, input_dim=896, latent_dim=64)
    acts = torch.randn(5, 896)
    metadata: list[dict[str, int | float | str | None]] = [
        {
            "prompt_id": 0,
            "prompt_text": "p",
            "token_index": i,
            "token_id": i,
            "token_text": f"tok{i}",
            "token_str": f"tok{i}",
            "generated_text": "g",
            "hook_layer_index": 12,
            "hook_layer_name": "model.model.layers[12]",
        }
        for i in range(5)
    ]
    _write_dataset(dataset, acts, metadata)

    interpreter = SAEInterpreter(checkpoint_path=checkpoint, dataset_path=dataset)
    interpreter.load()
    latents = interpreter.compute_latents()

    assert isinstance(latents, torch.Tensor)
    assert latents.shape == (5, 64)


def test_feature_ranking_and_examples_are_structured(tmp_path: Path) -> None:
    """Verify ranking and top-example outputs have expected keys and scalar types."""
    checkpoint = tmp_path / "sae_checkpoint.pt"
    dataset = tmp_path / "dataset.pt"

    _write_checkpoint(checkpoint, input_dim=896, latent_dim=32)
    acts = torch.randn(6, 896)
    metadata: list[dict[str, int | float | str | None]] = [
        {
            "prompt_id": 0,
            "prompt_text": "who won",
            "token_index": i,
            "token_id": i,
            "token_text": f"tok{i}",
            "token_str": f"tok{i}",
            "generated_text": "out",
            "hook_layer_index": 12,
            "hook_layer_name": "model.model.layers[12]",
        }
        for i in range(6)
    ]
    _write_dataset(dataset, acts, metadata)

    interpreter = SAEInterpreter(checkpoint_path=checkpoint, dataset_path=dataset)
    interpreter.load()
    interpreter.compute_latents()

    ranked = interpreter.rank_features_by_max_activation(top_k=3)
    assert len(ranked) == 3
    assert isinstance(ranked[0]["feature_index"], int)
    assert isinstance(ranked[0]["max_feature_value"], float)

    feature_index = int(ranked[0]["feature_index"])
    examples = interpreter.get_top_examples_for_feature(feature_index, top_k=3)
    assert len(examples) == 3
    assert isinstance(examples[0]["token_text"], str)


def test_interpreter_empty_dataset_raises_on_ranking(tmp_path: Path) -> None:
    """Verify empty activation matrices fail predictably during feature ranking."""
    checkpoint = tmp_path / "sae_checkpoint.pt"
    dataset = tmp_path / "dataset.pt"

    _write_checkpoint(checkpoint, input_dim=896, latent_dim=16)
    empty_acts = torch.empty(0, 896)
    _write_dataset(dataset, empty_acts, metadata=[])

    interpreter = SAEInterpreter(checkpoint_path=checkpoint, dataset_path=dataset)
    interpreter.load()
    latents = interpreter.compute_latents()

    assert isinstance(latents, torch.Tensor)
    assert latents.shape == (0, 16)

    with pytest.raises(IndexError):
        interpreter.rank_features_by_max_activation(top_k=3)


def test_compute_latents_requires_load(tmp_path: Path) -> None:
    """Verify compute_latents guards against pre-load usage."""
    interpreter = SAEInterpreter(
        checkpoint_path=tmp_path / "missing.ckpt",
        dataset_path=tmp_path / "missing.pt",
    )
    with pytest.raises(RuntimeError):
        interpreter.compute_latents()


def test_invalid_feature_and_row_indices_raise(tmp_path: Path) -> None:
    """Verify index bounds checks for feature and row lookup APIs."""
    checkpoint = tmp_path / "sae_checkpoint.pt"
    dataset = tmp_path / "dataset.pt"
    _write_checkpoint(checkpoint, input_dim=896, latent_dim=8)
    acts = torch.randn(3, 896)
    metadata: list[dict[str, int | float | str | None]] = [
        {
            "prompt_id": 0,
            "prompt_text": "p",
            "token_index": i,
            "token_id": i,
            "token_text": f"tok{i}",
            "token_str": f"tok{i}",
            "generated_text": "out",
            "hook_layer_index": 12,
            "hook_layer_name": "model.model.layers[12]",
        }
        for i in range(3)
    ]
    _write_dataset(dataset, acts, metadata)

    interpreter = SAEInterpreter(checkpoint_path=checkpoint, dataset_path=dataset)
    interpreter.load()
    interpreter.compute_latents()

    with pytest.raises(ValueError):
        interpreter.get_top_examples_for_feature(feature_index=99, top_k=2)
    with pytest.raises(ValueError):
        interpreter.get_top_features_for_row(row_index=99, top_k=2)


def test_get_top_features_for_row_happy_path(tmp_path: Path) -> None:
    """Verify token-row feature ranking returns typed values."""
    checkpoint = tmp_path / "sae_checkpoint.pt"
    dataset = tmp_path / "dataset.pt"
    _write_checkpoint(checkpoint, input_dim=896, latent_dim=10)
    acts = torch.randn(4, 896)
    metadata: list[dict[str, int | float | str | None]] = [
        {
            "prompt_id": 0,
            "prompt_text": "p",
            "token_index": i,
            "token_id": i,
            "token_text": f"tok{i}",
            "token_str": f"tok{i}",
            "generated_text": "g",
            "hook_layer_index": 12,
            "hook_layer_name": "model.model.layers[12]",
        }
        for i in range(4)
    ]
    _write_dataset(dataset, acts, metadata)

    interpreter = SAEInterpreter(checkpoint_path=checkpoint, dataset_path=dataset)
    interpreter.load()
    interpreter.compute_latents()

    rows = interpreter.get_top_features_for_row(row_index=0, top_k=3)
    assert len(rows) == 3
    assert isinstance(rows[0]["feature_index"], int)
    assert isinstance(rows[0]["feature_value"], float)


def test_interpreter_load_validation_errors(tmp_path: Path) -> None:
    """Verify load() validates presence/shape consistency of dataset metadata."""
    checkpoint = tmp_path / "sae_checkpoint.pt"
    _write_checkpoint(checkpoint, input_dim=896, latent_dim=8)

    missing_meta = tmp_path / "missing_meta.pt"
    torch.save(
        {"activation_matrix": torch.randn(2, 896), "metadata": None}, missing_meta
    )
    interpreter = SAEInterpreter(checkpoint_path=checkpoint, dataset_path=missing_meta)
    with pytest.raises(ValueError):
        interpreter.load()

    mismatch = tmp_path / "mismatch.pt"
    torch.save(
        {
            "activation_matrix": torch.randn(3, 896),
            "metadata": [{"token_text": "a"}],
        },
        mismatch,
    )
    interpreter2 = SAEInterpreter(checkpoint_path=checkpoint, dataset_path=mismatch)
    with pytest.raises(ValueError):
        interpreter2.load()


def test_interpreter_methods_require_latents_and_metadata(tmp_path: Path) -> None:
    """Verify guard rails before compute_latents() for lookup/ranking methods."""
    interpreter = SAEInterpreter(
        checkpoint_path=tmp_path / "x.pt",
        dataset_path=tmp_path / "y.pt",
    )
    with pytest.raises(RuntimeError):
        interpreter.get_top_examples_for_feature(0)
    with pytest.raises(RuntimeError):
        interpreter.get_top_features_for_row(0)
    with pytest.raises(RuntimeError):
        interpreter.rank_features_by_max_activation(1)
