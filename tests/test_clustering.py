"""Unit tests for clustering helpers and empty-latent edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brain_surgery.clustering import (
    cluster_features_kmeans,
    print_cluster_analysis,
)
from brain_surgery.interpret import SAEInterpreter


def _make_interpreter() -> SAEInterpreter:
    return SAEInterpreter(
        checkpoint_path=Path("dummy.ckpt"),
        dataset_path=Path("dummy.pt"),
    )


def test_cluster_features_requires_computed_latents() -> None:
    """Verify a clear error is raised when latents were never computed."""
    interpreter = _make_interpreter()
    interpreter.latents = None

    with pytest.raises(RuntimeError):
        cluster_features_kmeans(interpreter=interpreter)


def test_cluster_features_empty_latents_raises_value_error() -> None:
    """Verify empty latent matrices fail in a controlled manner."""
    interpreter = _make_interpreter()
    interpreter.latents = torch.empty(0, 8)

    with pytest.raises(ValueError):
        cluster_features_kmeans(interpreter=interpreter)


def test_cluster_features_and_print_summary_happy_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify clustering result structure and printable summary output."""
    latents = torch.randn(12, 20)
    interpreter = _make_interpreter()
    interpreter.latents = latents

    def _examples(
        feature_index: int,
        top_k: int = 5,
    ) -> list[dict[str, int | float | str | None]]:
        return [
            {
                "feature_value": 1.0,
                "token_text": f"tok{feature_index}",
            }
            for _ in range(top_k)
        ]

    interpreter.get_top_examples_for_feature = _examples

    result = cluster_features_kmeans(
        interpreter=interpreter,
        num_clusters=4,
        random_state=1,
    )

    assert "clusters" in result
    assert "cluster_summaries" in result
    assert len(result["cluster_summaries"]) == 4

    print_cluster_analysis(result)
    output = capsys.readouterr().out
    assert "K-MEANS CLUSTERING COMPLETE" in output
    assert "CLUSTER 0" in output


def test_print_cluster_analysis_dead_feature_branch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify summary printer handles clusters without representative features."""
    result = {
        "cluster_labels": torch.tensor([0]).numpy(),
        "kmeans_model": None,
        "clusters": {0: [0]},
        "cluster_summaries": [
            {
                "cluster_id": 0,
                "num_features": 1,
                "feature_indices": [0],
                "representative_feature": None,
                "representative_tokens": [],
            }
        ],
    }
    print_cluster_analysis(result)  # type: ignore[arg-type]
    output = capsys.readouterr().out
    assert "dead features" in output
