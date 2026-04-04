"""Unit tests for shared utilities and configuration helpers."""

from pathlib import Path

import torch

import brain_surgery.utils as utils


def test_ensure_dir_exists_creates_path(tmp_path: Path) -> None:
    """Verify nested directory creation helper returns existing path."""
    target = tmp_path / "a" / "b" / "c"
    out = utils.ensure_dir_exists(target)
    assert out == target
    assert out.exists()


def test_get_recommended_layer_idx_midpoint() -> None:
    """Verify midpoint selection helper."""
    assert utils.get_recommended_layer_idx(24) == 12


def test_get_device_name_cuda_branch(monkeypatch) -> None:
    """Verify device-name helper handles CUDA-available branch."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert utils.get_device_name() == "cuda"


def test_get_device_cpu_branch(monkeypatch) -> None:
    """Verify device helper returns CPU when CUDA is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert utils.get_device().type == "cpu"
