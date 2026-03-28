"""Pytest configuration and fixtures for model_wrapper tests.

This module provides reusable pytest fixtures for testing the ModelWrapper
class with lightweight models to avoid OOM errors on GPU-constrained systems.
"""

import sys
from collections.abc import Generator
from pathlib import Path

import pytest
import torch

# Add src to path so we can import brain_surgery module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_surgery.model_wrapper import ModelWrapper


@pytest.fixture  # type: ignore[misc]
def tiny_model_wrapper() -> ModelWrapper:
    """Fixture providing a ModelWrapper with a tiny test model.

    Uses hf-internal-testing/tiny-random-GPTJForCausalLM which is specifically
    designed for testing and is extremely lightweight (~2MB), avoiding OOM
    issues on constrained hardware like RTX 4070.

    Returns:
        ModelWrapper: An initialized ModelWrapper instance with the tiny model
            and layer_idx=1 (valid for small models with ~3-4 layers).

    Note:
        This fixture is function-scoped (reset for each test) to ensure
        test isolation and avoid state leakage between tests.
    """
    return ModelWrapper(
        model_name="hf-internal-testing/tiny-random-GPTJForCausalLM", layer_idx=1
    )


@pytest.fixture  # type: ignore[misc]
def gpt2_model_wrapper() -> ModelWrapper:
    """Fixture providing a ModelWrapper with GPT-2 (small but full model).

    Uses the standard GPT-2 model for tests that need more realistic model
    behavior while still being quick to load and run.

    Returns:
        ModelWrapper: An initialized ModelWrapper instance with GPT-2 and
            layer_idx=4 (middle layers for optimal SAE feature extraction).
    """
    return ModelWrapper(model_name="gpt2", layer_idx=4)


@pytest.fixture(autouse=True)  # type: ignore[misc]
def clear_gpu_cache() -> Generator[None, None, None]:
    """Fixture to clear GPU cache after each test.

    This is automatically used for all tests to prevent memory accumulation
    across test runs, which is important when running many tests in sequence.

    Yields:
        None: Test execution happens here.

    Note:
        torch.cuda.empty_cache() is safe to call even if CUDA is not available.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
