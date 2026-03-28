"""Pytest configuration and fixtures for model_wrapper tests.

This module provides reusable pytest fixtures for testing the ModelWrapper
class with lightweight models to avoid OOM errors on GPU-constrained systems.
"""

import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, TypeVar, cast, overload

import pytest
import torch

# Add src to path so we can import brain_surgery module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_surgery.model_wrapper import ModelWrapper

F = TypeVar("F", bound=Callable[..., Any])


@overload
def typed_fixture(func: F, /) -> F: ...


@overload
def typed_fixture(**kwargs: object) -> Callable[[F], F]: ...


def typed_fixture(func: F | None = None, /, **kwargs: object) -> F | Callable[[F], F]:
    """A typed wrapper around pytest.fixture for mypy --strict.

    Supports both:
    - `@typed_fixture`
    - `@typed_fixture(scope="session", autouse=True, ...)`
    """
    # Supports both:
    #   @typed_fixture
    #   def fx(...): ...
    # and
    #   @typed_fixture(scope="session")
    #   def fx(...): ...
    if func is not None:
        return cast(F, pytest.fixture(func))

    def _decorate(inner: F) -> F:
        decorator = cast(Any, pytest.fixture)(**kwargs)
        return cast(Callable[[F], F], decorator)(inner)

    return _decorate


@typed_fixture(scope="session")
def tiny_local_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a tiny local model+tokenizer directory for fully-offline tests.

    This avoids relying on network access or an existing Hugging Face cache.

    Returns:
        Path: Directory containing a saved model + tokenizer compatible with
            AutoModelForCausalLM/AutoTokenizer.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    out_dir = cast(Path, tmp_path_factory.mktemp("tiny_local_gpt2"))

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "Hello": 4,
        "world": 5,
        "The": 6,
        "capital": 7,
        "of": 8,
        "France": 9,
        "is": 10,
        "Test": 11,
        "Prompt": 12,
        "Model": 13,
        "vs": 14,
        "activation": 15,
        "device": 16,
    }

    tokenizer = cast(Any, Tokenizer)(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = cast(Any, PreTrainedTokenizerFast)(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    hf_tokenizer.save_pretrained(out_dir)

    config = cast(Any, GPT2Config)(
        vocab_size=hf_tokenizer.vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=32,
        bos_token_id=hf_tokenizer.bos_token_id,
        eos_token_id=hf_tokenizer.eos_token_id,
        pad_token_id=hf_tokenizer.pad_token_id,
    )
    model = cast(Any, GPT2LMHeadModel)(config)
    model.save_pretrained(out_dir)

    return out_dir


@typed_fixture
def tiny_model_wrapper(tiny_local_model_dir: Path) -> ModelWrapper:
    """Fixture providing a ModelWrapper with a tiny test model.

    Uses a tiny local GPT-2 style model saved to a temp directory, avoiding
    network access and Hugging Face cache assumptions.

    Returns:
        ModelWrapper: An initialized ModelWrapper instance with the tiny model
            and layer_idx=1 (valid for small models with ~3-4 layers).

    Note:
        This fixture is function-scoped (reset for each test) to ensure
        test isolation and avoid state leakage between tests.
    """
    return ModelWrapper(model_name=str(tiny_local_model_dir), layer_idx=1)


@typed_fixture(autouse=True)
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
