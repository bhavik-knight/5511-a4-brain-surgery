"""Shared pytest fixtures for the brain_surgery test suite."""

import sys
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_surgery.model_wrapper import ModelWrapper
from brain_surgery.sae import SparseAutoencoder


class FakeTokenizer:
    """Minimal tokenizer stub for deterministic unit tests."""

    pad_token_id: int | None = 0
    eos_token_id: int | None = 1
    pad_token: str | None = "<pad>"
    eos_token: str | None = "<eos>"

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        truncation: bool | None = None,
        max_length: int | None = None,
        padding: bool | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor | list[int]]:
        if return_tensors == "pt":
            ids = [2, 3, 4, 5]
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }

        token_to_id = {
            "Ronaldo": 10,
            "Messi": 11,
            "goal": 12,
            "football": 13,
        }
        token = text.strip()
        return {"input_ids": [token_to_id.get(token, 99)]}

    def decode(
        self, token_ids: object, skip_special_tokens: bool = True
    ) -> str | list[str]:
        _ = skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        normalized_ids: list[int] = []
        if isinstance(token_ids, int):
            normalized_ids = [token_ids]
        elif isinstance(token_ids, list):
            if token_ids and isinstance(token_ids[0], list):
                nested = token_ids[0]
                normalized_ids = [int(tid) for tid in nested if isinstance(tid, int)]
            else:
                normalized_ids = [int(tid) for tid in token_ids if isinstance(tid, int)]

        return " ".join(f"tok{tid}" for tid in normalized_ids)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str] | str:
        return [f"tok{tid}" for tid in token_ids]


class FakeLayer(nn.Module):  # type: ignore[misc]
    """Simple layer that preserves hidden state shape."""

    def __init__(self, hidden_dim: int = 896) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.bias


class FakeCausalLM(nn.Module):  # type: ignore[misc]
    """Tiny model stub exposing a 24-layer Qwen-like structure."""

    def __init__(
        self, layers: int = 24, hidden_dim: int = 896, vocab_size: int = 64
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers)
        self.model = SimpleNamespace(
            layers=nn.ModuleList([FakeLayer(hidden_dim) for _ in range(layers)])
        )
        self._dummy = nn.Parameter(torch.zeros(1))
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def eval(self) -> Self:
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        return_dict: bool = True,
        **kwargs: object,
    ) -> SimpleNamespace | torch.Tensor:
        hidden = F.one_hot(
            input_ids % self.hidden_dim, num_classes=self.hidden_dim
        ).float()
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        if return_dict:
            return SimpleNamespace(logits=logits)
        return logits

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        _ = self.forward(input_ids, return_dict=True)
        next_token = torch.full(
            (input_ids.shape[0], 1), 7, dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat([input_ids, next_token], dim=1)


@pytest.fixture(scope="session")  # type: ignore[untyped-decorator]
def sae_fixture() -> SparseAutoencoder:
    """Reusable SAE fixture to avoid redundant model construction."""
    return SparseAutoencoder(input_dim=896, latent_dim=3584)


@pytest.fixture  # type: ignore[untyped-decorator]
def mock_model_wrapper_24(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ModelWrapper:
    """Create a ModelWrapper wired to fake 24-layer model/tokenizer objects."""
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir(parents=True, exist_ok=True)

    fake_model = FakeCausalLM(layers=24, hidden_dim=896)
    fake_tokenizer = FakeTokenizer()

    monkeypatch.setattr(
        "brain_surgery.model_wrapper.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        "brain_surgery.model_wrapper.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: fake_tokenizer,
    )

    return ModelWrapper(model_name=str(model_dir), layer_idx=None)


@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def clear_gpu_cache() -> Generator[None, None, None]:
    """Clear CUDA allocator after each test to keep memory pressure low."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
