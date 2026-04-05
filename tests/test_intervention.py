"""Unit tests for SAEIntervention clamping and log-prob APIs."""

from types import SimpleNamespace
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_surgery.intervention import SAEIntervention
from brain_surgery.model_wrapper import ModelWrapper
from brain_surgery.sae import SparseAutoencoder


class FakeTokenizer:
    """Tokenizer stub for intervention tests."""

    pad_token_id = 0

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | list[int]]:
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }
        token = text.strip()
        token_map = {"Ronaldo": 10, "Messi": 11, "goal": 12}
        return {"input_ids": [token_map.get(token, 13)]}

    def decode(self, ids: object, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, list):
            return ""
        return " ".join(f"tok{int(i)}" for i in ids)


class FakeBlock(nn.Module):  # type: ignore[misc]
    """Module used to exercise forward hooks."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class TupleBlock(nn.Module):  # type: ignore[misc]
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, torch.zeros(1, dtype=hidden_states.dtype)


class FakeModel(nn.Module):  # type: ignore[misc]
    """Model stub exposing layers and logits for intervention tests."""

    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            layers=nn.ModuleList([FakeBlock() for _ in range(24)])
        )
        self._dummy = nn.Parameter(torch.zeros(1))
        self.clamped = False

    def forward(
        self,
        input_ids: torch.Tensor,
        return_dict: bool = True,
        **kwargs: object,
    ) -> SimpleNamespace:
        hidden = torch.ones(
            (input_ids.shape[0], input_ids.shape[1], 896), dtype=torch.float16
        )
        for layer in self.model.layers:
            hidden = layer(hidden)

        logits = torch.zeros((1, input_ids.shape[1], 32), dtype=torch.float32)
        if self.clamped:
            logits[0, -1, 10] = 3.0  # Ronaldo rises when clamped
        else:
            logits[0, -1, 10] = 1.0
        logits[0, -1, 11] = 0.5

        return SimpleNamespace(logits=logits)

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        _ = self.forward(input_ids, return_dict=True)
        return torch.cat([input_ids, torch.tensor([[7]], dtype=torch.long)], dim=1)


class TupleModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([TupleBlock()]))

    def forward(
        self,
        input_ids: torch.Tensor,
        return_dict: bool = True,
        **kwargs: object,
    ) -> SimpleNamespace:
        hidden = torch.ones(
            (input_ids.shape[0], input_ids.shape[1], 896), dtype=torch.float16
        )
        for layer in self.model.layers:
            out = layer(hidden)
            if isinstance(out, tuple):
                hidden = out[0]
            else:
                hidden = out

        logits = torch.zeros((1, input_ids.shape[1], 32), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


class FakeWrapper:
    """ModelWrapper-like object for intervention unit tests."""

    def __init__(self) -> None:
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.device = torch.device("cpu")

    def is_loaded(self) -> bool:
        return True


class UnloadedWrapper(FakeWrapper):
    def is_loaded(self) -> bool:
        return False


class NoLayerWrapper(FakeWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace()  # type: ignore[assignment]


class TupleWrapper(FakeWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.model = TupleModel()


def _as_model_wrapper(wrapper: FakeWrapper) -> ModelWrapper:
    return cast(ModelWrapper, wrapper)


def test_register_prompt_clamp_hook_modifies_and_preserves_fp16_dtype() -> None:
    """Verify clamping hook changes selected latent and outputs fp16 hidden states."""
    wrapper = FakeWrapper()
    sae = SparseAutoencoder(input_dim=896, latent_dim=3584)
    intervention = SAEIntervention(model_wrapper=_as_model_wrapper(wrapper), sae=sae)

    # Ensure feature max values exist for clamping.
    intervention.feature_max_values = torch.ones(sae.latent_dim)

    intervention.register_prompt_clamp_hook(feature_index=5, clamp_multiplier=2.0)

    # Trigger forward pass so hook runs.
    _ = wrapper.model.forward(
        torch.tensor([[1, 2, 3]], dtype=torch.long), return_dict=True
    )

    assert intervention.original_activations is not None
    assert intervention.modified_activations is not None
    assert intervention.original_activations.dtype == torch.float16
    assert intervention.modified_activations.dtype == torch.float16

    # Confirm hidden states changed by intervention.
    assert not torch.allclose(
        intervention.original_activations,
        intervention.modified_activations,
    )


def test_compare_next_token_logprobs_delta_is_sound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify clamped-vs-baseline log-prob shift math is correct."""
    wrapper = FakeWrapper()
    sae = SparseAutoencoder(input_dim=896, latent_dim=3584)
    intervention = SAEIntervention(model_wrapper=_as_model_wrapper(wrapper), sae=sae)
    intervention.feature_max_values = torch.ones(sae.latent_dim)

    def fake_register(feature_index: int, clamp_multiplier: float) -> None:
        _ = feature_index
        _ = clamp_multiplier
        wrapper.model.clamped = True

    def fake_remove() -> None:
        wrapper.model.clamped = False

    monkeypatch.setattr(intervention, "register_prompt_clamp_hook", fake_register)
    monkeypatch.setattr(intervention, "remove_hook", fake_remove)

    baseline = intervention.compare_next_token_logprobs(
        "Who won?",
        [" Ronaldo", " Messi"],
    )
    assert isinstance(baseline, dict)
    clamped = intervention.compare_next_token_logprobs(
        "Who won?",
        [" Ronaldo", " Messi"],
        feature_index=1625,
        clamp_multiplier=8.0,
    )
    assert isinstance(clamped, dict)

    delta_ronaldo = clamped[" Ronaldo"] - baseline[" Ronaldo"]
    baseline_logits = torch.zeros(32)
    baseline_logits[10] = 1.0
    baseline_logits[11] = 0.5
    clamped_logits = torch.zeros(32)
    clamped_logits[10] = 3.0
    clamped_logits[11] = 0.5

    expected_delta = (
        F.log_softmax(clamped_logits, dim=0)[10]
        - F.log_softmax(baseline_logits, dim=0)[10]
    ).item()
    assert abs(delta_ronaldo - expected_delta) < 1e-6


def test_generate_with_clamped_feature_returns_typed_payload() -> None:
    """Verify generation API returns tensor-backed intervention payload."""
    wrapper = FakeWrapper()
    sae = SparseAutoencoder(input_dim=896, latent_dim=3584)
    intervention = SAEIntervention(model_wrapper=_as_model_wrapper(wrapper), sae=sae)
    intervention.feature_max_values = torch.ones(sae.latent_dim)

    result = intervention.generate_with_clamped_feature(
        prompt="Who scored?",
        feature_index=12,
        clamp_multiplier=4.0,
        max_new_tokens=5,
    )

    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["generated_ids"], torch.Tensor)
    assert isinstance(result["generated_text"], str)


def test_register_prompt_clamp_hook_invalid_index_raises() -> None:
    """Verify feature index bounds check is enforced."""
    wrapper = FakeWrapper()
    sae = SparseAutoencoder(input_dim=896, latent_dim=3584)
    intervention = SAEIntervention(model_wrapper=_as_model_wrapper(wrapper), sae=sae)
    intervention.feature_max_values = torch.ones(sae.latent_dim)

    with pytest.raises(IndexError):
        intervention.register_prompt_clamp_hook(
            feature_index=sae.latent_dim + 1,
            clamp_multiplier=2.0,
        )


def test_intervention_requires_loaded_wrapper() -> None:
    """Verify constructor rejects wrappers that are not loaded."""
    with pytest.raises(ValueError):
        SAEIntervention(
            model_wrapper=_as_model_wrapper(UnloadedWrapper()),
            sae=SparseAutoencoder(),
        )


def test_intervention_requires_sae_or_checkpoint() -> None:
    """Verify constructor rejects missing SAE and checkpoint inputs."""
    with pytest.raises(ValueError):
        SAEIntervention(
            model_wrapper=_as_model_wrapper(FakeWrapper()),
            sae=None,
            checkpoint_path=None,
        )


def test_compare_logprobs_clamp_requires_feature_max_values() -> None:
    """Verify clamped scoring rejects missing feature max values."""
    intervention = SAEIntervention(
        model_wrapper=_as_model_wrapper(FakeWrapper()),
        sae=SparseAutoencoder(input_dim=896, latent_dim=3584),
    )
    with pytest.raises(RuntimeError):
        intervention.compare_next_token_logprobs(
            "Prompt",
            [" Ronaldo"],
            feature_index=1,
            clamp_multiplier=2.0,
        )


def test_intervention_loads_sae_from_checkpoint(tmp_path: Path) -> None:
    """Verify checkpoint path constructor branch loads SAE weights."""
    sae = SparseAutoencoder(input_dim=896, latent_dim=128)
    checkpoint = tmp_path / "sae.pt"
    torch.save(sae.state_dict_for_checkpoint(), checkpoint)

    intervention = SAEIntervention(
        model_wrapper=_as_model_wrapper(FakeWrapper()),
        checkpoint_path=checkpoint,
    )
    assert intervention.sae is not None
    assert intervention.sae.latent_dim == 128


def test_load_sae_requires_checkpoint_path() -> None:
    """Verify explicit load_sae() validation when checkpoint path is absent."""
    intervention = SAEIntervention(
        model_wrapper=_as_model_wrapper(FakeWrapper()),
        sae=SparseAutoencoder(input_dim=896, latent_dim=3584),
    )
    intervention.checkpoint_path = None
    with pytest.raises(ValueError):
        intervention.load_sae()


def test_get_transformer_blocks_raises_for_unsupported_model_shape() -> None:
    """Verify unsupported wrapped model shape raises clear runtime error."""
    intervention = SAEIntervention(
        model_wrapper=_as_model_wrapper(NoLayerWrapper()),
        sae=SparseAutoencoder(input_dim=896, latent_dim=64),
    )
    intervention.feature_max_values = torch.ones(64)
    with pytest.raises(RuntimeError):
        intervention.register_prompt_clamp_hook(feature_index=1, clamp_multiplier=2.0)


def test_generate_with_clamped_feature_raises_when_sae_missing() -> None:
    """Verify generation path rejects missing SAE state."""
    intervention = SAEIntervention(
        model_wrapper=_as_model_wrapper(FakeWrapper()),
        sae=SparseAutoencoder(input_dim=896, latent_dim=64),
    )
    intervention.sae = None
    with pytest.raises(RuntimeError):
        intervention.generate_with_clamped_feature(
            prompt="Q",
            feature_index=1,
            clamp_multiplier=2.0,
        )


def test_compute_feature_max_values_requires_loaded_sae() -> None:
    """Verify compute_feature_max_values rejects missing SAE."""
    intervention = SAEIntervention(
        model_wrapper=_as_model_wrapper(FakeWrapper()),
        sae=SparseAutoencoder(input_dim=896, latent_dim=64),
    )
    intervention.sae = None
    with pytest.raises(RuntimeError):
        intervention.compute_feature_max_values(torch.randn(2, 896))


def test_register_hook_requires_feature_max_values() -> None:
    """Verify hook registration requires precomputed feature max values."""
    intervention = SAEIntervention(
        model_wrapper=_as_model_wrapper(FakeWrapper()),
        sae=SparseAutoencoder(input_dim=896, latent_dim=64),
    )
    intervention.feature_max_values = None
    with pytest.raises(RuntimeError):
        intervention.register_prompt_clamp_hook(feature_index=1, clamp_multiplier=2.0)


def test_hook_tuple_output_branch_is_supported() -> None:
    """Verify hook handles tuple outputs and returns tuple branch safely."""
    intervention = SAEIntervention(
        model_wrapper=_as_model_wrapper(TupleWrapper()),
        sae=SparseAutoencoder(input_dim=896, latent_dim=64),
    )
    intervention.feature_max_values = torch.ones(64)
    intervention.register_prompt_clamp_hook(feature_index=1, clamp_multiplier=2.0)
    _ = intervention.model_wrapper.model.forward(
        torch.tensor([[1, 2]], dtype=torch.long)
    )
    assert intervention.modified_activations is not None


def test_compare_next_token_skips_multi_token_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify multi-token candidate ids are skipped from scored output map."""

    class MultiTokenTokenizer(FakeTokenizer):
        def __call__(
            self,
            text: str,
            return_tensors: str | None = None,
            add_special_tokens: bool = True,
            **kwargs: object,
        ) -> dict[str, torch.Tensor | list[int]]:
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                }
            if text.strip() == "Ronaldo":
                return {"input_ids": [10, 11]}
            return {"input_ids": [12]}

    wrapper = FakeWrapper()
    wrapper.tokenizer = MultiTokenTokenizer()
    intervention = SAEIntervention(
        model_wrapper=_as_model_wrapper(wrapper),
        sae=SparseAutoencoder(input_dim=896, latent_dim=64),
    )
    scores = intervention.compare_next_token_logprobs("Q", ["Ronaldo", "goal"])
    assert "Ronaldo" not in scores
    assert "goal" in scores
