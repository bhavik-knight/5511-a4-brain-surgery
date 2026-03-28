"""Tests for model_wrapper module.

This test suite validates the ModelWrapper class functionality including:
- Proper initialization with Hugging Face models
- Forward hook registration and activation capture
- Text generation with activation collection
- Error handling for invalid inputs
"""

import pytest
import torch
from brain_surgery.model_wrapper import ModelWrapper


class TestModelWrapperInitialization:
    """Test suite for ModelWrapper initialization."""

    def test_init_with_valid_model(self: "TestModelWrapperInitialization") -> None:
        """Test initialization with a valid model name.

        This test verifies that ModelWrapper can be instantiated with
        a valid Hugging Face model identifier and layer index.
        """
        # Using a small model for testing
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=4)
        assert wrapper.model_name == "gpt2"
        assert wrapper.layer_idx == 4
        assert wrapper.model is not None
        assert wrapper.tokenizer is not None

    def test_init_with_negative_layer_idx(
        self: "TestModelWrapperInitialization",
    ) -> None:
        """Test that negative layer_idx raises ValueError."""
        with pytest.raises(ValueError, match="layer_idx must be non-negative"):
            ModelWrapper(model_name="gpt2", layer_idx=-1)

    def test_device_detection(self: "TestModelWrapperInitialization") -> None:
        """Test that device is correctly detected (cuda or cpu)."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)
        expected_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        assert wrapper.device == expected_device

    def test_model_in_eval_mode(self: "TestModelWrapperInitialization") -> None:
        """Test that model is set to evaluation mode after initialization."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)
        assert not wrapper.model.training


class TestHookRegistration:
    """Test suite for forward hook registration."""

    def test_hooks_registered(self: "TestHookRegistration") -> None:
        """Test that hooks are properly registered on initialization."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)
        assert len(wrapper.hooks) > 0
        assert wrapper.activations == {}

    def test_hook_unregistration(self: "TestHookRegistration") -> None:
        """Test that unregister_hooks properly removes all hooks."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)
        num_hooks_before = len(wrapper.hooks)
        assert num_hooks_before > 0

        wrapper.unregister_hooks()
        assert len(wrapper.hooks) == 0
        assert len(wrapper.activations) == 0

    def test_layer_out_of_range(self: "TestHookRegistration") -> None:
        """Test that layer_idx >= num_layers raises RuntimeError."""
        # GPT2 has 12 layers, so 100 should be out of range
        with pytest.raises(RuntimeError, match="exceeds number of layers"):
            ModelWrapper(model_name="gpt2", layer_idx=100)


class TestTextGenerationWithActivations:
    """Test suite for text generation with activation capture."""

    def test_generate_with_valid_prompt(
        self: "TestTextGenerationWithActivations",
    ) -> None:
        """Test generation with a valid non-empty prompt."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)
        prompt = "The capital of France is"
        text, activations = wrapper.generate_with_activations(
            prompt=prompt, max_tokens=10
        )

        assert isinstance(text, str)
        assert len(text) > len(prompt)
        assert isinstance(activations, dict)
        assert "layer" in activations

    def test_generate_with_empty_prompt(
        self: "TestTextGenerationWithActivations",
    ) -> None:
        """Test that empty prompt raises ValueError."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            wrapper.generate_with_activations(prompt="", max_tokens=10)

    def test_generate_with_invalid_max_tokens(
        self: "TestTextGenerationWithActivations",
    ) -> None:
        """Test that non-positive max_tokens raises ValueError."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)
        prompt = "Hello"

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            wrapper.generate_with_activations(prompt=prompt, max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            wrapper.generate_with_activations(prompt=prompt, max_tokens=-5)

    def test_activations_shape(self: "TestTextGenerationWithActivations") -> None:
        """Test that captured activations have correct shape and type."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)
        text, activations = wrapper.generate_with_activations(
            prompt="Hello", max_tokens=5
        )

        assert "layer" in activations
        activation_tensor = activations["layer"]
        assert isinstance(activation_tensor, torch.Tensor)
        assert activation_tensor.dim() == 3  # (batch, seq_len, hidden_dim)
        # Activations should be on CPU
        assert activation_tensor.device.type == "cpu"

    def test_different_temperatures(self: "TestTextGenerationWithActivations") -> None:
        """Test generation with different temperature values."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)
        prompt = "The future is"

        text1, _ = wrapper.generate_with_activations(
            prompt=prompt, max_tokens=10, temperature=0.1
        )
        text2, _ = wrapper.generate_with_activations(
            prompt=prompt, max_tokens=10, temperature=1.5
        )

        # Different temperatures should (likely) produce different outputs
        assert isinstance(text1, str)
        assert isinstance(text2, str)

    def test_different_top_p(self: "TestTextGenerationWithActivations") -> None:
        """Test generation with different top_p (nucleus sampling) values."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)
        prompt = "Once upon a time"

        text1, _ = wrapper.generate_with_activations(
            prompt=prompt, max_tokens=10, top_p=0.5
        )
        text2, _ = wrapper.generate_with_activations(
            prompt=prompt, max_tokens=10, top_p=0.95
        )

        assert isinstance(text1, str)
        assert isinstance(text2, str)


class TestModelWrapperStringRepresentation:
    """Test suite for __repr__ method."""

    def test_repr_format(self: "TestModelWrapperStringRepresentation") -> None:
        """Test that __repr__ returns a properly formatted string."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=5)
        repr_str = repr(wrapper)

        assert "ModelWrapper" in repr_str
        assert "gpt2" in repr_str
        assert "layer_idx=5" in repr_str
        assert "device=" in repr_str


class TestMultipleGenerations:
    """Test suite for multiple sequential generations."""

    def test_activations_cleared_between_calls(self: "TestMultipleGenerations") -> None:
        """Test that activations are properly cleared between calls."""
        wrapper = ModelWrapper(model_name="gpt2", layer_idx=0)

        # First generation
        _, acts1 = wrapper.generate_with_activations(prompt="First", max_tokens=5)
        id1 = id(acts1["layer"])

        # Second generation
        _, acts2 = wrapper.generate_with_activations(prompt="Second", max_tokens=5)
        id2 = id(acts2["layer"])

        # Should be different tensor objects
        assert id1 != id2
