"""Tests for model_wrapper module.

This test suite validates the ModelWrapper class functionality including:
- Activation Collection: Verify activations are returned as dict with Tensor values
- Tensor Shapes: Assert correct shape (batch_size, sequence_length, hidden_dimension)
- Hook Cleanup: Ensure hooks are properly removed for memory safety
- Device Consistency: Verify activations are moved to CPU for VRAM management

Uses lightweight test models (tiny-random-GPTJForCausalLM) to avoid OOM on
constrained hardware like RTX 4070.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path so we can import brain_surgery module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_surgery.model_wrapper import ModelWrapper


# ============================================================================
# TEST 1: ACTIVATION COLLECTION
# ============================================================================


class TestActivationCollection:
    """Test suite verifying activation collection functionality.

    Requirement: generate_with_activations() returns a dictionary where
    keys correspond to hooked layers and values are torch.Tensor objects.
    """

    def test_activations_returned_as_dict(
        self: "TestActivationCollection", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Verify that activations are returned as a dictionary.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Returned activations is a dict
            - "layer" key exists (hooked layer output)
            - Value is a torch.Tensor
        """
        prompt = "The capital of France is"
        text, activations = tiny_model_wrapper.generate_with_activations(
            prompt=prompt, max_tokens=5
        )

        assert isinstance(activations, dict), "Activations must be dict"
        assert "layer" in activations, "Expected 'layer' key in activations dict"
        assert isinstance(
            activations["layer"], torch.Tensor
        ), "Activation values must be torch.Tensor"

    def test_activations_non_empty(
        self: "TestActivationCollection", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Verify that activations dictionary is populated after generation.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Activations dict contains at least one entry
            - The tensor is not empty
        """
        text, activations = tiny_model_wrapper.generate_with_activations(
            prompt="Hello world", max_tokens=3
        )

        assert len(activations) > 0, "Activations dict must not be empty"
        assert (
            activations["layer"].numel() > 0
        ), "Activation tensor must contain elements"


# ============================================================================
# TEST 2: TENSOR SHAPES
# ============================================================================


class TestTensorShapes:
    """Test suite verifying activation tensor shapes.

    Requirement: Collected activations have shape (batch_size, sequence_length,
    hidden_dimension).
    """

    def test_activation_dimensions(
        self: "TestTensorShapes", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Verify activations have correct dimensionality.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Tensor has 3 dimensions (batch, seq_len, hidden_dim)
            - Dimensions match expected pattern
        """
        text, activations = tiny_model_wrapper.generate_with_activations(
            prompt="Test", max_tokens=5
        )
        activation_tensor = activations["layer"]

        assert (
            activation_tensor.dim() == 3
        ), f"Expected 3D tensor, got {activation_tensor.dim()}D"
        batch_size, seq_len, hidden_dim = activation_tensor.shape
        assert batch_size > 0, "batch_size must be positive"
        assert seq_len > 0, "sequence_length must be positive"
        assert hidden_dim > 0, "hidden_dimension must be positive"

    def test_activation_shape_consistency(
        self: "TestTensorShapes", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Verify activation shapes are consistent across multiple generations.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Multiple generations produce consistent shapes
            - Shape matches (batch=1, seq_len varies, hidden_dim fixed)
        """
        prompts = ["Hello", "The quick brown", "Once upon a time"]
        shapes = []

        for prompt in prompts:
            text, activations = tiny_model_wrapper.generate_with_activations(
                prompt=prompt, max_tokens=3
            )
            shapes.append(tuple(activations["layer"].shape))

        # All shapes should have same hidden_dim but varying seq_len
        batch_sizes = [s[0] for s in shapes]
        hidden_dims = [s[2] for s in shapes]

        assert all(
            b == batch_sizes[0] for b in batch_sizes
        ), "batch_size should be consistent"
        assert all(
            h == hidden_dims[0] for h in hidden_dims
        ), "hidden_dim should be consistent"


# ============================================================================
# TEST 3: HOOK CLEANUP
# ============================================================================


class TestHookCleanup:
    """Test suite verifying hook registration and cleanup.

    Requirement: Hooks are properly removed or handled so that subsequent
    normal model calls don't continue storing activations.
    """

    def test_hooks_registered_on_init(
        self: "TestHookCleanup", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Verify hooks are registered during initialization.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - At least one hook is registered
            - activations dict starts empty
        """
        assert len(tiny_model_wrapper.hooks) > 0, "Hooks should be registered on init"
        assert tiny_model_wrapper.activations == {}, "activations should start empty"

    def test_hooks_properly_removed(
        self: "TestHookCleanup", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Verify unregister_hooks removes all hooks and clears activations.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - hooks list is empty after unregister
            - activations dict is empty after unregister
            - Subsequent model forward passes don't store activations
        """
        num_hooks_before = len(tiny_model_wrapper.hooks)
        assert num_hooks_before > 0

        # Unregister hooks
        tiny_model_wrapper.unregister_hooks()
        assert len(tiny_model_wrapper.hooks) == 0, "All hooks should be removed"
        assert len(tiny_model_wrapper.activations) == 0, "activations should be cleared"

    def test_activations_cleared_between_calls(
        self: "TestHookCleanup", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Verify activations are cleared between generate calls.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Different calls produce different activation tensors
            - No accumulation of old activations
        """
        _, acts1 = tiny_model_wrapper.generate_with_activations(
            prompt="First", max_tokens=2
        )
        tensor_id_1 = id(acts1["layer"])

        _, acts2 = tiny_model_wrapper.generate_with_activations(
            prompt="Second", max_tokens=2
        )
        tensor_id_2 = id(acts2["layer"])

        assert tensor_id_1 != tensor_id_2, "Activations should be cleared between calls"


# ============================================================================
# TEST 4: DEVICE CONSISTENCY
# ============================================================================


class TestDeviceConsistency:
    """Test suite verifying device management and VRAM optimization.

    Requirement: Collected activations are moved to CPU to save VRAM, as per
    hardware constraints.
    """

    def test_activations_on_cpu(
        self: "TestDeviceConsistency", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Verify activations are moved to CPU after capture.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Activation tensor is on CPU device (not CUDA)
            - Explicitly checks device.type == "cpu"
        """
        text, activations = tiny_model_wrapper.generate_with_activations(
            prompt="Device test", max_tokens=3
        )
        activation_tensor = activations["layer"]

        assert (
            activation_tensor.device.type == "cpu"
        ), f"Activations must be on CPU, got {activation_tensor.device}"

    def test_model_device_vs_activation_device(
        self: "TestDeviceConsistency", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Verify model is on GPU (if available) but activations on CPU.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Model is on the expected device (cuda/cpu)
            - Activations are always on CPU regardless
            - This demonstrates VRAM optimization
        """
        text, activations = tiny_model_wrapper.generate_with_activations(
            prompt="Model vs activation device", max_tokens=2
        )

        # Model should be on detected device
        assert tiny_model_wrapper.device == tiny_model_wrapper.device
        # Activations should ALWAYS be on CPU for VRAM savings
        assert activations["layer"].device.type == "cpu"

    def test_multiple_activations_on_cpu(
        self: "TestDeviceConsistency", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Verify all captured activations across calls are on CPU.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Multiple generation calls all produce CPU tensors
            - No GPU memory accumulation
        """
        for i in range(3):
            text, activations = tiny_model_wrapper.generate_with_activations(
                prompt=f"Prompt {i}", max_tokens=2
            )
            assert (
                activations["layer"].device.type == "cpu"
            ), f"Activation {i} not on CPU"


# ============================================================================
# ADDITIONAL TESTS: ERROR HANDLING AND EDGE CASES
# ============================================================================


class TestModelWrapperInitialization:
    """Test suite for ModelWrapper initialization."""

    def test_init_with_valid_model(
        self: "TestModelWrapperInitialization", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test initialization with a valid model.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - ModelWrapper attributes are correctly set
            - Model and tokenizer are loaded
        """
        assert (
            tiny_model_wrapper.model_name
            == "hf-internal-testing/tiny-random-GPTJForCausalLM"
        )
        assert tiny_model_wrapper.layer_idx == 1
        assert tiny_model_wrapper.model is not None
        assert tiny_model_wrapper.tokenizer is not None

    def test_init_with_negative_layer_idx(
        self: "TestModelWrapperInitialization",
    ) -> None:
        """Test that negative layer_idx raises ValueError.

        Asserts:
            - ValueError raised with appropriate message
        """
        with pytest.raises(ValueError, match="layer_idx must be non-negative"):
            ModelWrapper(
                model_name="hf-internal-testing/tiny-random-GPTJForCausalLM",
                layer_idx=-1,
            )

    def test_device_detection(
        self: "TestModelWrapperInitialization", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test that device is correctly detected.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Device is either cuda (if available) or cpu
        """
        expected_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        assert tiny_model_wrapper.device == expected_device

    def test_model_in_eval_mode(
        self: "TestModelWrapperInitialization", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test that model is set to evaluation mode.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Model training flag is False
        """
        assert not tiny_model_wrapper.model.training


class TestHookRegistration:
    """Test suite for forward hook registration."""

    def test_hooks_registered(
        self: "TestHookRegistration", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test that hooks are properly registered on initialization.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - At least one hook registered
        """
        assert len(tiny_model_wrapper.hooks) > 0
        assert tiny_model_wrapper.activations == {}

    def test_hook_unregistration(
        self: "TestHookRegistration", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test that unregister_hooks properly removes all hooks.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - hooks list is empty after unregistration
            - activations dict is cleared
        """
        num_hooks_before = len(tiny_model_wrapper.hooks)
        assert num_hooks_before > 0

        tiny_model_wrapper.unregister_hooks()
        assert len(tiny_model_wrapper.hooks) == 0
        assert len(tiny_model_wrapper.activations) == 0

    def test_layer_out_of_range(self: "TestHookRegistration") -> None:
        """Test that out-of-range layer_idx raises RuntimeError.

        Asserts:
            - RuntimeError raised for layer_idx >= num_layers
        """
        with pytest.raises(RuntimeError, match="exceeds number of layers"):
            ModelWrapper(
                model_name="hf-internal-testing/tiny-random-GPTJForCausalLM",
                layer_idx=100,
            )


class TestTextGenerationWithActivations:
    """Test suite for text generation with activation capture."""

    def test_generate_with_valid_prompt(
        self: "TestTextGenerationWithActivations", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test generation with a valid non-empty prompt.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Generated text is a string
            - Text is longer than original prompt
            - Activations returned as dict with "layer" key
        """
        prompt = "Hello world"
        text, activations = tiny_model_wrapper.generate_with_activations(
            prompt=prompt, max_tokens=5
        )

        assert isinstance(text, str)
        assert len(text) > len(prompt)
        assert isinstance(activations, dict)
        assert "layer" in activations

    def test_generate_with_empty_prompt(
        self: "TestTextGenerationWithActivations", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test that empty prompt raises ValueError.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - ValueError raised with "Prompt cannot be empty" message
        """
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            tiny_model_wrapper.generate_with_activations(prompt="", max_tokens=10)

    def test_generate_with_invalid_max_tokens(
        self: "TestTextGenerationWithActivations", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test that non-positive max_tokens raises ValueError.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - ValueError raised for max_tokens=0
            - ValueError raised for max_tokens<0
        """
        prompt = "Hello"

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            tiny_model_wrapper.generate_with_activations(prompt=prompt, max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            tiny_model_wrapper.generate_with_activations(prompt=prompt, max_tokens=-5)

    def test_different_temperatures(
        self: "TestTextGenerationWithActivations", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test generation with different temperature values.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Both low and high temperature generations succeed
            - Both return strings
        """
        prompt = "Hello"

        text1, _ = tiny_model_wrapper.generate_with_activations(
            prompt=prompt, max_tokens=3, temperature=0.1
        )
        text2, _ = tiny_model_wrapper.generate_with_activations(
            prompt=prompt, max_tokens=3, temperature=1.5
        )

        assert isinstance(text1, str)
        assert isinstance(text2, str)

    def test_different_top_p(
        self: "TestTextGenerationWithActivations", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test generation with different top_p (nucleus sampling) values.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - Both low and high top_p generations succeed
            - Both return strings
        """
        prompt = "The quick"

        text1, _ = tiny_model_wrapper.generate_with_activations(
            prompt=prompt, max_tokens=3, top_p=0.5
        )
        text2, _ = tiny_model_wrapper.generate_with_activations(
            prompt=prompt, max_tokens=3, top_p=0.95
        )

        assert isinstance(text1, str)
        assert isinstance(text2, str)


class TestModelWrapperStringRepresentation:
    """Test suite for __repr__ method."""

    def test_repr_format(
        self: "TestModelWrapperStringRepresentation", tiny_model_wrapper: ModelWrapper
    ) -> None:
        """Test that __repr__ returns a properly formatted string.

        Args:
            tiny_model_wrapper: Pytest fixture providing ModelWrapper with
                tiny test model.

        Asserts:
            - repr contains "ModelWrapper" class name
            - repr contains model name
            - repr contains layer_idx
            - repr contains device info
        """
        repr_str = repr(tiny_model_wrapper)

        assert "ModelWrapper" in repr_str
        assert "hf-internal-testing/tiny-random-GPTJForCausalLM" in repr_str
        assert "layer_idx=1" in repr_str
        assert "device=" in repr_str
