"""Model wrapper with forward hooks for capturing transformer activations.

This module provides utilities to wrap a pre-trained language model and register
forward hooks on specific layers to capture activation tensors during generation.
Activations are extracted from the residual stream of middle transformer layers,
enabling mechanistic interpretability analysis via Sparse Autoencoders.

Typical usage:
    >>> wrapper = ModelWrapper(model_name="Qwen/Qwen2.5-0.5B", layer_idx=4)
    >>> text, activations = wrapper.generate_with_activations(
    ...     prompt="The capital of France is",
    ...     max_tokens=20
    ... )
    >>> print(f"Generated: {text}")
    >>> print(f"Activations shape: {activations.shape}")
"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class ModelWrapper:
    """Wraps a pre-trained language model with forward hooks for activation capture.

    This class loads a causal language model and registers a forward hook on a
    specified transformer layer's residual stream. During text generation, activations
    from that layer are captured and returned alongside the generated text.

    Attributes:
        model_name: Hugging Face model identifier (e.g., "Qwen/Qwen2.5-0.5B").
        layer_idx: Index of the transformer layer to hook (0-indexed).
        model: The loaded pre-trained language model.
        tokenizer: The tokenizer for the model.
        device: Device on which the model runs (cuda or cpu).
        activations: Dictionary storing captured activations by layer.
        hooks: List of registered hook handles for cleanup.

    Example:
        >>> wrapper = ModelWrapper("Qwen/Qwen2.5-0.5B", layer_idx=4)
        >>> text, acts = wrapper.generate_with_activations("Hello", max_tokens=10)
        >>> print(acts.shape)  # (seq_len, hidden_dim)
    """

    def __init__(self: "ModelWrapper", model_name: str, layer_idx: int) -> None:
        """Initialize the ModelWrapper with a pre-trained model.

        Args:
            model_name: Name or path of the Hugging Face model to load.
            layer_idx: Index of the transformer layer to capture activations from.
                For Qwen-2.5-0.5B (8 layers), recommended: 3-5 for middle layers.

        Raises:
            ValueError: If layer_idx is negative.
            OSError: If model cannot be downloaded or loaded.
        """
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")

        self.model_name: str = model_name
        self.layer_idx: int = layer_idx
        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Load model and tokenizer from Hugging Face
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.to(self.device)  # pyright: ignore[reportCallIssue]
        self.model.eval()  # Set to evaluation mode

        # Storage for captured activations
        self.activations: dict[str, Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Register hooks on the target layer
        self._register_hooks()

    def _register_hooks(self: "ModelWrapper") -> None:
        """Register forward hooks on the residual stream of the target layer.

        This method registers hooks on the specified transformer layer's output,
        capturing the residual activation before layer normalization. Hooks are
        stored for later cleanup via unregister_hooks().

        Note:
            Hook placement on residual stream (vs. MLP output) captures the full
            attention output + residual, providing richer signal for SAE training.

        Raises:
            RuntimeError: If layer index is out of range for the model.
        """
        # Access the model's transformer layers
        if hasattr(self.model, "model"):
            # For models like Qwen that use model.model.layers
            layers = self.model.model.layers  # pyright: ignore[reportAttributeAccessIssue]
        else:
            # Fallback for other architectures
            layers = self.model.transformer.h  # pyright: ignore[reportAttributeAccessIssue]

        if self.layer_idx >= len(layers):  # pyright: ignore[reportArgumentType]
            raise RuntimeError(
                f"layer_idx {self.layer_idx} exceeds number of layers ({len(layers)})"  # pyright: ignore[reportArgumentType]
            )

        target_layer = layers[self.layer_idx]  # pyright: ignore[reportIndexIssue]

        # Hook function to capture activations
        def hook_fn(
            module: nn.Module,
            input_data: tuple[Tensor, ...],
            output_data: Tensor | tuple[Tensor, ...],
        ) -> None:
            """Forward hook to capture layer output activations.

            Args:
                module: The module being hooked.
                input_data: Input tensors to the module.
                output_data: Output from the module (tensor or tuple of tensors).
            """
            # Handle different output formats
            if isinstance(output_data, tuple):
                # Some models return (hidden_states, ...)
                hidden_states = output_data[0]
            else:
                # Others return just the tensor
                hidden_states = output_data

            # Store activations in CPU memory to save GPU memory
            self.activations["layer"] = hidden_states.detach().cpu()

        # Register the hook
        hook_handle = target_layer.register_forward_hook(hook_fn)  # pyright: ignore[reportAttributeAccessIssue]
        self.hooks.append(hook_handle)

    def unregister_hooks(self: "ModelWrapper") -> None:
        """Unregister all forward hooks to free memory and restore normal behavior.

        This should be called when activation capture is no longer needed.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()

    def generate_with_activations(
        self: "ModelWrapper",
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> tuple[str, dict[str, Tensor]]:
        """Generate text from a prompt while capturing layer activations.

        This method tokenizes the prompt, generates tokens via the model,
        and returns both the generated text and the captured activations from
        the hooked layer.

        Args:
            prompt: Input text prompt to generate from.
            max_tokens: Maximum number of tokens to generate (default: 50).
            temperature: Sampling temperature for generation (default: 0.7).
            top_p: Nucleus sampling parameter (default: 0.95).

        Returns:
            A tuple of:
            - generated_text (str): The full text (prompt + generated).
            - activations_dict (dict): Dictionary with key "layer" mapping to
              a tensor of shape (seq_len, hidden_dim) containing the captured
              residual stream activations.

        Raises:
            ValueError: If prompt is empty or max_tokens <= 0.
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        # Clear previous activations
        self.activations.clear()

        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        input_ids = inputs["input_ids"].to(self.device)  # pyright: ignore[reportAttributeAccessIssue]

        # Generate tokens with activation capture
        with torch.no_grad():
            output_ids = self.model.generate(  # pyright: ignore[reportCallIssue]
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the full generated text
        generated_text: str = self.tokenizer.decode(  # pyright: ignore[reportAssignmentType]
            output_ids[0], skip_special_tokens=True
        )

        # Return text and activations captured during generation
        return generated_text, self.activations.copy()

    def __repr__(self: "ModelWrapper") -> str:
        """Return string representation of the ModelWrapper.

        Returns:
            A string describing the model, layer, and device.
        """
        return (
            f"ModelWrapper(model_name='{self.model_name}', "
            f"layer_idx={self.layer_idx}, device={self.device})"
        )


# ============================================================================
# LAYER SELECTION GUIDANCE FOR 0.5B PARAMETER MODELS
# ============================================================================
# For Qwen-2.5-0.5B and similar 0.5B models:
#
# Architecture: Typically 8 transformer layers with ~128 hidden dim
#
# Layer Selection Recommendations:
# - Layer 0-1: Early layers capture low-level syntax/tokens
# - Layer 2-3: Early-middle layers (slightly entangled features)
# - Layer 4-5: MIDDLE LAYERS (RECOMMENDED for SAEs) ← Best interpretability
#   * Balance between early token-level and late semantic features
#   * Sufficient abstraction while maintaining interpretability
#   * Good feature separation for sparse decomposition
# - Layer 6-7: Late layers capture high-level semantic/context
#
# Specific Recommendation: Use layer_idx=4 (5th layer, 0-indexed)
# This layer captures rich, interpretable features for mechanistic analysis.
# ============================================================================
