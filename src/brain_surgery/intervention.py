"""Feature clamping and intervention for counterfactual experiments.

This module implements Q6: using activation clamping to test the causal
importance of discovered SAE features. The SAEIntervention class allows
parameterized clamping of any discovered feature to measure its causal effect
on model generation.

Example workflow:
    1. Train SAE on activations
    2. Discover interesting features via SAEInterpreter
    3. Use SAEIntervention to clamp specific features during generation
    4. Compare baseline vs. clamped outputs to measure causal importance

Example experiments (if features exist in trained SAE):
    - Feature 1625: Footballer Identity (Ronaldo/Messi focus)
    - Feature 1134: Superstar Identity (hallucination effects)

Typical usage:
    >>> from brain_surgery.model_wrapper import ModelWrapper
    >>> from brain_surgery.intervention import SAEIntervention
    >>> import torch
    >>>
    >>> # Load model and SAE
    >>> wrapper = ModelWrapper(model_name="./models/qwen2.5-0.5b", layer_idx=12)
    >>> sae = torch.load("models/sae.pt")  # Pre-trained SAE instance
    >>>
    >>> # Create intervention
    >>> interventioner = SAEIntervention(wrapper, sae)
    >>> activations = torch.load("data/activations/activations.pt")
    >>> interventioner.compute_feature_max_values(activations)
    >>>
    >>> # Clamp a feature during generation
    >>> result = interventioner.generate_with_clamped_feature(
    ...     prompt="What is the Champions League?",
    ...     feature_index=1625,
    ...     clamp_multiplier=8.0
    ... )
    >>> print(f"Clamped output: {result['generated_text']}")
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .model_wrapper import ModelWrapper


class SAEIntervention:
    """Feature clamping for counterfactual intervention experiments.

    Allows testing causal effects of SAE latent features by fixing them to
    specified values during text generation. Hook-based approach intercepts
    activations mid-forward pass, encodes to latent space, clamps specified
    features, decodes back, and continues generation with modified activations.

    This is a parameterized implementation allowing clamping of ANY feature index,
    not just pre-defined ones. Users discover interesting features via
    SAEInterpreter.rank_features_by_max_activation() first.
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        sae: Any,  # noqa: ANN401
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the intervention system.

        Args:
            model_wrapper: Loaded ModelWrapper with registered hooks.
            sae: Trained sparse autoencoder instance with encode() and decode()
                methods. Expected interface: sae.encode(tensor) → latents,
                sae.decode(latents) → activations, sae.latent_dim property.
            device: Device for intervention computations (default: "cpu").
                CPU is recommended for VRAM safety (RTX 4070 compatible).

        Raises:
            ValueError: If model_wrapper or sae is None.
        """
        if model_wrapper is None or not model_wrapper.is_loaded():
            raise ValueError("model_wrapper must be loaded (call ModelWrapper first)")
        if sae is None:
            raise ValueError("sae must not be None")

        self.model_wrapper: ModelWrapper = model_wrapper
        self.sae: Any = sae  # noqa: ANN401
        self.device: torch.device | str = device

        # Feature clamping state
        self.feature_index: int | None = None
        self.clamp_multiplier: float | None = None
        self.feature_max_values: Tensor | None = None

        # Activation storage for diagnostics
        self.original_activations: Tensor | None = None
        self.modified_activations: Tensor | None = None

        # Hook management
        self.hook_handle: Any = None
        self.hook_layer_index: int | None = None
        self.hook_layer_name: str | None = None

    def compute_feature_max_values(self, activation_matrix: Tensor) -> None:
        """Encode activations and find maximum per latent feature.

        Encodes all token activations through the SAE and computes the
        maximum activation value for each latent dimension. These max values
        are used in clamping: clamped_value = clamp_multiplier × max_value.

        Args:
            activation_matrix: Activation tensor of shape (num_tokens, hidden_dim).
                Typically loaded from data/activations/activations.pt.

        Raises:
            RuntimeError: If SAE is not in eval mode or device mismatch.

        Example:
            >>> # For Qwen2.5-0.5B (24 layers)
            >>> recommended = get_recommended_layer_idx(24)
            >>> print(f"Feature max values shape: {recommended_feature_shape}")
            Feature max values shape: torch.Size([1792])
        """
        if self.sae is None:
            raise RuntimeError("SAE not initialized")

        with torch.no_grad():
            # Encode all activations to latent space
            if isinstance(self.device, str):
                device = torch.device(self.device)
            else:
                device = self.device

            latents = self.sae.encode(activation_matrix.to(device))
            # Move back to CPU for storage (VRAM safety)
            latents = latents.cpu()

        # Compute max per feature dimension
        self.feature_max_values = torch.max(latents, dim=0).values

    def remove_hook(self) -> None:
        """Remove the currently registered intervention hook.

        Cleans up the forward hook if one is registered. Safe to call
        even if no hook is currently active.
        """
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def register_prompt_clamp_hook(
        self,
        feature_index: int,
        clamp_multiplier: float,
    ) -> None:
        """Register forward hook to clamp a specific latent feature.

        Sets up a hook on the middle transformer layer that will intercept
        activations, encode to latent space, clamp the specified feature to
        (multiplier × max_value), decode, and return modified activations.

        Args:
            feature_index: Index of the latent feature to clamp [0, latent_dim).
            clamp_multiplier: Multiplier for the feature's max activation
                (e.g., 8.0 for 8x amplification, 0.0 for suppression).

        Raises:
            RuntimeError: If SAE not loaded, max values not computed, or hook
                registration fails.
            IndexError: If feature_index is out of bounds for latent dimension.

        Example:
            >>> interventioner.register_prompt_clamp_hook(
            ...     feature_index=1625,
            ...     clamp_multiplier=8.0
            ... )
        """
        if self.sae is None:
            raise RuntimeError("SAE must be loaded before registering hook")
        if self.feature_max_values is None:
            raise RuntimeError(
                "Feature max values not computed. "
                "Call compute_feature_max_values(...) first"
            )
        if not self.model_wrapper.is_loaded():
            raise RuntimeError("ModelWrapper not loaded")

        # Validate feature index
        latent_dim = self.sae.latent_dim
        if feature_index < 0 or feature_index >= latent_dim:
            raise IndexError(
                f"feature_index {feature_index} out of range [0, {latent_dim})"
            )

        # Remove any existing hook
        self.remove_hook()

        # Get transformer blocks and find middle layer
        try:
            # Access the model's layer structure (handles Qwen architecture)
            model_any: Any = self.model_wrapper.model
            if hasattr(model_any, "model") and hasattr(model_any.model, "layers"):
                blocks = model_any.model.layers
            else:
                raise RuntimeError("Could not detect transformer layers in model")
        except Exception as e:
            raise RuntimeError(f"Failed to get transformer blocks: {e}") from e

        middle_index = len(blocks) // 2
        target_block = blocks[middle_index]

        # Store clamping configuration
        self.feature_index = feature_index
        self.clamp_multiplier = clamp_multiplier
        self.hook_layer_index = middle_index
        self.hook_layer_name = f"model.model.layers[{middle_index}]"

        # Reset activation storage
        self.original_activations = None
        self.modified_activations = None

        # Define hook function
        def hook_fn(  # noqa: ANN001,ANN002,ANN003,ANN201
            module: Any,  # noqa: ANN401
            inputs: Any,  # noqa: ANN401
            output: Any,  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            """Intercept activations, clamp feature, decode, return modified."""
            # Ensure feature_max_values is set (verified in register_prompt_clamp_hook)
            assert self.feature_max_values is not None  # Safety check

            # Extract hidden states from output (handles tuple vs tensor)
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Move to working device
            if isinstance(self.device, str):
                device = torch.device(self.device)
            else:
                device = self.device
            hidden_states = hidden_states.to(device)

            # Store original for diagnostics
            self.original_activations = hidden_states.detach().cpu()

            # Flatten for SAE encoding
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat_hidden = hidden_states.reshape(-1, hidden_dim)

            # Encode, clamp, decode
            with torch.no_grad():
                latent = self.sae.encode(flat_hidden)

                # Clamp the feature
                max_val = self.feature_max_values[self.feature_index].item()
                clamped_value = self.clamp_multiplier * max_val
                latent[:, self.feature_index] = clamped_value

                # Decode back to activation space
                modified_flat_hidden = self.sae.decode(latent)

            # Reshape back to sequence format
            modified_hidden_states = modified_flat_hidden.reshape(
                batch_size, seq_len, hidden_dim
            )

            # Store modified for diagnostics
            self.modified_activations = modified_hidden_states.detach().cpu()

            # Return with proper structure
            if rest is not None:
                return (modified_hidden_states, *rest)
            return modified_hidden_states

        # Register hook on target layer
        self.hook_handle = target_block.register_forward_hook(hook_fn)

    def generate_with_clamped_feature(
        self,
        prompt: str,
        feature_index: int,
        clamp_multiplier: float,
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> dict[str, Any]:
        """Generate text with a specific feature clamped during forward pass.

        Registers a hook that intercepts activations mid-forward, clamps the
        specified feature, then continues generation. The hook is removed
        after generation completes.

        Args:
            prompt: Input text prompt for generation.
            feature_index: SAE latent feature to clamp [0, latent_dim).
            clamp_multiplier: Multiplier for feature's max activation value.
                Typical range: 0.0-10.0 (0.0 suppresses, >1.0 amplifies).
            max_tokens: Max new tokens to generate (default: 50).
            temperature: Sampling temperature (default: 0.7).
                Lower values = more deterministic, higher = more random.
            top_p: Nucleus sampling threshold (default: 0.95).

        Returns:
            Dictionary with diagnostics:
                - "prompt": Original input prompt
                - "feature_index": Which feature was clamped
                - "clamp_multiplier": Multiplier used
                - "feature_max_value": Max activation observed in dataset
                - "effective_clamped_value": multiplier × max_value (actual clamp)
                - "hook_layer_index": Which layer was hooked (middle)
                - "hook_layer_name": Layer name string
                - "generated_text": Full output (prompt + generation)
                - "generated_continuation_text": Generation only (no prompt)
                - "original_activations": Unmodified activations (diagnostics)
                - "modified_activations": Clamp-modified activations (diagnostics)

        Raises:
            RuntimeError: If SAE not loaded or max values not computed.
            IndexError: If feature_index out of bounds.

        Example:
            >>> # Test if Feature 1625 exists and produces effects
            >>> if 1625 < interventioner.feature_max_values.shape[0]:
            ...     result = interventioner.generate_with_clamped_feature(
            ...         prompt="What is the Champions League?",
            ...         feature_index=1625,
            ...         clamp_multiplier=8.0
            ...     )
            ...     print(f"Generated: {result['generated_text']}")
        """
        if self.sae is None:
            raise RuntimeError("Call __init__ with SAE before generating")
        if self.feature_max_values is None:
            raise RuntimeError(
                "Feature max values not computed. "
                "Call compute_feature_max_values(...) first"
            )

        # Register the clamping hook
        self.register_prompt_clamp_hook(
            feature_index=feature_index,
            clamp_multiplier=clamp_multiplier,
        )

        try:
            # Tokenize prompt
            tokenizer = self.model_wrapper.tokenizer
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )

            # Move to model device
            model_device = self.model_wrapper.device
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

            # Set up generation parameters
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "pad_token_id": tokenizer.pad_token_id,
            }

            # Generate with hook active
            model = self.model_wrapper.model
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            # Decode full output and continuation
            generated_text = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )

            prompt_length = inputs["input_ids"].shape[1]
            new_token_ids = generated_ids[:, prompt_length:]
            generated_continuation_text = tokenizer.decode(
                new_token_ids[0],
                skip_special_tokens=True,
            )

            # Compute effective clamped value
            feature_max_value = self.feature_max_values[feature_index].item()
            effective_clamped_value = clamp_multiplier * feature_max_value

            return {
                "prompt": prompt,
                "feature_index": feature_index,
                "clamp_multiplier": clamp_multiplier,
                "feature_max_value": feature_max_value,
                "effective_clamped_value": effective_clamped_value,
                "hook_layer_index": self.hook_layer_index,
                "hook_layer_name": self.hook_layer_name,
                "generated_text": generated_text,
                "generated_continuation_text": generated_continuation_text,
                "input_ids": inputs["input_ids"].detach().cpu(),
                "generated_ids": generated_ids.detach().cpu(),
                "original_activations": self.original_activations,
                "modified_activations": self.modified_activations,
            }

        finally:
            # Always remove hook, even if generation fails
            self.remove_hook()
