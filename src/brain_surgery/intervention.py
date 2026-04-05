"""Feature clamping for counterfactual interventions.

Implements feature intervention by clamping specific latent features
to test causal effects on model behavior.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .model_wrapper import ModelWrapper
from .sae import SparseAutoencoder

type InterventionValue = str | int | float | Tensor | None
type InterventionResult = dict[str, InterventionValue]


class SAEIntervention:
    """Feature clamping for counterfactual feature intervention experiments.

    Allows testing causal effects by fixing specific SAE latent features
    to their maximum- or other specified values during generation.

    Two example experiments documented:
            1. Feature 1625: Champions League → Footballer identity hallucination
            2. Feature 1134: Arsenal FC → Messi hallucination (never played there!)
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        *,
        sae: SparseAutoencoder | None = None,
        checkpoint_path: Path | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the intervention system.

        Args:
                model_wrapper: ModelWrapper instance providing model and tokenizer.
                sae: Optional SAE instance. If omitted, provide checkpoint_path.
                checkpoint_path: Optional SAE checkpoint path to load.
                device: Device to run SAE on. Defaults to "cpu".

        Raises:
                ValueError: If model_wrapper is not loaded or SAE config is invalid.
        """
        if not model_wrapper.is_loaded():
            raise ValueError("ModelWrapper must be loaded before intervention setup.")
        if sae is None and checkpoint_path is None:
            raise ValueError(
                "Provide sae or checkpoint_path to initialize SAEIntervention."
            )

        self.model_wrapper = model_wrapper
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.sae: SparseAutoencoder | None = sae
        self.feature_index: int | None = None
        self.clamp_multiplier: float | None = None
        self.feature_max_values: Tensor | None = None

        self.original_activations: Tensor | None = None
        self.modified_activations: Tensor | None = None

        self.hook_handle: RemovableHandle | None = None
        self.hook_layer_index: int | None = None
        self.hook_layer_name: str | None = None

        if self.sae is None:
            self.load_sae()

    def load_sae(self) -> None:
        """Load SAE model from checkpoint file."""
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided to load SAE")

        checkpoint_file = Path(self.checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        input_dim = checkpoint["input_dim"]
        latent_dim = checkpoint["latent_dim"]

        self.sae = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        self.sae.load_state_dict(checkpoint["model_state_dict"])
        self.sae.to(self.device)
        self.sae.eval()

    def compute_feature_max_values(self, activation_matrix: Tensor) -> None:
        """Compute maximum activation for each latent feature.

        Args:
                activation_matrix: Activation tensor of shape (num_tokens, hidden_dim).

        Raises:
                RuntimeError: If SAE is not loaded.
        """
        if self.sae is None:
            raise RuntimeError("Load the SAE before computing feature max values.")

        with torch.no_grad():
            latents = self.sae.encode(activation_matrix.to(self.device)).cpu()

        self.feature_max_values = torch.max(latents, dim=0).values

    def _get_transformer_blocks(self) -> Sequence[nn.Module]:
        """Get transformer block modules from the wrapped model.

        Returns:
                List of transformer block modules.
        """
        model_root = getattr(self.model_wrapper.model, "model", None)
        if model_root is not None:
            layers = getattr(model_root, "layers", None)
            if isinstance(layers, nn.ModuleList):
                return cast(Sequence[nn.Module], layers)
        raise RuntimeError("Could not detect transformer layers in model")

    def remove_hook(self) -> None:
        """Remove the currently registered intervention hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def register_prompt_clamp_hook(
        self,
        feature_index: int,
        clamp_multiplier: float,
    ) -> None:
        """Register forward hook to clamp a specific latent feature.

        Args:
                feature_index: Index of the feature to clamp.
                clamp_multiplier: Multiplier for feature's max value (e.g., 8.0x).

        Raises:
                RuntimeError: If SAE not loaded or max values not computed.
                IndexError: If feature_index is out of bounds.
        """
        if self.sae is None:
            raise RuntimeError("SAE must be loaded before registering the hook.")
        if self.feature_max_values is None:
            raise RuntimeError(
                "Feature max values have not been computed. "
                "Call compute_feature_max_values(...) first."
            )

        latent_dim = self.sae.latent_dim
        if feature_index < 0 or feature_index >= latent_dim:
            raise IndexError(
                f"feature_index {feature_index} out of range [0, {latent_dim})"
            )

        self.remove_hook()

        # Persist current intervention config before creating local captures.
        self.feature_index = feature_index
        self.clamp_multiplier = clamp_multiplier

        model_device = self.model_wrapper.device
        self.sae.to(model_device)
        sae = self.sae
        feature_max_values = self.feature_max_values
        feature_index_local = self.feature_index
        clamp_multiplier_local = self.clamp_multiplier

        assert sae is not None
        assert feature_max_values is not None
        assert feature_index_local is not None
        assert clamp_multiplier_local is not None

        blocks = self._get_transformer_blocks()
        middle_index = len(blocks) // 2
        target_block = blocks[middle_index]

        self.hook_layer_index = middle_index
        self.hook_layer_name = f"model.model.layers[{middle_index}]"

        self.original_activations = None
        self.modified_activations = None

        def hook_fn(
            _module: nn.Module,
            _inputs: tuple[Tensor, ...],
            output: Tensor | tuple[Tensor, ...],
        ) -> Tensor | tuple[Tensor, ...]:
            """Intercept, clamp, decode, and return modified activations."""
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            hidden_dtype: torch.dtype = hidden_states.dtype
            hidden_states = hidden_states.to(model_device)

            self.original_activations = hidden_states.detach().cpu()

            batch_size, seq_len, hidden_dim = hidden_states.shape
            if hidden_dim != sae.input_dim:
                raise RuntimeError(
                    "Activation width and SAE input_dim mismatch: "
                    f"hidden_dim={hidden_dim}, sae.input_dim={sae.input_dim}."
                )
            flat_hidden = hidden_states.reshape(-1, hidden_dim)

            with torch.no_grad():
                latent = sae.encode(flat_hidden)
                max_val = feature_max_values[feature_index_local].item()
                clamped_value = clamp_multiplier_local * max_val
                latent[:, feature_index_local] = torch.as_tensor(
                    clamped_value,
                    device=latent.device,
                    dtype=latent.dtype,
                )
                modified_flat_hidden = sae.decode(latent)

            modified_hidden_states = modified_flat_hidden.reshape(
                batch_size, seq_len, hidden_dim
            )
            modified_hidden_states = modified_hidden_states.to(hidden_dtype)

            self.modified_activations = modified_hidden_states.detach().cpu()

            if rest is not None:
                return (modified_hidden_states, *rest)
            return modified_hidden_states

        self.hook_handle = target_block.register_forward_hook(hook_fn)

    def generate_with_clamped_feature(
        self,
        prompt: str,
        feature_index: int,
        clamp_multiplier: float,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> InterventionResult:
        """Generate text with a specific feature clamped.

        Args:
                prompt: Input text prompt.
                feature_index: Latent feature to clamp.
                clamp_multiplier: Multiplier for feature's max value (e.g., 8.0, 4.0).
                max_new_tokens: Max tokens to generate. Defaults to 50.
                temperature: Sampling temperature. Defaults to 1.0.
                top_p: Nucleus sampling threshold. Defaults to 0.95.

        Returns:
                Dictionary with generated text and intervention diagnostics:
                        - "generated_text": Full generated text
                        - "generated_continuation_text": Continuation only
                        - "feature_index": Which feature was clamped
                        - "feature_max_value": Observed max activation
                        - "effective_clamped_value": Multiplied clamped value
                        - "original_activations": Before clamping
                        - "modified_activations": After clamping

        Raises:
                RuntimeError: If SAE not loaded or max values not computed.

        Example:
                >>> result = intervention.generate_with_clamped_feature(
                ...     prompt="What is the Champions League?",
                ...     feature_index=1625,
                ...     clamp_multiplier=8.0,
                ...     max_new_tokens=60,
                ... )
                >>> print(result["generated_text"])
        """
        if self.sae is None:
            raise RuntimeError(
                "Call load_sae() before generate_with_clamped_feature()."
            )
        if self.feature_max_values is None:
            raise RuntimeError(
                "Feature max values have not been computed. "
                "Call compute_feature_max_values(...) first."
            )

        self.register_prompt_clamp_hook(
            feature_index=feature_index,
            clamp_multiplier=clamp_multiplier,
        )

        inputs = self.model_wrapper.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        inputs = {
            key: value.to(self.model_wrapper.device) for key, value in inputs.items()
        }

        generation_kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": self.model_wrapper.tokenizer.pad_token_id,
        }

        with torch.no_grad():
            generate_fn = getattr(self.model_wrapper.model, "generate")
            generated_ids = cast(Tensor, generate_fn(**inputs, **generation_kwargs))

        self.remove_hook()

        generated_text = str(
            self.model_wrapper.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )
        )

        prompt_length = inputs["input_ids"].shape[1]
        new_token_ids = generated_ids[:, prompt_length:]
        generated_continuation_text = str(
            self.model_wrapper.tokenizer.decode(
                new_token_ids[0],
                skip_special_tokens=True,
            )
        )

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
            "input_ids": inputs["input_ids"].detach().cpu(),
            "generated_ids": generated_ids.detach().cpu(),
            "generated_text": generated_text,
            "generated_continuation_text": generated_continuation_text,
            "original_activations": self.original_activations,
            "modified_activations": self.modified_activations,
        }

    def compare_next_token_logprobs(
        self,
        prompt: str,
        candidate_tokens: list[str],
        *,
        feature_index: int | None = None,
        clamp_multiplier: float = 0.0,
    ) -> dict[str, float]:
        """Compare next-token log-probabilities with optional feature clamp.

        Args:
                prompt: Input prompt to score.
                candidate_tokens: Candidate token strings to score.
                feature_index: Optional feature index to clamp.
                clamp_multiplier: Clamp multiplier applied when feature_index is set.

        Returns:
                Mapping from token string to log-probability.

        Raises:
                RuntimeError: If SAE not loaded or max values not computed
                    when clamping.
        """
        if feature_index is not None:
            if self.sae is None or self.feature_max_values is None:
                raise RuntimeError(
                    "Compute feature max values before clamped log-prob scoring."
                )
            self.register_prompt_clamp_hook(
                feature_index=feature_index,
                clamp_multiplier=clamp_multiplier,
            )

        inputs = self.model_wrapper.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        inputs = {
            key: value.to(self.model_wrapper.device) for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model_wrapper.model(
                **inputs,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)[0]

        result: dict[str, float] = {}
        for token in candidate_tokens:
            tokenized = self.model_wrapper.tokenizer(token, add_special_tokens=False)
            ids_obj = tokenized["input_ids"]
            if not isinstance(ids_obj, list):
                continue
            # Next-token scoring only supports candidates that map to exactly one token.
            if len(ids_obj) != 1 or not isinstance(ids_obj[0], int):
                continue
            result[token] = float(log_probs[ids_obj[0]].item())

        if feature_index is not None:
            self.remove_hook()

        return result
