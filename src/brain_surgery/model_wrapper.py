"""Model wrapper with forward hooks for capturing transformer activations.

This module provides utilities to wrap a pre-trained language model and register
forward hooks on specific layers to capture activation tensors during generation.
Activations are extracted from the residual stream of middle transformer layers,
enabling mechanistic interpretability analysis via Sparse Autoencoders.

Typical usage:
    >>> wrapper = ModelWrapper(model_name="./models/qwen2.5-0.5b", layer_idx=12)
    >>> text, activations = wrapper.generate_with_activations(
    ...     prompt="The capital of France is",
    ...     max_tokens=20
    ... )
    >>> print(f"Generated: {text}")
    >>> print(f"Activations shape: {activations.shape}")
"""

# ruff: noqa: ANN101

from pathlib import Path
from collections.abc import Sequence
from typing import Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.tokenization_utils_base import BatchEncoding

from .utils import (
    ACTIVATIONS_DIR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_LAYER_IDX,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    ROOT_DIR,
    ensure_dir_exists,
)

type ActivationPayloadValue = str | int | Tensor | list[str] | None


def get_default_device() -> torch.device:
    """Get appropriate device with priority: CUDA > CPU.

    Detects the best available device for computation on the current system,
    with fallback hierarchy: NVIDIA CUDA → CPU.

    Returns:
        Device object for torch operations (cuda or cpu).

    Example:
        >>> device = get_default_device()
        >>> print(f"Using device: {device}")
        Using device: cuda  # or cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class ModelWrapper:
    """Wraps a pre-trained language model with forward hooks for activation capture.

    This class loads a causal language model and registers a forward hook on a
    specified transformer layer's residual stream. During text generation, activations
    from that layer are captured and returned alongside the generated text.

    Attributes:
        model_name: Local model directory (e.g., "./models/qwen2.5-0.5b").
        layer_idx: Index of the transformer layer to hook (0-indexed).
        model: The loaded pre-trained language model.
        tokenizer: The tokenizer for the model.
        device: Device on which the model runs (cuda or cpu).
        activations: Dictionary storing captured activations by layer.
        hooks: List of registered hook handles for cleanup.

    Example:
        >>> wrapper = ModelWrapper("./models/qwen2.5-0.5b", layer_idx=12)
        >>> text, acts = wrapper.generate_with_activations("Hello", max_tokens=10)
        >>> print(acts.shape)  # (seq_len, hidden_dim)
    """

    def __init__(
        self,
        model_name: str,
        layer_idx: int | None = None,
        *,
        activation_device: torch.device | Literal["cpu", "model"] = "cpu",
    ) -> None:
        """Initialize the ModelWrapper with a pre-trained model.

        Args:
            model_name: Local directory containing a `transformers`-compatible
                model + tokenizer (downloaded ahead of time into `./models/`).
            layer_idx: Index of the transformer layer to capture activations from.
                If None, defaults to the midpoint layer after model load.
                For Qwen-2.5-0.5B (24 layers), recommended: 12 for middle layer.
                See get_recommended_layer_idx() for dynamic calculation.
            activation_device: Where to store captured activations.
                - "cpu" (default): move activations off-GPU immediately.
                - "model": keep activations on the same device as the model.
                - torch.device(...): store on an explicit device.

                Defaulting to CPU is safer for VRAM management (e.g., RTX 4070
                12GB) when collecting many prompts / long sequences.

        Raises:
            ValueError: If layer_idx is negative or out of range.
            OSError: If model cannot be downloaded or loaded.
        """
        self.model_name: str = model_name
        self.layer_idx: int | None = layer_idx
        self.device: torch.device = get_default_device()

        model_dir = Path(model_name)
        if not model_dir.is_absolute():
            model_dir = ROOT_DIR / model_dir
        if not model_dir.exists():
            raise FileNotFoundError(
                "Local model directory not found: "
                f"{model_dir}. Download the model into ./models first."
            )
        if not model_dir.is_dir():
            raise NotADirectoryError(f"Expected a model directory, got: {model_dir}")

        if activation_device == "cpu":
            self.activation_device: torch.device = torch.device("cpu")
        elif activation_device == "model":
            self.activation_device = self.device
        else:
            self.activation_device = activation_device

        # Load model and tokenizer from local disk (offline).
        # Use BFLOAT16 + device_map='auto' on CUDA for A100 VRAM efficiency.
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model: PreTrainedModel = cast(
            PreTrainedModel,
            AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                device_map="auto",
                dtype=dtype,
            ),
        )
        self.tokenizer: PreTrainedTokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True,
            ),
        )

        cast(nn.Module, self.model).eval()  # Set to evaluation mode

        # Storage for captured activations
        self.activations: dict[str, Tensor] = {}
        self._activation_steps: list[Tensor] = []
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Metadata for saving (populated by generate_with_activations)
        self._last_prompt: str | None = None
        self._last_generated_text: str | None = None
        self._last_output_ids: Tensor | None = None  # (batch, seq)
        self._last_token_texts: list[str] | None = None
        self._last_token_strs: list[str] | None = None

        # Register hooks on the target layer
        if self.layer_idx is None:
            self.layer_idx = DEFAULT_LAYER_IDX
            print(
                f"Defaulting to layer {self.layer_idx} (optimized for A100/Qwen-2.5-7B)"
            )

        if self.layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {self.layer_idx}")
        if self.layer_idx >= self.total_layers:
            raise ValueError(
                "layer_idx out of range: "
                f"{self.layer_idx} >= total_layers ({self.total_layers})"
            )

        self._register_hooks()

    def is_loaded(self) -> bool:
        """Check if model and tokenizer are initialized.

        Returns:
            True if both model and tokenizer are loaded, False otherwise.
        """
        return self.model is not None and self.tokenizer is not None

    @property
    def total_layers(self) -> int:
        """Get the total number of transformer layers in the model.

        Dynamically detects the layer count from the loaded model's config,
        supporting multiple architecture types (Qwen, GPT, LLaMA, etc.).

        Returns:
            Total number of transformer blocks in the model.

        Raises:
            RuntimeError: If model is not loaded (call is_loaded() first).
            RuntimeError: If unable to detect layer count from model config.

        Example:
            >>> wrapper = ModelWrapper(...)
            >>> total = wrapper.total_layers
            >>> print(f"Model has {total} layers")
            Model has 24 layers
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load model first.")

        # Try to get num_hidden_layers from config
        if hasattr(self.model, "config") and hasattr(
            self.model.config, "num_hidden_layers"
        ):
            num_hidden_layers = getattr(self.model.config, "num_hidden_layers")
            if isinstance(num_hidden_layers, int):
                return num_hidden_layers

        raise RuntimeError("Could not detect total layer count from model config")

    def _resolve_transformer_layers(self) -> Sequence[nn.Module]:
        """Resolve transformer blocks across supported architecture layouts."""
        model_root = getattr(self.model, "model", None)
        if model_root is not None:
            layers = getattr(model_root, "layers", None)
            if isinstance(layers, nn.ModuleList):
                return cast(Sequence[nn.Module], layers)

        transformer_root = getattr(self.model, "transformer", None)
        if transformer_root is not None:
            layers = getattr(transformer_root, "h", None)
            if isinstance(layers, nn.ModuleList):
                return cast(Sequence[nn.Module], layers)

        raise RuntimeError(
            "Unsupported model architecture: cannot locate transformer layers"
        )

    def _register_hooks(self) -> None:
        """Register forward hooks on the residual stream of the target layer.

        This method registers hooks on the specified transformer layer's output,
        capturing the residual activation before layer normalization. Hooks are
        stored for later cleanup via unregister_hooks().

        Note:
            Hook placement on residual stream (vs. MLP output) captures the full
            attention output + residual, providing richer signal for SAE training.

            By default we move captured activations to CPU immediately. This is a
            deliberate VRAM-management strategy: activation collection can be much
            larger than the model weights, and keeping large activation buffers on
            GPU can quickly exhaust VRAM during dataset-scale runs.

        Raises:
            RuntimeError: If layer index is out of range for the model.
        """
        # Access the model's transformer layers.
        # Different architectures expose layers under different attribute paths.
        layers = self._resolve_transformer_layers()
        if self.layer_idx is None:
            raise RuntimeError("layer_idx must be set before registering hooks")
        layer_idx = self.layer_idx

        if layer_idx >= len(layers):
            raise RuntimeError(
                f"layer_idx {layer_idx} exceeds number of layers ({len(layers)})"
            )

        target_layer: nn.Module = cast(nn.Module, layers[layer_idx])

        # Hook function to capture activations
        def hook_fn(
            _module: nn.Module,
            _input_data: tuple[Tensor, ...],
            output_data: Tensor | tuple[Tensor, ...],
        ) -> None:
            """Forward hook to capture layer output activations.

            Args:
                _module: The module being hooked (unused).
                _input_data: Input tensors to the module (unused).
                output_data: Output from the module (tensor or tuple of tensors).
            """
            # Handle different output formats
            if isinstance(output_data, tuple):
                # Some models return (hidden_states, ...)
                hidden_states = output_data[0]
            else:
                # Others return just the tensor
                hidden_states = output_data

            # Store activations on CPU immediately to preserve VRAM for A100.
            # During generation, the model is called multiple times (often with
            # seq_len=1 after the first step). We accumulate steps and stitch
            # them into a single (batch, seq_len, hidden_dim) tensor later.
            stored = hidden_states.detach().cpu()
            if stored.dim() == 2:
                stored = stored.unsqueeze(0)

            self._activation_steps.append(stored)

        # Register the hook
        hook_handle = target_layer.register_forward_hook(hook_fn)
        self.hooks.append(hook_handle)

    def _infer_model_input_device(self) -> torch.device:
        device_attr = getattr(self.model, "device", None)
        if isinstance(device_attr, torch.device):
            return device_attr
        if isinstance(device_attr, str):
            return torch.device(device_attr)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    def unregister_hooks(self) -> None:
        """Unregister all forward hooks to free memory and restore normal behavior.

        This should be called when activation capture is no longer needed.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self._activation_steps.clear()

    def generate_with_activations(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
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
        self._activation_steps.clear()
        self._last_prompt = prompt
        self._last_generated_text = None
        self._last_output_ids = None
        self._last_token_texts = None
        self._last_token_strs = None

        # Tokenize the prompt
        inputs = cast(
            BatchEncoding,
            self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ),
        )
        model_device = self._infer_model_input_device()
        input_ids = cast(Tensor, inputs["input_ids"]).to(model_device)
        attention_mask = cast(Tensor | None, inputs.get("attention_mask"))
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        # Ensure pad token is set (common for causal LMs)
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        # Generate tokens with activation capture
        generate_fn = getattr(self.model, "generate", None)
        if not callable(generate_fn):
            raise RuntimeError("Loaded model does not expose a callable generate().")

        with torch.no_grad():
            generate_out = generate_fn(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=pad_token_id,
                return_dict_in_generate=False,
            )

        output_ids: Tensor
        if isinstance(generate_out, torch.Tensor):
            output_ids = generate_out
        else:
            # Some configs return a Generate*Output; sequences holds the token ids.
            sequences = getattr(generate_out, "sequences", None)
            if not isinstance(sequences, torch.Tensor):
                raise RuntimeError(
                    "Model generate() output is missing tensor sequences"
                )
            output_ids = sequences

        # Combine activation steps into a single tensor
        if self._activation_steps:
            # One cat at the end avoids repeated reallocations.
            # Make contiguous so downstream slicing/saving doesn't end up with
            # unexpected striding.
            combined = torch.cat(self._activation_steps, dim=1).contiguous()
            self.activations["layer"] = combined

        # Decode the full generated text
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if isinstance(generated_text, list):
            # Defensive: some stubs model decode as str|list[str]
            generated_text = generated_text[0] if generated_text else ""

        # Store token metadata for saving
        self._last_generated_text = generated_text
        self._last_output_ids = output_ids.detach().cpu()
        token_ids = output_ids[0].detach().cpu().tolist()
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
        if isinstance(token_strs, str):
            token_strs = [token_strs]
        self._last_token_strs = list(token_strs)

        def _decode_single_token(tid: int) -> str:
            decoded = self.tokenizer.decode([tid], skip_special_tokens=False)
            if isinstance(decoded, list):
                return decoded[0] if decoded else ""
            return cast(str, decoded)

        self._last_token_texts = [_decode_single_token(tid) for tid in token_ids]

        # Return text and activations captured during generation
        return generated_text, self.activations.copy()

    def save_activations(
        self,
        *,
        save_dir: Path | None = None,
        batch_idx: int = 0,
        file_stem: str | None = None,
        fmt: Literal["pt"] = "pt",
        device: torch.device | Literal["cpu", "keep"] = "cpu",
        gitignore_if_large: bool = True,
        max_mb: int = 50,
        gitignore_path: Path | None = None,
        gitignore_mode: Literal["file", "folder"] = "file",
    ) -> Path:
        """Save token-aligned activations + prompt text to disk.

        This writes a single artifact containing:
        - prompt and generated_text
        - token ids and token text snippets
        - activation tensor aligned per token

        The artifact is designed so other code can load it and iterate over
        tokens and their corresponding activation vectors.

        Args:
            save_dir: Directory to save into. Defaults to data/activations/.
            batch_idx: Batch index used in the filename.
            file_stem: Optional explicit stem (no extension). If omitted, a
                descriptive name is generated.
            fmt: Save format. Currently supports "pt".
            device: Device to move activations onto for serialization.
                - "cpu" (default): saves CPU tensors (portable, VRAM-safe).
                - "keep": saves activations on their current device.
                - torch.device(...): move activations to a specific device.

                Even if you capture activations on GPU for speed, saving them on
                CPU is usually preferred to avoid requiring CUDA when loading.
            gitignore_if_large: If True, append this file (or its folder) to
                .gitignore when size exceeds max_mb.
            max_mb: Size threshold for gitignore automation.
            gitignore_path: Path to .gitignore; defaults to project root.
            gitignore_mode: "file" to ignore the specific artifact path, or
                "folder" to ignore the entire activations directory.

        Returns:
            Path to the saved artifact.

        Raises:
            RuntimeError: If no generation has been run or activations are missing.
            ValueError: If token and activation lengths cannot be aligned.
        """
        if fmt != "pt":
            raise ValueError(f"Unsupported fmt: {fmt}")

        if not self.activations or "layer" not in self.activations:
            raise RuntimeError(
                "No activations available. Call generate_with_activations() first."
            )
        if (
            self._last_prompt is None
            or self._last_generated_text is None
            or self._last_output_ids is None
            or self._last_token_texts is None
        ):
            raise RuntimeError(
                "No generation metadata available. "
                "Call generate_with_activations() first."
            )

        save_dir = ensure_dir_exists(save_dir or ACTIVATIONS_DIR)
        gitignore_path = gitignore_path or (ROOT_DIR / ".gitignore")

        layer_name = f"layers_{self.layer_idx}_acts_batch_{batch_idx}"
        stem = file_stem or layer_name
        out_path = save_dir / f"{stem}.{fmt}"
        if out_path.exists():
            version = 1
            while True:
                candidate = save_dir / f"{stem}_v{version}.{fmt}"
                if not candidate.exists():
                    out_path = candidate
                    break
                version += 1

        acts = self.activations["layer"]
        if acts.dim() == 3:
            if acts.shape[0] != 1:
                raise ValueError(f"Expected batch=1 activations, got {acts.shape}")
            acts_2d = acts[0]
        elif acts.dim() == 2:
            acts_2d = acts
        else:
            raise ValueError(
                f"Unexpected activations tensor shape: {tuple(acts.shape)}"
            )

        token_texts = self._last_token_texts
        token_strs = self._last_token_strs or []
        token_ids_1d = self._last_output_ids[0].to(torch.int64)

        # Align lengths if needed (best-effort), but require at least 1 element.
        seq_len = min(len(token_texts), acts_2d.shape[0])
        if seq_len <= 0:
            raise ValueError("No tokens/activations to save")
        if len(token_texts) != acts_2d.shape[0]:
            token_texts = token_texts[:seq_len]
            token_strs = token_strs[:seq_len]
            token_ids_1d = token_ids_1d[:seq_len]
            acts_2d = acts_2d[:seq_len]

        payload: dict[str, ActivationPayloadValue] = {
            "model_name": self.model_name,
            "layer_idx": self.layer_idx,
            "device": str(self.device),
            "activation_device": str(self.activation_device),
            "prompt": self._last_prompt,
            "generated_text": self._last_generated_text,
            "token_ids": token_ids_1d,
            "token_texts": token_texts,
            "token_strs": token_strs,
            "activations": acts_2d,
        }

        if device != "keep":
            target_device = torch.device("cpu") if device == "cpu" else device
            payload["token_ids"] = cast(Tensor, payload["token_ids"]).to(target_device)
            payload["activations"] = cast(Tensor, payload["activations"]).to(
                target_device
            )

        payload["activations"] = cast(Tensor, payload["activations"]).contiguous()

        torch.save(payload, out_path)

        if gitignore_if_large:
            self._gitignore_large_artifact(
                out_path,
                gitignore_path=gitignore_path,
                max_mb=max_mb,
                mode=gitignore_mode,
            )

        return out_path

    @staticmethod
    def _gitignore_large_artifact(
        artifact_path: Path,
        *,
        gitignore_path: Path,
        max_mb: int,
        mode: Literal["file", "folder"],
    ) -> None:
        threshold_bytes = max_mb * 1024 * 1024
        try:
            size_bytes = artifact_path.stat().st_size
        except OSError:
            return
        if size_bytes <= threshold_bytes:
            return

        if mode == "folder":
            pattern = "data/activations/"
        else:
            try:
                pattern = str(artifact_path.relative_to(ROOT_DIR)).replace("\\", "/")
            except ValueError:
                pattern = str(artifact_path).replace("\\", "/")

        existing = ""
        if gitignore_path.exists():
            existing = gitignore_path.read_text(encoding="utf-8")
            if pattern in existing.splitlines():
                return

        with gitignore_path.open("a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(f"{pattern}\n")

    def __repr__(self) -> str:
        """Return string representation of the ModelWrapper.

        Returns:
            A string describing the model, layer, and device.
        """
        return (
            f"ModelWrapper(model_name='{self.model_name}', "
            f"layer_idx={self.layer_idx}, device={self.device})"
        )


# ============================================================================
# LAYER SELECTION GUIDANCE FOR TRANSFORMER MODELS
# ============================================================================
# For Qwen-2.5-0.5B (24 transformer layers, 896 hidden dim):
#
# Layer Selection Recommendations:
# - Layer 0-5: Early layers capture low-level syntax/tokens
# - Layer 6-11: Early-middle layers (slightly entangled features)
# - Layer 12-17: MIDDLE LAYERS (RECOMMENDED for SAEs) ← Best interpretability
#   * Balance between early token-level and late semantic features
#   * Sufficient abstraction while maintaining interpretability
#   * Good feature separation for sparse decomposition
#   * Dynamic calculation: use get_recommended_layer_idx(24) = 12
# - Layer 18-23: Late layers capture high-level semantic/context
#
# Specific Recommendation: Use layer_idx=12 (0-indexed)
# This layer captures rich, interpretable features for mechanistic analysis.
# Or use: layer_idx=get_recommended_layer_idx(model.config.num_hidden_layers)
# ============================================================================


def main() -> None:
    """Smoke-test the ModelWrapper.

    Loads the default model, captures activations from the default layer, and
    prints the generated text plus the captured activation tensor shape.
    """
    wrapper: ModelWrapper | None = None
    try:
        wrapper = ModelWrapper(
            model_name=DEFAULT_MODEL_NAME,
            layer_idx=DEFAULT_LAYER_IDX,
        )
        prompt = "The capital of France is"

        print("forward pass with activation capture...")
        print(f"Prompt: {prompt}\n")

        text, activations = wrapper.generate_with_activations(
            prompt=prompt,
            max_tokens=20,
        )
        print(f"Generated text: {text}")

        layer_acts = activations.get("layer")
        if layer_acts is None:
            print("No activations captured (missing key 'layer').")
        else:
            print(
                f"Captured activations: shape={tuple(layer_acts.shape)}, "
                f"dtype={layer_acts.dtype}"
            )

        saved_path = wrapper.save_activations(batch_idx=0)
        print(f"Saved activations to: {saved_path}")
    finally:
        if wrapper is not None:
            wrapper.unregister_hooks()


if __name__ == "__main__":
    main()
