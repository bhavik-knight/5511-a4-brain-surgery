"""Main entry point for the brain surgery project.

This module is intended for interactive development sessions, e.g.:

    uv run python -i -m brain_surgery.main

When run with `-i`, the Python process stays alive after initialization and
keeps model weights resident in GPU VRAM (useful for long sessions).
"""

# ruff: noqa: E402

import os

# Force offline mode for Transformers (requires models to be present in ./models).
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .model_wrapper import ModelWrapper
from .utils import DEFAULT_LAYER_IDX, DEFAULT_MODEL_NAME

wrapper: ModelWrapper | None = None


def main() -> None:
    """Initialize a default ModelWrapper for interactive use."""
    global wrapper
    if wrapper is None:
        wrapper = ModelWrapper(
            model_name=DEFAULT_MODEL_NAME,
            layer_idx=DEFAULT_LAYER_IDX,
        )
        print(
            "Initialized ModelWrapper as `wrapper`. "
            "Use wrapper.generate_with_activations(...) and "
            "wrapper.save_activations(...)."
        )
    else:
        print("ModelWrapper already initialized as `wrapper`.")


if __name__ == "__main__":
    main()
