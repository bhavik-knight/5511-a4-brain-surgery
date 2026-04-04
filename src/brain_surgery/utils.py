"""Common configuration and utilities for modular code organization.

This module centralizes project configuration including directory paths,
device management, and helper functions. Using a single configuration source
enables easy switching of models, datasets, and output directories without
modifying core code.

Typical usage:
    >>> from brain_surgery.utils import (
    ...     ROOT_DIR, DATA_DIR, get_device, get_recommended_layer_idx
    ... )
    >>> print(f"Project root: {ROOT_DIR}")
    >>> print(f"Data directory: {DATA_DIR}")
    >>> device = get_device()
    >>> print(f"Using device: {device}")
    >>> # For Qwen2.5-0.5B (24 layers), get recommended middle layer
    >>> recommended = get_recommended_layer_idx(24)
    >>> print(f"Recommended layer: {recommended}")  # Output: 12
"""

from datetime import datetime
from pathlib import Path

import torch


# ============================================================================
# PROJECT DIRECTORY CONFIGURATION
# ============================================================================

# Root directory of the project (a4-brain-surgery/)
ROOT_DIR: Path = Path(__file__).parent.parent.parent

# Data directory for corpus and activations
DATA_DIR: Path = ROOT_DIR / "data"

# Results directory for experiments, features, and metrics
RESULTS_DIR: Path = ROOT_DIR / "results"

# Model cache directory (for downloaded Hugging Face models)
MODELS_DIR: Path = ROOT_DIR / "models"

# Subdirectories within data/
CORPUS_DIR: Path = DATA_DIR / "corpus"
ACTIVATIONS_DIR: Path = DATA_DIR / "activations"

# Subdirectories within results/
FEATURES_DIR: Path = RESULTS_DIR / "features"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
EXPERIMENTS_DIR: Path = RESULTS_DIR / "experiments"
CHECKPOINTS_DIR: Path = RESULTS_DIR / "checkpoints"
CLUSTERS_DIR: Path = RESULTS_DIR / "clusters"
INTERVENTIONS_DIR: Path = RESULTS_DIR / "interventions"
PLOTS_DIR: Path = RESULTS_DIR / "plots"

# Create all necessary directories on module import
_DIRS_TO_CREATE = [
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    CORPUS_DIR,
    ACTIVATIONS_DIR,
    FEATURES_DIR,
    METRICS_DIR,
    EXPERIMENTS_DIR,
    CHECKPOINTS_DIR,
    CLUSTERS_DIR,
    INTERVENTIONS_DIR,
    PLOTS_DIR,
]

type RunDirs = dict[str, Path]


def ensure_dir_exists(directory: Path) -> Path:
    """Create a directory and all parent directories if they don't exist.

    Args:
        directory: Path to the directory to create.

    Returns:
        The Path object for the (now existing) directory.

    Example:
        >>> custom_dir = ensure_dir_exists(ROOT_DIR / "custom_outputs")
        >>> assert custom_dir.exists()
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# Initialize all project directories
for _dir in _DIRS_TO_CREATE:
    ensure_dir_exists(_dir)


def generate_run_id() -> str:
    """Generate a timestamped run identifier.

    Returns:
        Run id formatted as ``run_YYYYMMDD_HHMM``.
    """
    return datetime.now().strftime("run_%Y%m%d_%H%M")


def create_run_output_dirs(run_id: str) -> RunDirs:
    """Create a run-scoped output hierarchy under results/.

    Args:
        run_id: Unique run identifier (for example ``run_20260403_2030``).

    Returns:
        Dictionary of created paths with keys:
            - ``root`` / ``experiment_root``
            - ``features_root`` / ``features_run``
            - ``metrics_root`` / ``metrics_run``
            - ``checkpoints``
            - ``interventions``
            - ``plots``
            - ``logs``
    """
    experiment_root = ensure_dir_exists(EXPERIMENTS_DIR / run_id)
    features_root = ensure_dir_exists(FEATURES_DIR)
    metrics_root = ensure_dir_exists(METRICS_DIR)
    features_run = ensure_dir_exists(FEATURES_DIR / run_id)
    metrics_run = ensure_dir_exists(METRICS_DIR / run_id)

    return {
        "root": experiment_root,
        "experiment_root": experiment_root,
        "features_root": features_root,
        "metrics_root": metrics_root,
        "features_run": features_run,
        "metrics_run": metrics_run,
        "checkpoints": ensure_dir_exists(experiment_root / "checkpoints"),
        "interventions": ensure_dir_exists(experiment_root / "interventions"),
        "plots": ensure_dir_exists(experiment_root / "plots"),
        "logs": ensure_dir_exists(experiment_root / "logs"),
    }


def get_device() -> torch.device:
    """Get the appropriate PyTorch device (cuda or cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_name() -> str:
    """Get the name of the current device as a string ("cuda" or "cpu")."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_recommended_layer_idx(num_layers: int) -> int:
    """Get recommended middle layer index for SAE feature extraction.

    For optimal SAE feature extraction, use the middle transformer layer
    to balance early syntax patterns with late semantic specialization.

    Args:
        num_layers: Total number of transformer layers in the model.

    Returns:
        Recommended layer index (0-indexed).

    Example:
        >>> # For Qwen2.5-0.5B (24 layers)
        >>> recommended = get_recommended_layer_idx(24)
        >>> print(recommended)
        12
    """
    return num_layers // 2


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Default model for Q1 (local directory; downloaded once, then used offline)
DEFAULT_MODEL_NAME: str = str(MODELS_DIR / "qwen2.5-0.5b")

# Default layer index for Qwen2.5-0.5B (24 layers)
# NOTE: Recommended is get_recommended_layer_idx(24) = 12. Users can override via:
# ModelWrapper(model_name, layer_idx=12) or layer_idx=get_recommended_layer_idx(24).
DEFAULT_LAYER_IDX: int = 12

# Default generation parameters
DEFAULT_MAX_TOKENS: int = 50
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.95

# Device
DEVICE: torch.device = get_device()
DEVICE_NAME: str = get_device_name()
