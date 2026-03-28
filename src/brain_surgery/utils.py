"""Common configuration and utilities for modular code organization.

This module centralizes project configuration including directory paths,
device management, and helper functions. Using a single configuration source
enables easy switching of models, datasets, and output directories without
modifying core code.

Typical usage:
    >>> from brain_surgery.utils import ROOT_DIR, DATA_DIR, get_device
    >>> print(f"Project root: {ROOT_DIR}")
    >>> print(f"Data directory: {DATA_DIR}")
    >>> device = get_device()
    >>> print(f"Using device: {device}")
"""

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
]


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


def get_device() -> torch.device:
    """Get the appropriate PyTorch device (cuda or cpu).

    Automatically detects GPU availability and returns cuda if available,
    otherwise falls back to cpu.

    Returns:
        A torch.device object set to cuda if GPU is available, else cpu.

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
        Using device: cuda
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_name() -> str:
    """Get the name of the current device as a string.

    Returns:
        Either "cuda" or "cpu" depending on availability.

    Example:
        >>> device_name = get_device_name()
        >>> print(f"Device: {device_name}")
        Device: cuda
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Default model for Q1 (can be changed to other Hugging Face models)
DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B"

# Default layer index for 0.5B models (middle layers recommended)
DEFAULT_LAYER_IDX: int = 4

# Default generation parameters
DEFAULT_MAX_TOKENS: int = 50
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TOP_P: float = 0.95

# Device
DEVICE: torch.device = get_device()
DEVICE_NAME: str = get_device_name()
