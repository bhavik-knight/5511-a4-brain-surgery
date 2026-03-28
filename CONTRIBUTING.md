# Contributing to Brain Surgery SAE

Thank you for your interest in contributing to our Sparse Autoencoders and Neural Network Interpretability project! We welcome contributions from the community. This document provides guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Code Standards](#code-standards)
- [Branching Strategy](#branching-strategy)
- [Commit Guidelines](#commit-guidelines)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)

## Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) for package management

### Development Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url> a4-brain-surgery
   cd a4-brain-surgery
   ```

2. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

4. **Verify your setup:**
   ```bash
   uv run pytest
   ```

This ensures all developers use consistent tool versions and configurations.

## Code Standards

### Python Version

- **Minimum:** Python 3.10+
- Use modern Python type hints:
  - ✅ `list[str]` instead of `List[str]`
  - ✅ `dict[str, int]` instead of `Dict[str, int]`
  - ✅ `tuple[int, ...]` instead of `Tuple[int, ...]`

### Type Annotations

**All functions and classes must have complete type annotations.** This is enforced by `mypy` in strict mode.

#### Function Example

```python
def extract_activations(model_output: dict[str, any], layer: str) -> list[float]:
    """Extract neural network activations from a specific layer.

    Args:
        model_output: Dictionary containing model layer outputs.
        layer: The target layer name.

    Returns:
        A list of activation values from the specified layer.

    Raises:
        ValueError: If layer is not found in model output.
    """
    if layer not in model_output:
        raise ValueError(f"Layer '{layer}' not found in model output")
    
    return model_output[layer]
```

#### Class Example with Pydantic

```python
from pydantic import BaseModel, Field

class ActivationData(BaseModel):
    """Schema for activation data in the SAE analysis pipeline.

    Attributes:
        layer_name: The neural network layer name.
        activations: Raw activation values.
        model_id: Identifier for the source model.
    """
    
    layer_name: str = Field(..., description="Name of the neural network layer")
    activations: list[float] = Field(..., description="Activation values")
    model_id: str = Field(..., description="Identifier for source model")
```

### Docstrings

**All public functions, classes, and modules must have Google-style docstrings.**

#### Module-level docstring

```python
"""Sparse Autoencoder utilities for neural network interpretation.

This module provides functions for extracting activations, training SAEs,
and analyzing learned representations in deep neural networks.
"""
```

#### Function docstring

```python
def calculate_reconstruction_loss(
    original: list[float],
    reconstructed: list[float],
) -> float:
    """Calculate reconstruction loss between original and reconstructed activations.

    Args:
        original: Original activation values.
        reconstructed: Reconstructed activation values from SAE.

    Returns:
        Mean squared error loss between original and reconstructed.

    Raises:
        ValueError: If arrays have different lengths.

    Example:
        >>> orig = [0.1, 0.5, 0.9]
        >>> recon = [0.12, 0.48, 0.91]
        >>> calculate_reconstruction_loss(orig, recon)
        0.000333...
    """
    if len(original) != len(reconstructed):
        raise ValueError("Arrays must have equal length")
    
    return sum((o - r) ** 2 for o, r in zip(original, reconstructed)) / len(original)
```

### Linting and Formatting with Ruff

All code is automatically formatted and linted using **Ruff**. Pre-commit hooks enforce this.

#### Key Ruff rules enforced:
- Line length: 100 characters
- Unused imports are removed
- Trailing commas for multi-line structures
- Sort imports using `isort` plugin

Run before committing:
```bash
uv run ruff check --fix .
uv run ruff format .
```

### Type Checking with Mypy

All code must pass strict mypy type checking. This is enforced in CI/CD.

Run locally:
```bash
uv run mypy .
```

Fix common issues:
- Ensure all function parameters have type annotations
- Ensure all function return types are annotated
- Use `Optional[T]` or `T | None` for nullable types
- Use protocol/abstract base classes for duck typing

## Branching Strategy

Our project uses a **Git Flow** branching model with specific naming conventions.

### Branch Name Format

Branch names must follow one of these prefixes:

| Prefix | Usage | Example |
|--------|-------|---------|
| `feat/` | New feature | `feat/named-entity-recognition` |
| `fix/` | Bug fix | `fix/tokenizer-edge-case` |
| `exp/` | Experimental work | `exp/bert-fine-tuning` |
| `docs/` | Documentation | `docs/api-reference` |
| `refactor/` | Code refactoring | `refactor/reduce-complexity` |

### Rules

- ✅ Use lowercase with hyphens: `feat/my-feature-name`
- ❌ Do NOT use spaces or underscores: `feat/myFeatureName`, `feat/my_feature_name`
- ✅ Keep names descriptive but concise
- ❌ Never push directly to `main` — all changes require a Pull Request

## Commit Guidelines

We follow **Conventional Commits** specification for clear, semantic commit messages.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat:** A new feature
- **fix:** A bug fix
- **docs:** Documentation changes
- **style:** Code style changes (formatting, semicolons, etc.)
- **refactor:** Code refactoring without feature changes
- **perf:** Performance improvements
- **test:** Adding or updating tests
- **chore:** Build process, dependencies, tools

### Examples

```bash
# Feature with scope
git commit -m "feat(tokenizer): add support for subword tokenization"

# Bug fix
git commit -m "fix(ner): handle empty token lists gracefully"

# Documentation
git commit -m "docs: update installation instructions"

# With body and footer
git commit -m "feat(model): implement attention mechanism

Implement multi-head attention for transformer encoder.
This improves model performance on benchmark datasets.

Closes #42"
```

### Commit Best Practices

- ✅ Make small, focused commits (one logical change per commit)
- ✅ Write clear, descriptive commit messages
- ✅ Reference issues: `Closes #123` or `Fixes #456`
- ✅ Commit early and often
- ❌ Do NOT commit incomplete work or failing tests

## Submitting Changes

When you're ready to submit your changes:

1. **Create and switch to your feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

3. **Ensure code quality before pushing:**
   ```bash
   uv run ruff check --fix .
   uv run ruff format .
   uv run mypy .
   uv run pytest
   ```

4. **Push your branch:**
   ```bash
   git push origin feat/your-feature-name
   ```

5. **Open a Pull Request** using the provided [PR template](.github/pull_request_template.md).

GitHub will automatically load the PR template when you open a new pull request. Please fill it out completely with:
- A clear description of your changes
- The type of change (feature, bug fix, experiment, docs, refactor)
- Evidence that you've verified code quality (`ruff check`, `mypy`, tests)
- Related issue numbers if applicable

## Project Structure

```
a4-brain-surgery/
├── src/
│   └── brain_surgery/
│       ├── __init__.py
│       ├── sae.py                  # Sparse Autoencoder implementation
│       ├── data_gen.py             # Activation data generation
│       ├── interpret.py            # SAE interpretation utilities
│       ├── intervention.py         # Network intervention methods
│       ├── model_wrapper.py        # Model wrapping and interface
│       ├── main.py                 # Main entry point
│       └── utils.py                # Utility functions
├── tests/
│   ├── test_sae.py                 # SAE module tests
│   ├── test_activation_capture.py  # Data generation tests
│   ├── test_intervention.py        # Intervention method tests
│   └── test_model_wrapper.py       # Model wrapper tests
├── data/
│   ├── activations/                # Captured neuron activations
│   └── corpus/                     # Training corpus data
├── results/
│   ├── experiments/                # Experiment outputs
│   ├── features/                   # Extracted features
│   └── metrics/                    # Performance metrics
├── pyproject.toml                  # uv configuration, dependencies
├── .pre-commit-config.yaml         # Pre-commit hooks
├── README.md                       # Project documentation
├── CONTRIBUTING.md                 # This file
└── LICENSE                         # Project license
```

## Tools Overview

### uv - Package Management

Fast Python package installer and lockfile manager.

```bash
# Install dependencies from pyproject.toml
uv sync

# Run a command in the venv
uv run python -c "import nltk"
uv run pytest

# Add a new dependency
uv add numpy

# Add a dev dependency
uv add --group dev pytest
```

### Ruff - Linting & Formatting

Ultra-fast Python linter and code formatter.

```bash
# Check for linting issues
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/

# Format code
uv run ruff format src/
```

### Mypy - Type Checking

Static type checker for Python in strict mode.

```bash
# Check all types
uv run mypy .

# Check specific file
uv run mypy src/nlp/preprocessing.py
```

## Questions or Need Help?

- Open an issue for bugs or feature requests
- Check existing issues before creating a new one
- Use discussion threads for questions

---

**Thank you for contributing! Happy coding! 🧠**
