# Contributing to Brain Surgery NLP

Thank you for your interest in contributing to our NLP project! We welcome contributions from the community. This document provides guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Code Standards](#code-standards)
- [Branching Strategy](#branching-strategy)
- [Commit Guidelines](#commit-guidelines)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) for package management

### Development Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url> a4-brain-surgery
   cd brain-surgery
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
def preprocess_text(text: str) -> list[str]:
    """Preprocess raw text into tokenized sentences.

    Args:
        text: The raw input text to preprocess.

    Returns:
        A list of preprocessed sentences.

    Raises:
        ValueError: If text is empty or None.
    """
    if not text:
        raise ValueError("Text cannot be empty")
    
    sentences = text.split(".")
    return [s.strip() for s in sentences if s.strip()]
```

#### Class Example with Pydantic

```python
from pydantic import BaseModel, Field

class MedicalRecord(BaseModel):
    """Schema for medical records used in the NLP pipeline.

    Attributes:
        patient_id: Unique identifier for the patient.
        notes: Clinical notes text.
        timestamp: When the record was created.
    """
    
    patient_id: str = Field(..., description="Unique patient identifier")
    notes: str = Field(..., description="Clinical notes text")
    timestamp: str = Field(default_factory=lambda: str(datetime.now()))
```

### Docstrings

**All public functions, classes, and modules must have Google-style docstrings.**

#### Module-level docstring

```python
"""NLP utilities for brain surgery documentation analysis.

This module provides functions for preprocessing, tokenizing, and analyzing
clinical notes related to neurosurgical procedures.
"""
```

#### Function docstring

```python
def calculate_entity_confidence(
    entities: list[dict[str, str]],
    threshold: float = 0.8,
) -> list[dict[str, str | float]]:
    """Filter entities by confidence threshold.

    Args:
        entities: List of entity dictionaries with 'type' and 'text'.
        threshold: Minimum confidence score (0.0 to 1.0).

    Returns:
        Filtered list of entities meeting the threshold.

    Raises:
        ValueError: If threshold is not between 0 and 1.

    Example:
        >>> entities = [{"type": "PROCEDURE", "confidence": 0.95}]
        >>> calculate_entity_confidence(entities, threshold=0.9)
        [{"type": "PROCEDURE", "confidence": 0.95}]
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    return [e for e in entities if e.get("confidence", 0) >= threshold]
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
brain-surgery/
├── src/
│   └── nlp/
│       ├── __init__.py
│       ├── preprocessing.py      # Text cleaning, tokenization
│       ├── models/
│       │   ├── __init__.py
│       │   └── ner.py            # Named entity recognition
│       └── schemas/
│           ├── __init__.py
│           └── medical.py        # Pydantic data schemas
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── pyproject.toml               # uv configuration, dependencies
├── ruff.toml                    # Ruff settings
├── mypy.ini                     # Mypy strict mode config
├── .pre-commit-config.yaml      # Pre-commit hooks
└── README.md
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
