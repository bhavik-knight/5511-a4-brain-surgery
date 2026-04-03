"""Environment-loading tests for headless cluster execution settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def test_dotenv_loads_wandb_api_key(tmp_path: Path) -> None:
    """Verify WANDB_API_KEY is loaded from a mock .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=test_key_123\n", encoding="utf-8")

    os.environ.pop("WANDB_API_KEY", None)
    loaded = load_dotenv(dotenv_path=env_file)

    assert loaded is True
    assert os.getenv("WANDB_API_KEY") == "test_key_123"
