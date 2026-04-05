"""Unit tests for SAETrainer training loop and resume decisions."""

from pathlib import Path

import pytest
import torch

import brain_surgery.trainer as trainer_module
from brain_surgery.sae import SparseAutoencoder
from brain_surgery.trainer import SAETrainer


def _invoke_should_resume(trainer: SAETrainer) -> bool:
    return bool(getattr(trainer, "_should_resume")())


def test_trainer_train_produces_checkpoint_and_summary(tmp_path: Path) -> None:
    """Verify core training loop emits history, summary, and checkpoint."""
    checkpoint = tmp_path / "sae.pt"
    model = SparseAutoencoder(input_dim=896, latent_dim=64)
    trainer = SAETrainer(
        model=model,
        learning_rate=1e-3,
        batch_size=4,
        num_epochs=2,
        l1_lambda=1e-3,
        patience=5,
        checkpoint_path=checkpoint,
        auto_resume=False,
        use_wandb=False,
        use_tensorboard=False,
    )

    activation_matrix = torch.randn(12, 896)
    history, summary = trainer.train(activation_matrix)

    assert len(history) >= 1
    assert summary.epochs >= 1
    assert isinstance(summary.final_loss, float)
    assert checkpoint.exists()


def test_trainer_should_resume_interactive_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify interactive resume prompt path returns True on affirmative input."""
    checkpoint = tmp_path / "sae.pt"
    checkpoint.write_bytes(b"stub")

    trainer = SAETrainer(
        model=SparseAutoencoder(input_dim=896, latent_dim=32),
        checkpoint_path=checkpoint,
        auto_resume=None,
        use_wandb=False,
        use_tensorboard=False,
        num_epochs=1,
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert _invoke_should_resume(trainer) is True


def test_trainer_resume_from_existing_checkpoint(tmp_path: Path) -> None:
    """Verify auto-resume path loads existing checkpoint and continues training."""
    checkpoint = tmp_path / "sae.pt"

    first_model = SparseAutoencoder(input_dim=896, latent_dim=16)
    first_trainer = SAETrainer(
        model=first_model,
        checkpoint_path=checkpoint,
        auto_resume=False,
        use_wandb=False,
        use_tensorboard=False,
        num_epochs=1,
        batch_size=2,
    )
    _history1, _summary1 = first_trainer.train(torch.randn(4, 896))

    second_model = SparseAutoencoder(input_dim=896, latent_dim=16)
    second_trainer = SAETrainer(
        model=second_model,
        checkpoint_path=checkpoint,
        auto_resume=True,
        use_wandb=False,
        use_tensorboard=False,
        num_epochs=1,
        batch_size=2,
    )
    history2, summary2 = second_trainer.train(torch.randn(4, 896))

    assert len(history2) >= 1
    assert summary2.epochs >= 1


def test_trainer_should_resume_false_when_checkpoint_missing(tmp_path: Path) -> None:
    """Verify missing checkpoint short-circuits resume logic."""
    trainer = SAETrainer(
        model=SparseAutoencoder(input_dim=896, latent_dim=16),
        checkpoint_path=tmp_path / "missing.pt",
        auto_resume=None,
        use_wandb=False,
        use_tensorboard=False,
        num_epochs=1,
    )
    assert _invoke_should_resume(trainer) is False


def test_trainer_should_resume_false_when_auto_resume_disabled(
    tmp_path: Path,
) -> None:
    """Verify explicit auto_resume=False overrides existing checkpoints."""
    checkpoint = tmp_path / "exists.pt"
    checkpoint.write_bytes(b"data")
    trainer = SAETrainer(
        model=SparseAutoencoder(input_dim=896, latent_dim=16),
        checkpoint_path=checkpoint,
        auto_resume=False,
        use_wandb=False,
        use_tensorboard=False,
        num_epochs=1,
    )
    assert _invoke_should_resume(trainer) is False


def test_trainer_wandb_and_tensorboard_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify optional WandB/TensorBoard branches execute without network calls."""

    class FakeRun:
        def __init__(self) -> None:
            self.logged_artifact = False
            self.finished = False

        def log_artifact(self, artifact: object) -> None:
            _ = artifact
            self.logged_artifact = True

        def finish(self) -> None:
            self.finished = True

    class FakeArtifact:
        def __init__(self, name: str, type: str) -> None:  # noqa: A002
            _ = name
            _ = type
            self.files: list[str] = []

        def add_file(self, file_path: str) -> None:
            self.files.append(file_path)

    class FakeWandbModule:
        def __init__(self) -> None:
            self.run = FakeRun()
            self.logged = False

        def login(self, key: str, relogin: bool) -> None:
            _ = key
            _ = relogin

        def init(self, **kwargs: object) -> FakeRun:
            _ = kwargs
            return self.run

        def log(self, data: dict[str, float | int]) -> None:
            _ = data
            self.logged = True

        Artifact = FakeArtifact

    fake_wandb = FakeWandbModule()
    monkeypatch.setattr(trainer_module, "_wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setenv("WANDB_PROJECT", "dummy-proj")

    checkpoint = tmp_path / "wandb_sae.pt"
    tb_dir = tmp_path / "tb"
    trainer = SAETrainer(
        model=SparseAutoencoder(input_dim=896, latent_dim=16),
        checkpoint_path=checkpoint,
        auto_resume=False,
        use_wandb=True,
        use_tensorboard=True,
        tensorboard_log_dir=tb_dir,
        num_epochs=1,
        batch_size=2,
    )

    _history, _summary = trainer.train(torch.randn(6, 896))

    assert fake_wandb.logged is True
    assert fake_wandb.run.logged_artifact is True
    assert fake_wandb.run.finished is True
    assert tb_dir.exists()


def test_trainer_early_stopping_branch(tmp_path: Path) -> None:
    """Verify early stopping branch triggers when loss does not improve."""

    class ConstantLossSAE(SparseAutoencoder):
        def compute_loss(
            self, x: torch.Tensor, l1_lambda: float
        ) -> dict[str, torch.Tensor]:
            _ = x
            _ = l1_lambda
            value = torch.tensor(1.0, requires_grad=True)
            latent = torch.zeros(2, self.latent_dim)
            recon = torch.zeros(2, self.input_dim)
            return {
                "loss": value,
                "reconstruction_loss": torch.tensor(1.0),
                "l1_loss": torch.tensor(0.0),
                "latent": latent,
                "reconstruction": recon,
            }

    trainer = SAETrainer(
        model=ConstantLossSAE(input_dim=896, latent_dim=8),
        checkpoint_path=tmp_path / "early_stop.pt",
        auto_resume=False,
        use_wandb=False,
        use_tensorboard=False,
        num_epochs=5,
        patience=1,
        batch_size=2,
    )

    history, summary = trainer.train(torch.randn(4, 896))
    assert len(history) < 5
    assert summary.epochs == len(history)
