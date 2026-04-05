"""Training loop for Sparse Autoencoder (SAE) with early stopping.

Handles the training process including loss computation, optimization,
and early stopping based on validation loss plateau.
"""

from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
from collections.abc import Callable
from types import ModuleType
from typing import cast

import torch
from torch import Tensor
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter

from .sae import SparseAutoencoder
from .utils import CHECKPOINTS_DIR, RESULTS_DIR

_wandb: ModuleType | None
try:
    _wandb = import_module("wandb")
except ImportError:  # pragma: no cover - optional dependency at runtime
    _wandb = None


@dataclass(frozen=True)
class TrainingSummary:
    """Summary metrics from an SAE training run.

    Args:
        epochs: Number of epochs completed.
        final_loss: Final total loss value.
        dead_neuron_fraction: Fraction of latent units with zero activation
            after the first epoch.
        best_loss: Best loss observed during training.
    """

    epochs: int
    final_loss: float
    dead_neuron_fraction: float
    best_loss: float


class SAETrainer:
    """Trainer for sparse autoencoder with early stopping."""

    def __init__(
        self,
        model: SparseAutoencoder,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 20,
        l1_lambda: float = 1e-3,
        patience: int = 5,
        device: torch.device | None = None,
        checkpoint_path: Path | None = None,
        auto_resume: bool | None = None,
        use_wandb: bool = True,
        wandb_project: str = "a4-brain-surgery",
        wandb_run_name: str | None = None,
        wandb_dir: Path | None = None,
        use_tensorboard: bool = True,
        tensorboard_log_dir: Path | None = None,
    ) -> None:
        """Initialize the SAE trainer.

        Args:
            model: Sparse autoencoder model to train.
            learning_rate: Learning rate for Adam optimizer. Defaults to 1e-4.
            batch_size: Batch size for data loader. Defaults to 32.
            num_epochs: Maximum number of training epochs. Defaults to 20.
            l1_lambda: Weight for L1 sparsity penalty. Defaults to 1e-3.
            patience: Number of epochs without improvement before early stopping.
            device: Device to train on. Defaults to CUDA if available.
            checkpoint_path: Optional checkpoint path for resume support.
            auto_resume: Resume behavior for existing checkpoints.
                - None: ask interactively
                - True: always resume if checkpoint exists
                - False: always train from scratch
            use_wandb: Whether to enable WandB logging.
            wandb_project: WandB project name.
            wandb_run_name: Optional WandB run name.
            wandb_dir: Optional local directory for WandB run files.
            use_tensorboard: Whether to enable TensorBoard logging.
            tensorboard_log_dir: Optional TensorBoard log directory.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.l1_lambda = l1_lambda
        self.patience = patience
        self.auto_resume = auto_resume
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_path = checkpoint_path or (
            CHECKPOINTS_DIR / "sae_checkpoint.pt"
        )
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_dir = wandb_dir
        self.use_tensorboard = use_tensorboard
        self.tensorboard_log_dir = tensorboard_log_dir or (
            RESULTS_DIR / "logs" / "tensorboard"
        )

        checkpoint_dir = self.checkpoint_path.parent
        self.best_checkpoint_path = checkpoint_dir / "sae_best.pt"
        self.final_checkpoint_path = checkpoint_dir / "sae_final.pt"

        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.writer: SummaryWriter | None = None
        self._wandb: ModuleType | None = _wandb

        self._wandb_run: object | None = None
        if self.use_wandb and self._wandb is not None:
            api_key = os.getenv("WANDB_API_KEY")
            login_fn = getattr(self._wandb, "login", None)
            if api_key and callable(login_fn):
                cast_login_fn = cast(Callable[..., object], login_fn)
                cast_login_fn(key=api_key, relogin=True)

            project = os.getenv("WANDB_PROJECT", self.wandb_project)
            entity = os.getenv("WANDB_ENTITY")
            init_fn = getattr(self._wandb, "init", None)
            if callable(init_fn):
                cast_init_fn = cast(Callable[..., object], init_fn)
                wandb_dir_str: str | None = None
                if self.wandb_dir is not None:
                    self.wandb_dir.mkdir(parents=True, exist_ok=True)
                    wandb_dir_str = str(self.wandb_dir)
                self._wandb_run = cast_init_fn(
                    project=project,
                    entity=entity,
                    name=self.wandb_run_name,
                    dir=wandb_dir_str,
                    config={
                        "lr": self.learning_rate,
                        "l1_coeff": self.l1_lambda,
                        "batch_size": self.batch_size,
                        "epochs": self.num_epochs,
                        "expansion_factor": self.model.latent_dim
                        / self.model.input_dim,
                        "input_dim": self.model.input_dim,
                        "latent_dim": self.model.latent_dim,
                        "device": str(self.device),
                    },
                )

        if self.use_tensorboard:
            self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir))

    def _should_resume(self) -> bool:
        """Determine whether to resume from an existing checkpoint.

        Returns:
            True if training should resume from checkpoint, False otherwise.
        """
        if not self.checkpoint_path.exists():
            return False

        if self.auto_resume is True:
            return True
        if self.auto_resume is False:
            return False

        response = (
            input(
                f"Checkpoint found at {self.checkpoint_path}. Resume from it? [y/N]: "
            )
            .strip()
            .lower()
        )
        return response in {"y", "yes"}

    def train(
        self, activation_matrix: Tensor
    ) -> tuple[list[dict[str, float | int]], TrainingSummary]:
        """Train the sparse autoencoder on activation data.

        Args:
            activation_matrix: Activation tensor of shape (num_tokens, hidden_dim).

        Returns:
            Tuple containing:
                - History list with per-epoch loss metrics
                - TrainingSummary with dead neuron fraction and best loss
        """
        if self._should_resume():
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            print("Loaded SAE checkpoint for resume.")

        dataset = TensorDataset(activation_matrix)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        history: list[dict[str, float | int]] = []
        best_loss = float("inf")
        patience_counter = 0
        final_loss = 0.0
        dead_neuron_fraction = 0.0

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_l1 = 0.0
            latent_active_counts = torch.zeros(
                self.model.latent_dim, device="cpu", dtype=torch.int64
            )

            for batch in dataloader:
                x = batch[0].to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model.compute_loss(x, self.l1_lambda)
                loss = outputs["loss"]
                recon_loss = outputs["reconstruction_loss"]
                l1_loss = outputs["l1_loss"]

                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.item())
                epoch_recon += float(recon_loss.item())
                epoch_l1 += float(l1_loss.item())

                with torch.no_grad():
                    latents = outputs["latent"]
                    active = (latents > 0).sum(dim=0).to("cpu")
                    latent_active_counts += active

            num_batches = max(1, len(dataloader))
            epoch_loss /= num_batches
            epoch_recon /= num_batches
            epoch_l1 /= num_batches
            final_loss = epoch_loss

            history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "reconstruction_loss": epoch_recon,
                    "l1_loss": epoch_l1,
                }
            )

            print(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Recon: {epoch_recon:.4f} | "
                f"L1: {epoch_l1:.4f}"
            )

            dead_neuron_fraction_epoch = float(
                (latent_active_counts == 0).float().mean().item()
            )

            if epoch == 0:
                dead_neuron_fraction = dead_neuron_fraction_epoch
                print(f"Dead-Neuron Fraction after Epoch 1: {dead_neuron_fraction:.4f}")

            if self._wandb_run is not None and self._wandb is not None:
                log_fn = getattr(self._wandb, "log", None)
                if callable(log_fn):
                    cast_log_fn = cast(Callable[..., object], log_fn)
                    cast_log_fn(
                        {
                            "epoch": epoch + 1,
                            "mse_loss": epoch_recon,
                            "l1_loss": epoch_l1,
                            "total_loss": epoch_loss,
                            "dead_neuron_fraction": dead_neuron_fraction_epoch,
                        }
                    )

            if self.writer is not None:
                step = epoch + 1
                self.writer.add_scalar("loss/mse", epoch_recon, step)
                self.writer.add_scalar("loss/l1", epoch_l1, step)
                self.writer.add_scalar("loss/total", epoch_loss, step)
                self.writer.add_scalar(
                    "health/dead_neuron_fraction",
                    dead_neuron_fraction_epoch,
                    step,
                )

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                self.best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.model.state_dict_for_checkpoint(),
                    self.best_checkpoint_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(
                        f"\nEarly stopping triggered at epoch {epoch + 1} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

        summary = TrainingSummary(
            epochs=len(history),
            final_loss=final_loss,
            dead_neuron_fraction=dead_neuron_fraction,
            best_loss=best_loss,
        )

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        final_state = self.model.state_dict_for_checkpoint()
        torch.save(final_state, self.final_checkpoint_path)
        torch.save(final_state, self.checkpoint_path)

        if not self.best_checkpoint_path.exists():
            torch.save(final_state, self.best_checkpoint_path)

        if self._wandb_run is not None and self._wandb is not None:
            artifact_cls = getattr(self._wandb, "Artifact", None)
            if callable(artifact_cls):
                artifact = artifact_cls("sae-checkpoint", type="model")
                add_file_fn = getattr(artifact, "add_file", None)
                if callable(add_file_fn):
                    cast_add_file_fn = cast(Callable[..., object], add_file_fn)
                    cast_add_file_fn(str(self.checkpoint_path))
                log_artifact_fn = getattr(self._wandb_run, "log_artifact", None)
                if callable(log_artifact_fn):
                    cast_log_artifact_fn = cast(Callable[..., object], log_artifact_fn)
                    cast_log_artifact_fn(artifact)
            finish_fn = getattr(self._wandb_run, "finish", None)
            if callable(finish_fn):
                cast_finish_fn = cast(Callable[..., object], finish_fn)
                cast_finish_fn()

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        return history, summary
