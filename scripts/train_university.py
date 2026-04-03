"""Headless university-cluster training entrypoint for SAE experiments."""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime, UTC
from pathlib import Path

import torch

from brain_surgery.sae import SparseAutoencoder
from brain_surgery.trainer import SAETrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for headless training.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train SparseAutoencoder in headless cluster mode."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/activations/soccer_activations_dataset.pt"),
        help="Path to .pt dataset containing activation_matrix.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/sae_checkpoint.pt"),
        help="Checkpoint path for resume/save.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--l1",
        type=float,
        default=2e-3,
        help="L1 sparsity coefficient.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from checkpoint if it exists.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging and use TensorBoard only.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="a4-brain-surgery",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional WandB run name.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=Path("runs"),
        help="TensorBoard log directory.",
    )
    return parser.parse_args()


def log_crash(exc: BaseException, log_path: Path = Path("crash_report.log")) -> None:
    """Append crash details to crash_report.log.

    Args:
        exc: Exception that occurred.
        log_path: Crash log file path.
    """
    stamp = datetime.now(tz=UTC).isoformat()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {type(exc).__name__}: {exc}\n")
        fh.write(traceback.format_exc())
        fh.write("\n" + "-" * 80 + "\n")


def main() -> None:
    """Run headless SAE training for university cluster execution."""
    args = parse_args()

    try:
        payload = torch.load(args.dataset, map_location="cpu")
        activation_matrix = payload["activation_matrix"]

        model = SparseAutoencoder(input_dim=896, latent_dim=3584)
        trainer = SAETrainer(
            model=model,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            l1_lambda=args.l1,
            patience=args.patience,
            checkpoint_path=args.checkpoint,
            auto_resume=args.resume,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            use_tensorboard=True,
            tensorboard_log_dir=args.tensorboard_dir,
        )

        _, summary = trainer.train(activation_matrix)
        print(
            "Training complete: "
            f"epochs={summary.epochs}, "
            f"final_loss={summary.final_loss:.6f}, "
            f"dead_neuron_fraction={summary.dead_neuron_fraction:.6f}"
        )

    except torch.cuda.OutOfMemoryError as exc:
        log_crash(exc)
        print("CUDA OOM encountered. Details saved to crash_report.log")
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            log_crash(exc)
            print("Runtime OOM encountered. Details saved to crash_report.log")
        raise


if __name__ == "__main__":
    main()
