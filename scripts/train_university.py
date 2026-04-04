"""Headless university-cluster training entrypoint for SAE experiments."""

import argparse
import json
import traceback
from collections.abc import Sequence
from datetime import datetime, UTC
from pathlib import Path

import torch
from dotenv import load_dotenv

from brain_surgery.sae import SparseAutoencoder
from brain_surgery.trainer import SAETrainer
from brain_surgery.utils import ACTIVATIONS_DIR, create_run_output_dirs, generate_run_id

load_dotenv()

DEFAULT_EXPANSION_FACTOR = 4


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for headless training.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train SparseAutoencoder in headless cluster mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ACTIVATIONS_DIR / "soccer_activations_dataset.pt",
        help="Path to .pt dataset containing activation_matrix.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path for resume/save.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
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
        default=None,
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. If omitted, generates run_YYYYMMDD_HHMM.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 epoch on first 5 prompts to verify environment.",
    )
    return parser.parse_args()


def log_crash(exc: BaseException, log_path: Path) -> None:
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


def _subset_activation_matrix_for_smoke(payload: dict[str, object]) -> torch.Tensor:
    """Extract a 5-prompt activation subset from dataset payload for smoke tests."""
    activation_matrix_obj = payload.get("activation_matrix")
    if not isinstance(activation_matrix_obj, torch.Tensor):
        raise ValueError("Dataset payload missing activation_matrix tensor.")

    metadata_obj = payload.get("metadata")
    if not isinstance(metadata_obj, Sequence):
        return activation_matrix_obj[: min(256, activation_matrix_obj.shape[0])]

    selected_prompt_ids: set[int] = set()
    selected_rows: list[int] = []
    for idx, row in enumerate(metadata_obj):
        if not isinstance(row, dict):
            continue
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, int):
            continue
        if prompt_id not in selected_prompt_ids and len(selected_prompt_ids) >= 5:
            continue
        selected_prompt_ids.add(prompt_id)
        selected_rows.append(idx)

    if not selected_rows:
        return activation_matrix_obj[: min(256, activation_matrix_obj.shape[0])]

    index_tensor = torch.tensor(selected_rows, dtype=torch.long)
    return activation_matrix_obj.index_select(0, index_tensor)


def main() -> None:
    """Run headless SAE training for university cluster execution."""
    args = parse_args()
    run_id = args.run_id or generate_run_id()
    run_dirs = create_run_output_dirs(run_id)
    checkpoint_path = args.checkpoint or (run_dirs["checkpoints"] / "sae_best.pt")
    tensorboard_dir = args.tensorboard_dir or (run_dirs["logs"] / "tensorboard")
    wandb_dir = run_dirs["logs"] / "wandb"
    crash_log_path = run_dirs["logs"] / "crash_report.log"

    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dirs['root']}")
    print(f"Checkpoint output: {checkpoint_path}")
    print(f"TensorBoard dir: {tensorboard_dir}")
    print(f"WandB dir: {wandb_dir}")

    try:
        payload = torch.load(args.dataset, map_location="cpu", weights_only=True)
        activation_matrix = payload["activation_matrix"]
        if not isinstance(activation_matrix, torch.Tensor):
            raise ValueError("Dataset payload missing activation_matrix tensor.")
        if activation_matrix.ndim != 2:
            raise ValueError(
                "activation_matrix must be 2D with shape "
                "(num_tokens, hidden_dim), got "
                f"{tuple(activation_matrix.shape)}"
            )
        run_epochs = args.epochs

        if args.smoke_test:
            activation_matrix = _subset_activation_matrix_for_smoke(payload)
            run_epochs = 1
            print(
                "Smoke test active: running 1 epoch on subset with "
                f"{activation_matrix.shape[0]} token rows (from first 5 prompts)."
            )

        input_dim = int(activation_matrix.shape[1])
        latent_dim = input_dim * DEFAULT_EXPANSION_FACTOR
        print(
            "SAE architecture: "
            f"input_dim={input_dim}, latent_dim={latent_dim}, "
            f"expansion_factor={DEFAULT_EXPANSION_FACTOR}"
        )

        model = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        trainer = SAETrainer(
            model=model,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_epochs=run_epochs,
            l1_lambda=args.l1,
            patience=args.patience,
            checkpoint_path=checkpoint_path,
            auto_resume=args.resume,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_dir=wandb_dir,
            use_tensorboard=True,
            tensorboard_log_dir=tensorboard_dir,
        )

        history, summary = trainer.train(activation_matrix)
        summary_path = run_dirs["metrics_root"] / f"training_losses_{run_id}.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "run_id": run_id,
                    "dataset": str(args.dataset),
                    "checkpoint": str(checkpoint_path),
                    "smoke_test": bool(args.smoke_test),
                    "loss_history": history,
                    "epochs": summary.epochs,
                    "final_loss": summary.final_loss,
                    "dead_neuron_fraction": summary.dead_neuron_fraction,
                    "best_loss": summary.best_loss,
                },
                fh,
                indent=2,
            )
        print(f"Saved training summary: {summary_path}")
        print(
            "Training complete: "
            f"epochs={summary.epochs}, "
            f"final_loss={summary.final_loss:.6f}, "
            f"dead_neuron_fraction={summary.dead_neuron_fraction:.6f}"
        )

    except torch.cuda.OutOfMemoryError as exc:
        log_crash(exc, crash_log_path)
        print(f"CUDA OOM encountered. Details saved to {crash_log_path}")
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            log_crash(exc, crash_log_path)
            print(f"Runtime OOM encountered. Details saved to {crash_log_path}")
        raise


if __name__ == "__main__":
    main()
