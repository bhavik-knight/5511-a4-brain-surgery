"""Local L1 hyperparameter experiment for Qwen 2.5 0.5B optimization.

This script tests multiple L1 lambda values to find the optimal trade-off
between sparsity (L0) and reconstruction error (MSE) before moving to
the A100 cluster.
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
from brain_surgery.sae import SparseAutoencoder
from brain_surgery.trainer import SAETrainer
from brain_surgery.utils import ACTIVATIONS_DIR, METRICS_DIR, ensure_dir_exists


def run_l1_test() -> None:
    """Run SAE training loop across multiple L1 values and log metrics."""
    dataset_path = ACTIVATIONS_DIR / "soccer_activations_dataset.pt"

    if not dataset_path.exists():
        print(f"Error: Activation dataset not found at {dataset_path}")
        print("Please run 'uv run python -m brain_surgery.data_gen' first.")
        return

    print(f"Loading activations from {dataset_path}...")
    payload = torch.load(dataset_path, weights_only=True)
    activations = payload["activation_matrix"]
    input_dim = activations.shape[1]

    # Configuration
    # For Qwen 2.5 0.5B, hidden_dim is 896.
    # Use expansion factor 4 for local testing (3584 latents).
    latent_dim = 3584
    l1_values = [0.01, 0.001, 0.0001]
    num_epochs = 5
    batch_size = 4096

    results = []

    ensure_dir_exists(METRICS_DIR)

    for l1 in l1_values:
        print(f"\n{'=' * 40}")
        print(f"Testing L1 Lambda: {l1}")
        print(f"{'=' * 40}")

        sae = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        trainer = SAETrainer(
            model=sae,
            learning_rate=1e-4,
            batch_size=batch_size,
            num_epochs=num_epochs,
            l1_lambda=l1,
            # Disable wandb for local sweep to keep it simple
            use_wandb=False,
        )

        _history, summary = trainer.train(activations)

        # Calculate final L0 (average number of active features per token)
        # We run a subset of the activations to get an estimate of final sparsity
        with torch.no_grad():
            # Test on first 10k tokens for speed
            test_batch = activations[:10000].to(sae.device)
            z = sae.encode(test_batch)
            l0 = (z > 0).float().sum(dim=-1).mean().item()

        results.append(
            {
                "l1_lambda": l1,
                "final_mse": summary.final_loss,
                "avg_l0": round(l0, 2),
                "dead_neuron_pct": round(summary.dead_neuron_fraction * 100, 2),
            }
        )

        print(f"Results for L1={l1}: L0={l0:.2f}, MSE={summary.final_loss:.6f}")

    # Create summary report
    df = pd.DataFrame(results)
    report_path = METRICS_DIR / "local_l1_test_results.csv"
    df.to_csv(report_path, index=False)
    print(f"\nFull report saved to {report_path}")
    print(df.to_string(index=False))

    # Generate summary plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # L1 vs L0
    ax1.plot(df["l1_lambda"], df["avg_l0"], marker="o", linestyle="-", color="teal")
    ax1.set_xscale("log")
    ax1.set_xlabel("L1 Lambda (Log Scale)")
    ax1.set_ylabel("L0 (Avg Active Features)")
    ax1.set_title("Sparsity vs Penalty")
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    # L1 vs MSE
    ax2.plot(df["l1_lambda"], df["final_mse"], marker="s", ls="--", color="crimson")
    ax2.set_xscale("log")
    ax2.set_xlabel("L1 Lambda (Log Scale)")
    ax2.set_ylabel("Final MSE Loss")
    ax2.set_title("Reconstruction Error vs Penalty")
    ax2.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plot_path = METRICS_DIR / "local_l1_test_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    run_l1_test()
