"""Orchestrate the golden activation dataset run."""

from typing import cast

import torch

from brain_surgery.clustering import cluster_features_kmeans, print_cluster_analysis
from brain_surgery.data_gen import ActivationWrapper, DataGenerator
from brain_surgery.interpret import SAEInterpreter
from brain_surgery.intervention import SAEIntervention
from brain_surgery.model_wrapper import ModelWrapper
from brain_surgery.sae import SparseAutoencoder
from brain_surgery.trainer import SAETrainer
from brain_surgery.utils import (
    ACTIVATIONS_DIR,
    CHECKPOINTS_DIR,
    DEFAULT_MODEL_NAME,
    METRICS_DIR,
    ensure_dir_exists,
)


def main() -> None:
    """Run the full activation capture + SAE pipeline."""
    wrapper = ModelWrapper(model_name=DEFAULT_MODEL_NAME)
    generator = DataGenerator(cast(ActivationWrapper, wrapper))

    _, _, summary = generator.generate_dataset()

    print("Golden run completed.")
    print(f"Total prompts: {summary.num_prompts}")
    print(f"Total tokens captured: {summary.total_tokens}")
    print(f"Average sequence length: {summary.average_seq_len:.2f}")
    print(f"Activation matrix shape: {summary.activation_shape}")

    dataset_path = ACTIVATIONS_DIR / "soccer_activations_dataset.pt"
    data = torch.load(dataset_path)
    activation_matrix = data["activation_matrix"]
    print(f"Saved dataset shape: {tuple(activation_matrix.shape)}")

    sae = SparseAutoencoder(input_dim=896, latent_dim=3584)
    trainer = SAETrainer(
        model=sae,
        learning_rate=1e-4,
        batch_size=256,
        num_epochs=3,
        l1_lambda=1e-3,
        patience=5,
        checkpoint_path=CHECKPOINTS_DIR / "sae_checkpoint.pt",
    )
    history, train_summary = trainer.train(activation_matrix)
    print(
        "Training summary: "
        f"epochs={train_summary.epochs}, "
        f"final_loss={train_summary.final_loss:.6f}, "
        f"dead_neuron_fraction={train_summary.dead_neuron_fraction:.6f}"
    )

    ensure_dir_exists(METRICS_DIR)
    ensure_dir_exists(CHECKPOINTS_DIR)
    checkpoint_path = CHECKPOINTS_DIR / "sae_checkpoint.pt"
    torch.save(sae.state_dict_for_checkpoint(), checkpoint_path)
    print(f"Saved SAE checkpoint: {checkpoint_path}")

    interpreter = SAEInterpreter(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
    )
    interpreter.load()
    interpreter.compute_latents()

    clustering_result = cluster_features_kmeans(interpreter, num_clusters=10)
    print_cluster_analysis(clustering_result)

    intervention = SAEIntervention(
        model_wrapper=wrapper,
        checkpoint_path=checkpoint_path,
    )
    intervention.compute_feature_max_values(activation_matrix)
    prompt = "What is the Champions League?"
    candidate_tokens = [" Ronaldo", " Messi", " goal", " football"]
    baseline = intervention.compare_next_token_logprobs(
        prompt,
        candidate_tokens,
    )
    clamped = intervention.compare_next_token_logprobs(
        prompt,
        candidate_tokens,
        feature_index=1625,
        clamp_multiplier=8.0,
    )

    print("Ronaldo log-prob shift (clamped - baseline):")
    for token in candidate_tokens:
        if token in baseline and token in clamped:
            delta = clamped[token] - baseline[token]
            print(f"{token}: {delta:+.4f}")


if __name__ == "__main__":
    main()
