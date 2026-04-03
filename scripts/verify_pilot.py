"""Pilot verification script for Q4/Q5/Q6 with dtype audit.

Run with:
    uv run scripts/verify_pilot.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans

from brain_surgery.clustering import cluster_features_kmeans
from brain_surgery.interpret import SAEInterpreter
from brain_surgery.intervention import SAEIntervention
from brain_surgery.model_wrapper import ModelWrapper
from brain_surgery.utils import DEFAULT_MODEL_NAME, ROOT_DIR

DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "sae_checkpoint.pt"
DEFAULT_DATASET = ROOT_DIR / "data" / "activations" / "soccer_activations_dataset.pt"


def _print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _validate_default_files(checkpoint_path: Path, dataset_path: Path) -> None:
    missing: list[Path] = []
    if not checkpoint_path.exists():
        missing.append(checkpoint_path)
    if not dataset_path.exists():
        missing.append(dataset_path)

    if missing:
        lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required pilot artifacts are missing:\n"
            f"{lines}\n"
            "Generate them first (golden run / training pipeline), then retry."
        )


def _safe_token_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "<none>"
    return str(value)


def run_phase_q4_q5(interpreter: SAEInterpreter) -> None:
    """Run feature interpretation and clustering summary for Q4/Q5."""
    _print_header("Phase 1: Q4/Q5 Interpretability Check")

    top_features = interpreter.rank_features_by_max_activation(top_k=10)
    print("Top 10 features with top 10 activating tokens:")
    for feature in top_features:
        feature_index_raw = feature.get("feature_index")
        if not isinstance(feature_index_raw, int):
            continue
        feature_index = feature_index_raw
        max_value = feature.get("max_feature_value")
        max_value_str = (
            f"{max_value:.6f}" if isinstance(max_value, float) else str(max_value)
        )

        top_examples = interpreter.get_top_examples_for_feature(feature_index, top_k=10)
        token_texts = [
            _safe_token_text(example.get("token_text")) for example in top_examples
        ]
        print(
            f"  Feature {feature_index:4d} | max={max_value_str} | tokens={token_texts}"
        )

    if interpreter.latents is None:
        raise RuntimeError("Interpreter latents missing. Call compute_latents() first.")

    feature_profiles = interpreter.latents.T.cpu().numpy()
    _print_elbow_table(feature_profiles)
    _print_dbscan_summary(feature_profiles)

    clustering = cluster_features_kmeans(interpreter, num_clusters=10, random_state=42)
    print("\nK-Means cluster summary (10 clusters):")
    for summary in clustering["cluster_summaries"]:
        cluster_id = summary["cluster_id"]
        num_features = summary["num_features"]
        representative = summary["representative_feature"]
        top_tokens = summary["representative_tokens"][:3]
        print(
            f"  Cluster {cluster_id:2d} | size={num_features:4d} | "
            f"rep_feature={representative} | tokens={top_tokens}"
        )


def _print_elbow_table(feature_profiles: np.ndarray) -> None:
    """Print SSE for k in [2, 20] as an elbow-method diagnostic table."""
    print("\nElbow diagnostics (K-Means SSE):")
    print("  k | inertia (SSE)")
    print("  --+----------------")
    for k in range(2, 21):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(feature_profiles)
        print(f"  {k:2d} | {model.inertia_:14.4f}")


def _print_dbscan_summary(feature_profiles: np.ndarray) -> None:
    """Run DBSCAN and print density/noise summary as a secondary pass."""
    print("\nDBSCAN secondary pass:")
    dbscan = DBSCAN(eps=2.0, min_samples=5)
    labels = dbscan.fit_predict(feature_profiles)

    noise_count = int(np.sum(labels == -1))
    unique_labels = sorted(int(label) for label in np.unique(labels) if label != -1)
    print(f"  clusters_found={len(unique_labels)} | noise_features={noise_count}")
    if unique_labels:
        counts = [int(np.sum(labels == label)) for label in unique_labels]
        print(f"  cluster_sizes={counts}")
    else:
        print("  No dense DBSCAN clusters found with current hyperparameters.")


def run_phase_q6(
    interpreter: SAEInterpreter,
    model_wrapper: ModelWrapper,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], SAEIntervention]:
    """Run Ronaldo intervention and print baseline vs clamped deltas for Q6."""
    _print_header("Phase 2: Q6 Ronaldo Intervention")

    intervention = SAEIntervention(
        model_wrapper=model_wrapper,
        checkpoint_path=DEFAULT_CHECKPOINT,
        device="cpu",
    )

    if interpreter.activation_matrix is None:
        raise RuntimeError("Interpreter activation_matrix is missing after load().")
    intervention.compute_feature_max_values(interpreter.activation_matrix)

    prompt = "Who is the better football player, Ronaldo or Messi?"
    candidate_tokens = ["Ronaldo", "Messi", "football", "goal"]

    baseline = intervention.compare_next_token_logprobs(
        prompt,
        candidate_tokens,
    )
    clamped_target = intervention.compare_next_token_logprobs(
        prompt,
        candidate_tokens,
        feature_index=1625,
        clamp_multiplier=8.0,
    )
    clamped_control = intervention.compare_next_token_logprobs(
        prompt,
        candidate_tokens,
        feature_index=0,
        clamp_multiplier=8.0,
    )

    print(f"Prompt: {prompt}")
    print("Token log-prob deltas (target feature vs control feature):")
    for token in candidate_tokens:
        base = baseline.get(token)
        target = clamped_target.get(token)
        control = clamped_control.get(token)
        if base is None or target is None or control is None:
            print(f"  {token:8s}: not-scored (likely multi-token under tokenizer)")
            continue
        target_delta = target - base
        control_delta = control - base
        specificity = target_delta - control_delta
        print(
            f"  {token:8s}: baseline={base:+.6f} | "
            f"target_delta={target_delta:+.6f} | "
            f"control_delta={control_delta:+.6f} | "
            f"specificity={specificity:+.6f}"
        )

    return baseline, clamped_target, clamped_control, intervention


def run_dtype_audit(
    interpreter: SAEInterpreter,
    model_wrapper: ModelWrapper,
    intervention: SAEIntervention,
) -> None:
    """Print dtype audit report and confirm float16/float32 compatibility."""
    _print_header("Phase 3: Dtype Audit")

    allowed = {torch.float16, torch.float32}

    model_param_dtype = next(model_wrapper.model.parameters()).dtype
    sae_weight_dtype: torch.dtype
    if interpreter.model is None:
        raise RuntimeError("Interpreter model is missing after load().")
    sae_weight_dtype = interpreter.model.encoder_weight.dtype

    activation_dtype: torch.dtype | None = None
    latent_dtype: torch.dtype | None = None
    if interpreter.activation_matrix is not None:
        activation_dtype = interpreter.activation_matrix.dtype
    if interpreter.latents is not None:
        latent_dtype = interpreter.latents.dtype

    hook_original_dtype = (
        intervention.original_activations.dtype
        if intervention.original_activations is not None
        else None
    )
    hook_modified_dtype = (
        intervention.modified_activations.dtype
        if intervention.modified_activations is not None
        else None
    )

    entries: list[tuple[str, torch.dtype | None]] = [
        ("model.param", model_param_dtype),
        ("sae.encoder_weight", sae_weight_dtype),
        ("dataset.activation_matrix", activation_dtype),
        ("interpreter.latents", latent_dtype),
        ("hook.original_activations", hook_original_dtype),
        ("hook.modified_activations", hook_modified_dtype),
    ]

    all_ok = True
    for name, dtype in entries:
        if dtype is None:
            print(f"  {name:28s}: <none>")
            continue
        ok = dtype in allowed
        all_ok = all_ok and ok
        status = "OK" if ok else "BAD"
        print(f"  {name:28s}: {str(dtype):12s} [{status}]")

    if not all_ok:
        raise RuntimeError("Dtype audit failed: found non-float16/float32 tensors.")

    print("\nDtype audit passed: all checked tensors are float16 or float32.")


def main() -> None:
    """Run full pilot verification report for Q4/Q5/Q6."""
    checkpoint_path = DEFAULT_CHECKPOINT
    dataset_path = DEFAULT_DATASET

    _print_header("Pilot Verification Report")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset:    {dataset_path}")
    print(f"Model dir:  {DEFAULT_MODEL_NAME}")

    _validate_default_files(checkpoint_path, dataset_path)

    interpreter = SAEInterpreter(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        device="cpu",
    )
    interpreter.load()
    interpreter.compute_latents()

    run_phase_q4_q5(interpreter)

    model_wrapper = ModelWrapper(model_name=DEFAULT_MODEL_NAME, layer_idx=12)
    _, _, _, intervention = run_phase_q6(interpreter, model_wrapper)

    run_dtype_audit(interpreter, model_wrapper, intervention)

    _print_header("Verification Complete")
    print("All requested pilot checks completed successfully.")


if __name__ == "__main__":
    main()
