"""Pilot verification script for Q4/Q5/Q6 with dtype audit.

Run with:
    uv run scripts/verify_pilot.py
"""

import argparse
import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from brain_surgery.clustering import cluster_features_kmeans
from brain_surgery.interpret import SAEInterpreter
from brain_surgery.intervention import SAEIntervention
from brain_surgery.model_wrapper import ModelWrapper
from brain_surgery.utils import (
    ACTIVATIONS_DIR,
    CHECKPOINTS_DIR,
    DEFAULT_MODEL_NAME,
    create_run_output_dirs,
    generate_run_id,
)

DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "sae_checkpoint.pt"
DEFAULT_DATASET = ACTIVATIONS_DIR / "soccer_activations_dataset.pt"
DEFAULT_ELBOW_START_K = 4
DEFAULT_ELBOW_STEP = 4
DEFAULT_ELBOW_MAX_K = 40


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pilot verification with run-scoped output artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to SAE checkpoint (.pt).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to activation dataset (.pt).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. If omitted, generates run_YYYYMMDD_HHMM.",
    )
    parser.add_argument(
        "--elbow-start-k",
        type=int,
        default=DEFAULT_ELBOW_START_K,
        help="Starting k for elbow sweep.",
    )
    parser.add_argument(
        "--elbow-step",
        type=int,
        default=DEFAULT_ELBOW_STEP,
        help="Step size for elbow sweep.",
    )
    parser.add_argument(
        "--elbow-max-k",
        type=int,
        default=DEFAULT_ELBOW_MAX_K,
        help="Maximum k for elbow sweep.",
    )
    return parser.parse_args()


def run_phase_q4_q5(
    interpreter: SAEInterpreter,
    *,
    elbow_start_k: int = DEFAULT_ELBOW_START_K,
    elbow_step: int = DEFAULT_ELBOW_STEP,
    elbow_max_k: int = DEFAULT_ELBOW_MAX_K,
    elbow_csv_path: Path | None = None,
    dbscan_json_path: Path | None = None,
) -> None:
    """Run feature interpretation and clustering summary for Q4/Q5."""
    _print_header("Phase 1: Q4/Q5 Interpretability Check")

    top_features = interpreter.rank_features_by_max_activation(top_k=10)
    print("Top 10 features with top 10 activating tokens:")
    print("  rank | feature | max_activation | top_tokens")
    print(
        "  -----+---------+----------------+------------------------------------------"
    )
    for feature in top_features:
        rank_raw = feature.get("rank")
        feature_index_raw = feature.get("feature_index")
        if not isinstance(rank_raw, int) or not isinstance(feature_index_raw, int):
            continue
        rank = rank_raw
        feature_index = feature_index_raw
        max_value = feature.get("max_feature_value")
        max_value_str = (
            f"{max_value:.6f}" if isinstance(max_value, float) else str(max_value)
        )

        top_examples = interpreter.get_top_examples_for_feature(feature_index, top_k=10)
        token_texts = [
            _safe_token_text(example.get("token_text")) for example in top_examples
        ]
        token_summary = " | ".join(token_texts)
        print(
            f"  {rank:4d} | {feature_index:7d} | {max_value_str:14s} | {token_summary}"
        )

    if interpreter.latents is None:
        raise RuntimeError("Interpreter latents missing. Call compute_latents() first.")

    feature_profiles = interpreter.latents.T.cpu().numpy()
    best_k, elbow_rows = _print_elbow_table(
        feature_profiles,
        start_k=elbow_start_k,
        step=elbow_step,
        max_k=elbow_max_k,
    )
    if elbow_csv_path is not None:
        _save_elbow_table_csv(elbow_rows, elbow_csv_path)

    dbscan_summary = _print_dbscan_summary(interpreter, feature_profiles, best_k)
    if dbscan_json_path is not None:
        _save_dbscan_summary_json(dbscan_summary, dbscan_json_path)

    clustering = cluster_features_kmeans(
        interpreter,
        num_clusters=best_k,
        random_state=42,
    )
    print(f"\nK-Means cluster summary ({best_k} clusters):")
    for summary in clustering["cluster_summaries"]:
        cluster_id = summary["cluster_id"]
        num_features = summary["num_features"]
        representative = summary["representative_feature"]
        top_tokens = summary["representative_tokens"][:3]
        print(
            f"  Cluster {cluster_id:2d} | size={num_features:4d} | "
            f"rep_feature={representative} | tokens={top_tokens}"
        )


def _print_elbow_table(
    feature_profiles: np.ndarray,
    *,
    start_k: int,
    step: int,
    max_k: int,
) -> tuple[int, list[tuple[int, float]]]:
    """Print SSE sweep and return automatically selected elbow k."""
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if start_k <= 1:
        raise ValueError(f"start_k must be >=2, got {start_k}")
    if max_k < start_k:
        raise ValueError(f"max_k must be >= start_k, got {max_k} < {start_k}")

    print("\nElbow diagnostics (K-Means SSE sweep):")
    print(f"  start_k={start_k}, step={step}, max_k={max_k}")
    print("  k | inertia (SSE)")
    print("  --+----------------")
    k_values = list(range(start_k, max_k + 1, step))
    inertias: list[float] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(feature_profiles)
        inertia = float(model.inertia_)
        inertias.append(inertia)
        print(f"  {k:3d} | {inertia:14.4f}")

    # Elbow heuristic: pick the point with max perpendicular distance to
    # the line connecting first and last SSE points.
    x = np.asarray(k_values, dtype=np.float64)
    y = np.asarray(inertias, dtype=np.float64)
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    denominator = np.hypot(y2 - y1, x2 - x1)
    if denominator == 0:
        best_k = k_values[0]
    else:
        distances = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        distances = distances / denominator
        best_idx = int(np.argmax(distances))
        best_k = int(k_values[best_idx])

    print(f"\nEstimated elbow k: {best_k}")
    elbow_rows = list(zip(k_values, inertias))
    return best_k, elbow_rows


def _save_elbow_table_csv(rows: list[tuple[int, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["k", "inertia_sse"])
        for k, inertia in rows:
            writer.writerow([k, f"{inertia:.6f}"])
    print(f"Saved elbow table CSV: {csv_path}")


def _print_dbscan_summary(
    interpreter: SAEInterpreter,
    feature_profiles: np.ndarray,
    elbow_k: int,
) -> dict[str, object]:
    """Run DBSCAN, tuned from elbow_k, and print top-cluster token summaries."""
    print("\nDBSCAN secondary pass:")

    n_samples = feature_profiles.shape[0]
    if n_samples < 2:
        print("  Not enough features to run DBSCAN.")
        return {
            "elbow_k": elbow_k,
            "eps": None,
            "clusters_found": 0,
            "noise_features": 0,
            "top_clusters": [],
        }

    n_neighbors = max(2, min(elbow_k, n_samples - 1))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(feature_profiles)
    distances, _ = neighbors.kneighbors(feature_profiles)
    eps = float(np.percentile(distances[:, -1], 90))

    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(feature_profiles)

    noise_count = int(np.sum(labels == -1))
    unique_labels = sorted(int(label) for label in np.unique(labels) if label != -1)
    print(
        "  "
        f"clusters_found={len(unique_labels)} | "
        f"noise_features={noise_count} | "
        f"eps={eps:.6f} | tuned_from_elbow_k={elbow_k}"
    )
    top_clusters: list[dict[str, object]] = []
    if unique_labels:
        cluster_sizes = [
            (label, int(np.sum(labels == label))) for label in unique_labels
        ]
        cluster_sizes.sort(key=lambda item: item[1], reverse=True)

        print("\n  Top 10 DBSCAN clusters with 10 representative tokens:")
        print("  cluster | size | tokens")
        print("  --------+------+-----------------------------------------------")
        for label, size in cluster_sizes[:10]:
            feature_indices = np.where(labels == label)[0].tolist()
            token_counter: Counter[str] = Counter()
            for feature_index in feature_indices:
                examples = interpreter.get_top_examples_for_feature(
                    feature_index=feature_index,
                    top_k=10,
                )
                for item in examples:
                    token_text = _safe_token_text(item.get("token_text")).strip()
                    if token_text:
                        token_counter[token_text] += 1
            top_tokens = [token for token, _ in token_counter.most_common(10)]
            top_clusters.append(
                {
                    "cluster_id": int(label),
                    "size": int(size),
                    "top_tokens": top_tokens,
                }
            )
            print(f"  {label:7d} | {size:4d} | {' | '.join(top_tokens)}")
    else:
        print("  No dense DBSCAN clusters found with current hyperparameters.")

    return {
        "elbow_k": int(elbow_k),
        "eps": float(eps),
        "clusters_found": int(len(unique_labels)),
        "noise_features": int(noise_count),
        "top_clusters": top_clusters,
    }


def _save_dbscan_summary_json(summary: dict[str, object], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved DBSCAN summary JSON: {json_path}")


def run_phase_q6(
    interpreter: SAEInterpreter,
    model_wrapper: ModelWrapper,
    *,
    intervention_csv_path: Path | None = None,
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
    rows: list[dict[str, object]] = []
    for token in candidate_tokens:
        base = baseline.get(token)
        target = clamped_target.get(token)
        control = clamped_control.get(token)
        if base is None or target is None or control is None:
            print(f"  {token:8s}: not-scored (likely multi-token under tokenizer)")
            rows.append(
                {
                    "token": token,
                    "baseline": None,
                    "target_delta": None,
                    "control_delta": None,
                    "specificity": None,
                }
            )
            continue
        target_delta = target - base
        control_delta = control - base
        specificity = target_delta - control_delta
        rows.append(
            {
                "token": token,
                "baseline": float(base),
                "target_delta": float(target_delta),
                "control_delta": float(control_delta),
                "specificity": float(specificity),
            }
        )
        print(
            f"  {token:8s}: baseline={base:+.6f} | "
            f"target_delta={target_delta:+.6f} | "
            f"control_delta={control_delta:+.6f} | "
            f"specificity={specificity:+.6f}"
        )

    if intervention_csv_path is not None:
        _save_intervention_csv(rows, intervention_csv_path)

    return baseline, clamped_target, clamped_control, intervention


def _save_intervention_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "token",
            "baseline",
            "target_delta",
            "control_delta",
            "specificity",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved intervention deltas CSV: {csv_path}")


def run_dtype_audit(
    interpreter: SAEInterpreter,
    model_wrapper: ModelWrapper,
    intervention: SAEIntervention,
    *,
    metadata_json_path: Path,
) -> None:
    """Print dtype audit and confirm rich metadata report export."""
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

    metadata_ok = _save_metadata_report(interpreter, metadata_json_path)
    if not metadata_ok:
        raise RuntimeError(
            "Metadata audit failed: required rich metadata fields are missing."
        )

    print("\nDtype audit passed: all checked tensors are float16 or float32.")


def _save_metadata_report(interpreter: SAEInterpreter, out_path: Path) -> bool:
    """Save metadata audit report and validate rich schema fields."""
    required_fields = [
        "category",
        "subcategory",
        "topic",
        "tags",
        "era",
        "region",
    ]

    metadata = interpreter.metadata or []
    coverage: dict[str, int] = {field: 0 for field in required_fields}
    for row in metadata:
        for field in required_fields:
            value = row.get(field)
            if field == "tags":
                if isinstance(value, list):
                    coverage[field] += 1
            elif value is not None:
                coverage[field] += 1

    total_rows = len(metadata)
    all_present = all(count > 0 for count in coverage.values()) and total_rows > 0

    sample_rows = metadata[:10]
    payload = {
        "total_rows": total_rows,
        "required_fields": required_fields,
        "field_non_null_counts": coverage,
        "all_required_fields_present": all_present,
        "sample_rows": sample_rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved metadata audit JSON: {out_path}")
    return all_present


def main() -> None:
    """Run full pilot verification report for Q4/Q5/Q6."""
    args = _parse_args()
    checkpoint_path = args.checkpoint
    dataset_path = args.dataset
    run_id = args.run_id or generate_run_id()
    run_dirs = create_run_output_dirs(run_id)
    elbow_csv_path = run_dirs["clusters"] / "elbow_sse.csv"
    dbscan_json_path = run_dirs["clusters"] / "dbscan_summary.json"
    intervention_csv_path = run_dirs["interventions"] / "logprob_deltas.csv"
    metadata_json_path = run_dirs["root"] / "metadata.json"

    _print_header("Pilot Verification Report")
    print(f"Run ID:     {run_id}")
    print(f"Run dir:    {run_dirs['root']}")
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

    run_phase_q4_q5(
        interpreter,
        elbow_start_k=args.elbow_start_k,
        elbow_step=args.elbow_step,
        elbow_max_k=args.elbow_max_k,
        elbow_csv_path=elbow_csv_path,
        dbscan_json_path=dbscan_json_path,
    )

    model_wrapper = ModelWrapper(model_name=DEFAULT_MODEL_NAME, layer_idx=12)
    _, _, _, intervention = run_phase_q6(
        interpreter,
        model_wrapper,
        intervention_csv_path=intervention_csv_path,
    )

    run_dtype_audit(
        interpreter,
        model_wrapper,
        intervention,
        metadata_json_path=metadata_json_path,
    )

    _print_header("Verification Complete")
    print("All requested pilot checks completed successfully.")


if __name__ == "__main__":
    main()
