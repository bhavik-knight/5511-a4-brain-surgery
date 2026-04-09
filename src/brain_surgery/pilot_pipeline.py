"""Reusable pilot verification pipeline logic.

This module contains the heavy-lifting logic used by the verify-pilot CLI scripts.
"""

import csv
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import torch
from sklearn.cluster import KMeans

from .clustering import ClusteringResult
from .interpret import SAEInterpreter
from .intervention import SAEIntervention
from .model_wrapper import ModelWrapper


def safe_token_text(value: object) -> str:
    """Convert token-like values to printable text."""
    if isinstance(value, str):
        return value
    if value is None:
        return "<none>"
    return str(value)


def pick_dynamic_elbow_k(k_values: list[int], inertias: list[float]) -> int:
    """Pick k where SSE improvement rate slows down significantly."""
    if not k_values:
        raise ValueError("k_values cannot be empty")
    if len(k_values) != len(inertias):
        raise ValueError("k_values and inertias must have the same length")
    if len(k_values) <= 2:
        return int(k_values[0])

    sse = np.asarray(inertias, dtype=np.float64)
    improvements = sse[:-1] - sse[1:]
    improvements = np.clip(improvements, a_min=0.0, a_max=None)

    if len(improvements) <= 1 or float(np.sum(improvements)) == 0.0:
        return int(k_values[0])

    eps = 1e-12
    improvement_ratios = improvements[1:] / np.maximum(improvements[:-1], eps)
    slowdown = 1.0 - improvement_ratios

    mean_slowdown = float(np.mean(slowdown))
    std_slowdown = float(np.std(slowdown))
    dynamic_threshold = max(0.25, mean_slowdown + 0.5 * std_slowdown)

    elbow_index = 0
    for idx, slowdown_value in enumerate(slowdown):
        if float(slowdown_value) >= dynamic_threshold:
            elbow_index = idx + 1
            break
    else:
        elbow_index = int(np.argmax(slowdown)) + 1

    return int(k_values[elbow_index])


def print_elbow_table(
    feature_profiles: np.ndarray,
    *,
    start_k: int,
    step: int,
    max_k: int,
    pick_elbow_k_fn: Callable[[list[int], list[float]], int] | None = None,
) -> tuple[int, list[dict[str, object]]]:
    """Print SSE sweep and return automatically selected elbow k."""
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if start_k <= 1:
        raise ValueError(f"start_k must be >=2, got {start_k}")
    if max_k < start_k:
        raise ValueError(f"max_k must be >= start_k, got {max_k} < {start_k}")

    print("\nElbow diagnostics (Spherical K-Means SSE sweep):")
    print(f"  start_k={start_k}, step={step}, max_k={max_k}")
    print("  k | inertia (SSE)")
    print("  --+----------------")
    k_values = list(range(start_k, max_k + 1, step))
    inertias: list[float] = []
    elbow_records: list[dict[str, object]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(feature_profiles)
        inertia = float(model.inertia_)
        inertias.append(inertia)
        elbow_records.append(
            {
                "k": int(k),
                "inertia": inertia,
                "cluster_centers": model.cluster_centers_.tolist(),
            }
        )
        print(f"  {k:3d} | {inertia:14.4f}")

    pick_fn = pick_elbow_k_fn or pick_dynamic_elbow_k
    best_k = pick_fn(k_values, inertias)
    print(f"\nEstimated elbow k: {best_k}")
    return best_k, elbow_records


def save_top_features_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    """Save top feature rows to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["rank", "feature_index", "max_activation", "top_tokens"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved top features CSV: {csv_path}")


def save_elbow_sweep_json(records: list[dict[str, object]], json_path: Path) -> None:
    """Save elbow sweep records as JSON."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"elbow_sweep": records}, fh, indent=2)
    print(f"Saved elbow sweep JSON: {json_path}")


def save_elbow_plot(
    records: list[dict[str, object]],
    elbow_k: int,
    plot_path: Path,
) -> None:
    """Save elbow plot image."""
    if not records:
        return

    import matplotlib.pyplot as plt

    k_values: list[int] = []
    inertias: list[float] = []
    for row in records:
        k_raw = row.get("k")
        inertia_raw = row.get("inertia")
        if isinstance(k_raw, int) and isinstance(inertia_raw, float):
            k_values.append(k_raw)
            inertias.append(inertia_raw)

    if not k_values:
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o", linewidth=2)
    plt.axvline(x=elbow_k, linestyle="--", linewidth=1.5, color="tab:red")
    plt.title("Spherical K-Means Elbow Plot")
    plt.xlabel("K")
    plt.ylabel("Inertia (SSE)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved elbow plot PNG: {plot_path}")


def normalize_cluster_theme(category: str) -> str:
    """Map category strings to coarse themes."""
    lowered = category.casefold()
    if "club" in lowered or "team" in lowered:
        return "Clubs"
    if "history" in lowered or "historical" in lowered or "era" in lowered:
        return "Historical"
    return "Tactical"


def feature_category_purity(
    interpreter: SAEInterpreter,
    *,
    feature_index: int,
    top_k_rows: int = 25,
) -> tuple[str, float, dict[str, int]]:
    """Compute dominant metadata category and purity for a feature."""
    if interpreter.latents is None or interpreter.metadata is None:
        raise RuntimeError("Interpreter latents/metadata missing for category purity.")

    feature_values = interpreter.latents[:, feature_index]
    top_k = min(top_k_rows, feature_values.shape[0])
    top_indices = torch.topk(feature_values, k=top_k).indices.tolist()

    category_counter: Counter[str] = Counter()
    for row_index in top_indices:
        category = interpreter.metadata[row_index].get("category")
        if isinstance(category, str):
            cleaned = category.strip()
            if cleaned:
                category_counter[cleaned] += 1

    if not category_counter:
        return "unknown", 0.0, {}

    dominant_category, dominant_count = category_counter.most_common(1)[0]
    total = int(sum(category_counter.values()))
    purity = float(dominant_count / total) if total > 0 else 0.0
    return dominant_category, purity, dict(category_counter)


def save_cluster_report(
    *,
    interpreter: SAEInterpreter,
    clustering: ClusteringResult,
    json_path: Path,
    selected_k: int,
) -> None:
    """Save cluster purity/theme report to JSON."""
    cluster_summaries = clustering.get("cluster_summaries", [])
    if not isinstance(cluster_summaries, list):
        raise RuntimeError("cluster_summaries missing from clustering payload")

    report_clusters: list[dict[str, object]] = []
    for summary in cluster_summaries:
        if not isinstance(summary, dict):
            continue
        cluster_id = summary.get("cluster_id")
        feature_indices = summary.get("feature_indices")
        if not isinstance(cluster_id, int) or not isinstance(feature_indices, list):
            continue

        per_feature_category: Counter[str] = Counter()
        per_feature_purity: list[float] = []
        for feature_index in feature_indices:
            if not isinstance(feature_index, int):
                continue
            dominant_category, purity, _ = feature_category_purity(
                interpreter,
                feature_index=feature_index,
            )
            per_feature_category[dominant_category] += 1
            per_feature_purity.append(purity)

        cluster_size = len(feature_indices)
        dominant_cluster_category = "unknown"
        category_purity = 0.0
        if per_feature_category and cluster_size > 0:
            dominant_cluster_category, dominant_count = (
                per_feature_category.most_common(1)[0]
            )
            category_purity = float(dominant_count / cluster_size)

        average_feature_purity = (
            float(np.mean(per_feature_purity)) if per_feature_purity else 0.0
        )
        theme = normalize_cluster_theme(dominant_cluster_category)
        report_clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "dominant_category": dominant_cluster_category,
                "cluster_theme": theme,
                "cluster_cohesion": summary.get("cluster_cohesion"),
                "category_purity": category_purity,
                "average_feature_purity": average_feature_purity,
                "category_votes": dict(per_feature_category),
            }
        )

    payload = {
        "selected_k": int(selected_k),
        "clusters": report_clusters,
        "theme_summary": {
            "Tactical": sum(
                1
                for cluster in report_clusters
                if cluster.get("cluster_theme") == "Tactical"
            ),
            "Historical": sum(
                1
                for cluster in report_clusters
                if cluster.get("cluster_theme") == "Historical"
            ),
            "Clubs": sum(
                1
                for cluster in report_clusters
                if cluster.get("cluster_theme") == "Clubs"
            ),
        },
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved cluster report JSON: {json_path}")


def run_phase_q4_q5(
    interpreter: SAEInterpreter,
    *,
    elbow_start_k: int,
    elbow_step: int,
    elbow_max_k: int,
    top_features_csv_path: Path | None,
    elbow_json_path: Path | None,
    elbow_plot_path: Path | None,
    cluster_report_json_path: Path | None,
    global_census_csv_path: Path | None,
    print_header_fn: Callable[[str], None],
    safe_token_text_fn: Callable[[object], str],
    print_elbow_table_fn: Callable[[np.ndarray], tuple[int, list[dict[str, object]]]]
    | Callable[..., tuple[int, list[dict[str, object]]]],
    save_top_features_csv_fn: Callable[[list[dict[str, object]], Path], None],
    save_elbow_sweep_json_fn: Callable[[list[dict[str, object]], Path], None],
    save_elbow_plot_fn: Callable[[list[dict[str, object]], int, Path], None],
    save_cluster_report_fn: Callable[..., None],
    cluster_features_kmeans_fn: Callable[..., ClusteringResult],
) -> None:
    """Run feature interpretation and clustering summary for Q4/Q5."""
    print_header_fn("Phase 1: Q4/Q5 Interpretability Check")

    top_features = interpreter.rank_features_by_max_activation(top_k=10)
    print("Top 10 features with top 10 activating tokens:")
    print("  rank | feature | max_activation | top_tokens")
    print(
        "  -----+---------+----------------+------------------------------------------"
    )
    top_feature_rows: list[dict[str, object]] = []
    for feature in top_features:
        rank_raw = feature.get("rank")
        feature_index_raw = feature.get("feature_index")
        if not isinstance(rank_raw, int) or not isinstance(feature_index_raw, int):
            continue

        max_value = feature.get("max_feature_value")
        max_value_str = (
            f"{max_value:.6f}" if isinstance(max_value, float) else str(max_value)
        )
        top_examples = interpreter.get_top_examples_for_feature(
            feature_index_raw, top_k=10
        )
        token_texts = [
            safe_token_text_fn(example.get("token_text")) for example in top_examples
        ]
        token_summary = " | ".join(token_texts)
        top_feature_rows.append(
            {
                "rank": rank_raw,
                "feature_index": feature_index_raw,
                "max_activation": max_value_str,
                "top_tokens": token_summary,
            }
        )
        print(
            f"  {rank_raw:4d} | {feature_index_raw:7d} | "
            f"{max_value_str:14s} | {token_summary}"
        )

    if top_features_csv_path is not None:
        save_top_features_csv_fn(top_feature_rows, top_features_csv_path)

    if global_census_csv_path is not None:
        print("\nPhase 1b: Global Feature Census (k=5000)")
        interpreter.export_feature_census(
            output_path=global_census_csv_path,
            k=5000,
            density_threshold=0.2,
        )

    if interpreter.latents is None:
        raise RuntimeError("Interpreter latents missing. Call compute_latents() first.")

    features = torch.nn.functional.normalize(interpreter.latents.T, p=2, dim=1)
    feature_profiles = features.cpu().numpy().astype(np.float64, copy=False)

    table_fn = cast(
        Callable[..., tuple[int, list[dict[str, object]]]],
        print_elbow_table_fn,
    )
    best_k, elbow_records = table_fn(
        feature_profiles,
        start_k=elbow_start_k,
        step=elbow_step,
        max_k=elbow_max_k,
    )
    if elbow_json_path is not None:
        save_elbow_sweep_json_fn(elbow_records, elbow_json_path)
    if elbow_plot_path is not None:
        save_elbow_plot_fn(elbow_records, best_k, elbow_plot_path)

    try:
        clustering = cluster_features_kmeans_fn(
            interpreter,
            num_clusters=best_k,
            random_state=42,
            feature_profiles=feature_profiles,
            backend="auto",
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
    except TypeError:
        clustering = cluster_features_kmeans_fn(
            interpreter,
            num_clusters=best_k,
            random_state=42,
            feature_profiles=feature_profiles,
        )

    print(f"\nSpherical K-Means cluster summary ({best_k} clusters):")
    print(
        "  rep_feature = Centroid-Proximal Feature (closest feature to the "
        "cluster centroid), representing the core semantic direction."
    )
    print("  cluster | size | cohesion | rep_feature | top_10_tokens")
    print("  --------+------+----------+-------------+--------------------------------")
    for summary in clustering["cluster_summaries"]:
        cluster_id = summary["cluster_id"]
        num_features = summary["num_features"]
        representative = summary["representative_feature"]
        cluster_cohesion = summary.get("cluster_cohesion")
        cohesion_str = (
            f"{cluster_cohesion:0.4f}" if isinstance(cluster_cohesion, float) else "n/a"
        )
        top_tokens = summary["representative_tokens"][:10]
        token_summary = " | ".join(top_tokens)
        print(
            f"  {cluster_id:7d} | {num_features:4d} | {cohesion_str:8s} | "
            f"{str(representative):11s} | {token_summary}"
        )

    if cluster_report_json_path is not None:
        save_cluster_report_fn(
            interpreter=interpreter,
            clustering=clustering,
            json_path=cluster_report_json_path,
            selected_k=best_k,
        )


def save_intervention_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    """Save intervention deltas as CSV."""
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


def run_phase_q6(
    interpreter: SAEInterpreter,
    model_wrapper: ModelWrapper,
    *,
    checkpoint_path: Path,
    prompt: str,
    candidate_tokens: list[str] | None,
    target_feature_index: int,
    control_feature_index: int,
    clamp_multiplier: float,
    intervention_csv_path: Path | None,
    print_header_fn: Callable[[str], None],
    save_intervention_csv_fn: Callable[[list[dict[str, object]], Path], None],
    intervention_cls: type[SAEIntervention],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], SAEIntervention]:
    """Run target/control clamping comparison (Q6)."""
    print_header_fn("Phase 2: Q6 Ronaldo Intervention")

    intervention = intervention_cls(
        model_wrapper=model_wrapper,
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    if interpreter.activation_matrix is None:
        raise RuntimeError("Interpreter activation_matrix is missing after load().")
    intervention.compute_feature_max_values(interpreter.activation_matrix)

    if candidate_tokens is None:
        candidate_tokens = ["Ronaldo", "Messi", "football", "goal"]

    baseline = intervention.compare_next_token_logprobs(prompt, candidate_tokens)
    clamped_target = intervention.compare_next_token_logprobs(
        prompt,
        candidate_tokens,
        feature_index=target_feature_index,
        clamp_multiplier=clamp_multiplier,
    )
    clamped_control = intervention.compare_next_token_logprobs(
        prompt,
        candidate_tokens,
        feature_index=control_feature_index,
        clamp_multiplier=clamp_multiplier,
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
        save_intervention_csv_fn(rows, intervention_csv_path)

    return baseline, clamped_target, clamped_control, intervention


def save_metadata_report(interpreter: SAEInterpreter, out_path: Path) -> bool:
    """Save metadata coverage report for auditability."""
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
    category_counts: Counter[str] = Counter()
    for row in metadata:
        for field in required_fields:
            value = row.get(field)
            if field == "tags":
                if isinstance(value, list):
                    coverage[field] += 1
            elif value is not None:
                coverage[field] += 1

        category_value = row.get("category")
        if isinstance(category_value, str) and category_value:
            category_counts[category_value] += 1

    total_rows = len(metadata)
    all_present = all(count > 0 for count in coverage.values()) and total_rows > 0

    payload = {
        "total_rows": total_rows,
        "required_fields": required_fields,
        "field_non_null_counts": coverage,
        "category_activation_counts": dict(category_counts),
        "all_required_fields_present": all_present,
        "sample_rows": metadata[:10],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved metadata audit JSON: {out_path}")
    return all_present


def run_dtype_audit(
    interpreter: SAEInterpreter,
    model_wrapper: ModelWrapper,
    intervention: SAEIntervention,
    *,
    metadata_json_path: Path,
    print_header_fn: Callable[[str], None],
    save_metadata_report_fn: Callable[[SAEInterpreter, Path], bool],
) -> None:
    """Run dtype checks and metadata audit."""
    print_header_fn("Phase 3: Dtype Audit")

    allowed = {torch.float16, torch.bfloat16, torch.float32}
    model_param_dtype = next(model_wrapper.model.parameters()).dtype
    if interpreter.model is None:
        raise RuntimeError("Interpreter model is missing after load().")
    sae_weight_dtype = interpreter.model.encoder_weight.dtype

    activation_dtype = (
        interpreter.activation_matrix.dtype
        if interpreter.activation_matrix is not None
        else None
    )
    latent_dtype = (
        interpreter.latents.dtype if interpreter.latents is not None else None
    )
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
        raise RuntimeError(
            "Dtype audit failed: found tensors outside float16/bfloat16/float32."
        )

    metadata_ok = save_metadata_report_fn(interpreter, metadata_json_path)
    if not metadata_ok:
        print(
            "WARNING: Metadata audit incomplete: required rich metadata fields "
            "are missing. Continuing for smoke-test compatibility."
        )

    print(
        "\nDtype audit passed: all checked tensors are float16, bfloat16, or float32."
    )
