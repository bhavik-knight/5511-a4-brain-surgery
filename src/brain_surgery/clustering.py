"""Spherical K-Means clustering for automated feature interpretability (Q5 Bonus).

Groups learned latent features into conceptual neighborhoods to discover
semantic patterns and feature relationships in the latent space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F

from .interpret import SAEInterpreter


@dataclass(frozen=True)
class TorchSphericalKMeans:
    """Minimal sklearn-like payload for torch spherical k-means runs."""

    cluster_centers_: NDArray[np.float64]
    n_clusters: int
    n_iter_: int
    device: str


class ClusterSummary(TypedDict):
    """Human-readable summary for one feature cluster."""

    cluster_id: int
    num_features: int
    feature_indices: list[int]
    representative_feature: int | None
    representative_tokens: list[str]
    cluster_cohesion: float | None


class ClusteringResult(TypedDict):
    """Full Spherical K-Means clustering result payload."""

    cluster_labels: NDArray[np.int_]
    kmeans_model: KMeans | TorchSphericalKMeans
    clusters: dict[int, list[int]]
    cluster_summaries: list[ClusterSummary]


def _l2_normalize_rows_numpy(features: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def _torch_spherical_kmeans(
    features: torch.Tensor,
    *,
    num_clusters: int,
    random_state: int,
    max_iters: int,
) -> tuple[NDArray[np.int_], NDArray[np.float64], int]:
    if features.dim() != 2:
        raise ValueError(f"Expected 2D features, got {tuple(features.shape)}")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"Empty feature matrix: {tuple(features.shape)}")
    if num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {num_clusters}")
    if num_clusters > int(features.shape[0]):
        raise ValueError(
            "num_clusters cannot exceed number of points: "
            f"{num_clusters} > {int(features.shape[0])}"
        )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(random_state))

    x = F.normalize(features, p=2, dim=1)
    n_points = int(x.shape[0])

    init_idx = torch.randperm(n_points, generator=gen)[:num_clusters]
    centers = x[init_idx].clone()
    labels = torch.argmax(x @ centers.T, dim=1)

    prev_labels: torch.Tensor | None = None
    n_iter = 0
    chunk_size = 20000  # Similarity calculation in chunks for VRAM safety

    for step in range(max_iters):
        sims = torch.empty((n_points, num_clusters), device=features.device)

        # Chunked similarity computation to prevent OOM
        for i in range(0, n_points, chunk_size):
            end_i = min(i + chunk_size, n_points)
            sims[i:end_i] = x[i:end_i] @ centers.T

        labels = torch.argmax(sims, dim=1)
        if prev_labels is not None and torch.equal(labels, prev_labels):
            n_iter = step
            break

        new_centers = torch.empty_like(centers)
        for cluster_id in range(num_clusters):
            mask = labels == cluster_id
            if torch.any(mask):
                # Using mean(dim=0) on masks of x - still efficient
                new_centers[cluster_id] = x[mask].mean(dim=0)
            else:
                ridx = torch.randint(0, n_points, (1,), generator=gen)
                new_centers[cluster_id] = x[ridx[0]]

        centers = F.normalize(new_centers, p=2, dim=1)
        prev_labels = labels
        n_iter = step + 1

    labels_np = labels.detach().cpu().numpy().astype(np.int_, copy=False)
    centers_np = centers.detach().cpu().numpy().astype(np.float64, copy=False)
    return labels_np, centers_np, n_iter


def cluster_features_kmeans(
    interpreter: SAEInterpreter,
    num_clusters: int = 10,
    random_state: int = 42,
    feature_profiles: NDArray[np.float64] | None = None,
    backend: Literal["auto", "sklearn", "torch"] = "sklearn",
    device: torch.device | str | None = None,
    max_iters: int = 50,
) -> ClusteringResult:
    """Cluster latent features using Spherical K-Means.

    Groups SAE latent features into conceptual neighborhoods by treating
    each feature as a data point with num_tokens dimensions.

    Args:
        interpreter: SAEInterpreter with loaded model and computed latents.
        num_clusters: Number of clusters to discover. Defaults to 10.
        random_state: Random seed for reproducibility. Defaults to 42.
        feature_profiles: Optional precomputed feature matrix of shape
            (num_features, num_tokens). If omitted, uses interpreter latents.
        backend: Clustering implementation backend.
            - "sklearn": CPU scikit-learn KMeans (default).
            - "torch": PyTorch spherical k-means (GPU-capable).
            - "auto": Use torch when `device` is CUDA, else sklearn.
        device: Device to run torch backend on (e.g., "cuda", "cpu").
            Ignored for the sklearn backend.
        max_iters: Maximum iterations for the torch backend.

    Returns:
        Dictionary with keys:
            - "cluster_labels": Array of cluster assignments per feature
            - "kmeans_model": Fitted KMeans model
            - "clusters": Dictionary mapping cluster_id to feature indices
            - "cluster_summaries": List of dicts with representative features

    Raises:
        RuntimeError: If latents not computed in interpreter.
    """
    if interpreter.latents is None:
        raise RuntimeError(
            "Interpreter must have computed latents. "
            "Call interpreter.compute_latents() first."
        )

    latents = interpreter.latents
    if feature_profiles is None:
        feature_profiles = latents.T.cpu().numpy().astype(np.float64, copy=False)

    if feature_profiles.size == 0:
        raise ValueError(
            f"Empty feature_profiles matrix. Got shape {feature_profiles.shape}."
        )

    feature_profiles = _l2_normalize_rows_numpy(feature_profiles)

    print(
        "Running Spherical K-Means clustering on "
        f"{feature_profiles.shape[0]} features into {num_clusters} clusters..."
    )

    resolved_device = (
        torch.device(device)
        if isinstance(device, str)
        else (device or torch.device("cpu"))
    )
    resolved_backend: Literal["sklearn", "torch"]
    if backend == "auto":
        resolved_backend = "torch" if resolved_device.type == "cuda" else "sklearn"
    else:
        resolved_backend = backend

    kmeans_model: KMeans | TorchSphericalKMeans
    cluster_centers: NDArray[np.float64]
    if resolved_backend == "torch":
        labels_np, centers_np, n_iter = _torch_spherical_kmeans(
            torch.from_numpy(feature_profiles).to(resolved_device),
            num_clusters=num_clusters,
            random_state=random_state,
            max_iters=max_iters,
        )
        cluster_labels = labels_np
        cluster_centers = centers_np
        kmeans_model = TorchSphericalKMeans(
            cluster_centers_=cluster_centers,
            n_clusters=num_clusters,
            n_iter_=n_iter,
            device=str(resolved_device),
        )
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_profiles)
        cluster_centers = kmeans.cluster_centers_.astype(np.float64, copy=False)
        kmeans_model = kmeans

    clusters: dict[int, list[int]] = {}
    for feature_idx, cluster_id in enumerate(cluster_labels):
        clusters.setdefault(int(cluster_id), []).append(feature_idx)

    cluster_summaries: list[ClusterSummary] = []
    for cluster_id in range(num_clusters):
        features_in_cluster = clusters.get(cluster_id, [])
        summary: ClusterSummary = {
            "cluster_id": cluster_id,
            "num_features": len(features_in_cluster),
            "feature_indices": features_in_cluster,
            "representative_feature": None,
            "representative_tokens": [],
            "cluster_cohesion": None,
        }

        if features_in_cluster:
            centroid = cluster_centers[cluster_id]
            centroid_norm = float(np.linalg.norm(centroid))

            cosine_scores: list[tuple[int, float]] = []
            for feature_idx in features_in_cluster:
                feature_vector = feature_profiles[feature_idx]
                feature_norm = float(np.linalg.norm(feature_vector))
                denom = max(1e-12, centroid_norm * feature_norm)
                similarity = float(np.dot(feature_vector, centroid) / denom)
                cosine_scores.append((feature_idx, similarity))

            cosine_scores.sort(key=lambda item: item[1], reverse=True)
            summary["representative_feature"] = int(cosine_scores[0][0])
            summary["cluster_cohesion"] = float(
                np.mean([score for _, score in cosine_scores])
            )

        representative_feature = summary["representative_feature"]
        if representative_feature is not None:
            top_examples = interpreter.get_top_examples_for_feature(
                feature_index=representative_feature,
                top_k=10,
            )
            summary["representative_tokens"] = []
            for item in top_examples:
                token_text = item.get("token_text")
                if isinstance(token_text, str):
                    cleaned = token_text.strip()
                    if cleaned:
                        summary["representative_tokens"].append(cleaned)

            summary["representative_tokens"] = summary["representative_tokens"][:10]

        cluster_summaries.append(summary)

    return {
        "cluster_labels": cluster_labels,
        "kmeans_model": kmeans_model,
        "clusters": clusters,
        "cluster_summaries": cluster_summaries,
    }


def print_cluster_analysis(clustering_result: ClusteringResult) -> None:
    """Print human-readable cluster analysis.

    Args:
        clustering_result: Dictionary from cluster_features_kmeans().
    """
    cluster_summaries = clustering_result["cluster_summaries"]

    print("\n" + "=" * 80)
    print("SPHERICAL K-MEANS CLUSTERING COMPLETE")
    print("=" * 80)
    print("\nAnalyzing discovered conceptual families:\n")

    for summary in cluster_summaries:
        cluster_id = summary["cluster_id"]
        num_features = summary["num_features"]
        rep_feature = summary["representative_feature"]
        rep_tokens = summary["representative_tokens"]

        print(f"=== CLUSTER {cluster_id} ===")
        print(f"Contains {num_features} features.")

        if rep_feature is not None:
            print(f"Representative Feature ID: {rep_feature}")
            if rep_tokens:
                print(f"Top clustered semantic tokens: {', '.join(rep_tokens)}")
        else:
            print("Cluster contains dead features or minimal activations.")

        print("-" * 50 + "\n")
