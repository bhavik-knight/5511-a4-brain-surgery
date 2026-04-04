"""Spherical K-Means clustering for automated feature interpretability (Q5 Bonus).

Groups learned latent features into conceptual neighborhoods to discover
semantic patterns and feature relationships in the latent space.
"""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from .interpret import SAEInterpreter


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
    kmeans_model: KMeans
    clusters: dict[int, list[int]]
    cluster_summaries: list[ClusterSummary]


def cluster_features_kmeans(
    interpreter: SAEInterpreter,
    num_clusters: int = 10,
    random_state: int = 42,
    feature_profiles: NDArray[np.float64] | None = None,
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

    print(
        "Running Spherical K-Means clustering on "
        f"{feature_profiles.shape[0]} features into {num_clusters} clusters..."
    )

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_profiles)

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
            centroid = kmeans.cluster_centers_[cluster_id]
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
        "kmeans_model": kmeans,
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
