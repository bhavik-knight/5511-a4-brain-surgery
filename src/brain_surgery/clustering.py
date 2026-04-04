"""K-Means clustering for automated feature interpretability (Q5 Bonus).

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


class ClusteringResult(TypedDict):
    """Full K-Means clustering result payload."""

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
    """Cluster latent features using K-Means.

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
        f"Running K-Means clustering on {feature_profiles.shape[0]} features "
        f"into {num_clusters} clusters..."
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
        }

        for feature_idx in features_in_cluster[:5]:
            top_examples = interpreter.get_top_examples_for_feature(
                feature_index=feature_idx, top_k=5
            )
            if top_examples:
                first_value = top_examples[0].get("feature_value")
                if not (isinstance(first_value, float) and first_value > 0):
                    continue
                summary["representative_feature"] = feature_idx
                summary["representative_tokens"] = []
                for item in top_examples:
                    feature_value = item.get("feature_value")
                    token_text = item.get("token_text")
                    if isinstance(feature_value, float) and feature_value > 0:
                        if isinstance(token_text, str):
                            summary["representative_tokens"].append(token_text)
                break

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
    print("K-MEANS CLUSTERING COMPLETE")
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
