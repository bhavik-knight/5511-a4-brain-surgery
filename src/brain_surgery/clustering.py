"""K-Means clustering for automated feature interpretability (Q5 Bonus).

Groups learned latent features into conceptual neighborhoods to discover
semantic patterns and feature relationships in the latent space.
"""

from __future__ import annotations

from typing import Any

from sklearn.cluster import KMeans

from .interpret import SAEInterpreter


def cluster_features_kmeans(
    interpreter: SAEInterpreter,
    num_clusters: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Cluster latent features using K-Means.

    Groups SAE latent features into conceptual neighborhoods by treating
    each feature as a data point with num_tokens dimensions.

    Args:
        interpreter: SAEInterpreter with loaded model and computed latents.
        num_clusters: Number of clusters to discover. Defaults to 10.
        random_state: Random seed for reproducibility. Defaults to 42.

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
    feature_profiles = latents.T.cpu().numpy()

    print(
        f"Running K-Means clustering on {feature_profiles.shape[0]} features "
        f"into {num_clusters} clusters..."
    )

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_profiles)

    clusters: dict[int, list[int]] = {}
    for feature_idx, cluster_id in enumerate(cluster_labels):
        clusters.setdefault(int(cluster_id), []).append(feature_idx)

    cluster_summaries = []
    for cluster_id in range(num_clusters):
        features_in_cluster = clusters.get(cluster_id, [])
        summary = {
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
            if top_examples and top_examples[0]["feature_value"] > 0:
                summary["representative_feature"] = feature_idx
                summary["representative_tokens"] = [
                    item["token_text"]
                    for item in top_examples
                    if item["feature_value"] > 0
                ]
                break

        cluster_summaries.append(summary)

    return {
        "cluster_labels": cluster_labels,
        "kmeans_model": kmeans,
        "clusters": clusters,
        "cluster_summaries": cluster_summaries,
    }


def print_cluster_analysis(clustering_result: dict[str, Any]) -> None:
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
