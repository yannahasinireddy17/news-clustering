"""Similarity-based clustering and cluster quality evaluation."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


def cluster_features(features, distance_threshold=0.4):
    if features is None or features.shape[0] == 0:
        return np.array([], dtype=int)
    if features.shape[0] == 1:
        return np.array([0], dtype=int)
    if hasattr(features, "toarray"):
        features = features.toarray()
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(features)


def cluster_kmeans(features, n_clusters=5, random_state=42):
    """Return K-Means labels for callers that choose a fixed cluster count."""
    if features is None or features.shape[0] == 0:
        return np.array([], dtype=int)
    cluster_count = min(max(1, n_clusters), features.shape[0])
    return KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10).fit_predict(features)


def evaluate_clusters(features, labels):
    """Return silhouette score when at least two valid groups exist."""
    unique_labels = set(labels.tolist()) if hasattr(labels, "tolist") else set(labels)
    if features is None or len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {"silhouette_score": None, "davies_bouldin_score": None}
    dense_features = features.toarray() if hasattr(features, "toarray") else features
    return {
        "silhouette_score": round(float(silhouette_score(features, labels, metric="cosine")), 3),
        "davies_bouldin_score": round(float(davies_bouldin_score(dense_features, labels)), 3),
    }