"""Low-dimensional coordinates for visualizing article clusters."""

import numpy as np
from sklearn.decomposition import PCA


def project_clusters(features, labels, max_points=2000):
    if features is None or features.shape[0] == 0:
        return []
    if features.shape[0] > max_points:
        indices = np.linspace(0, features.shape[0] - 1, max_points, dtype=int)
        features = features[indices]
        labels = np.asarray(labels)[indices]
    dense_features = features.toarray() if hasattr(features, "toarray") else features
    if dense_features.shape[0] == 1:
        coordinates = np.zeros((1, 2))
    else:
        dimensions = min(2, dense_features.shape[0], dense_features.shape[1])
        coordinates = PCA(n_components=dimensions).fit_transform(dense_features)
        if dimensions == 1:
            coordinates = np.column_stack([coordinates[:, 0], np.zeros(len(coordinates))])
    return [
        {"x": round(float(point[0]), 5), "y": round(float(point[1]), 5), "cluster": int(label)}
        for point, label in zip(coordinates, labels)
    ]