from abc import ABC, abstractmethod
from typing import Literal, cast

import hdbscan
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# =============================================================================
# ABSTRACT BASE + CONCRETE CLUSTERERS
# =============================================================================


class BaseClusterer(ABC):
    mode: Literal[
        "cluster", "anomaly"
    ]  # "cluster" for clustering algos, "anomaly" for anomaly detection algos

    @abstractmethod
    def fit_predict(self, X_scaled: NDArray) -> tuple[NDArray, NDArray | float]:
        """Return (labels, anomaly_scores | None)."""


# Registry: algorithm name → BaseClusterer subclass
clusterers_registry: dict[str, type[BaseClusterer]] = {}


def get_available_clusterers() -> list[str]:
    """Return the list of available clusterer names."""
    return list(clusterers_registry.keys())


def register_clusterer(name: str):
    """Decorator to register a clusterer class in the clusterers dict."""

    def decorator(cls: type[BaseClusterer]) -> type[BaseClusterer]:
        if not issubclass(cls, BaseClusterer):
            raise ValueError("Clusterer must inherit from BaseClusterer")

        clusterers_registry[name] = cls
        return cls

    return decorator


@register_clusterer("kmeans")
class KMeansClusterer(BaseClusterer):
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.mode = "cluster"

    def fit_predict(self, X_scaled: NDArray) -> tuple[NDArray, float]:
        model = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42, 
            n_init="auto"
        )

        labels = model.fit_predict(X_scaled)
        inertia = model.inertia_

        return labels, inertia


@register_clusterer("agglomerative")
class AgglomerativeClusterer(BaseClusterer):
    def __init__(
        self,
        n_clusters: int = 3,
        linkage: Literal["ward", "complete", "average", "single"] = "ward",
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.mode = "cluster"
        self.linkage_matrix = None

    def _sklearn_to_linkage(self, model: AgglomerativeClustering) -> np.ndarray:
        """
        Convert a fitted AgglomerativeClustering model to a SciPy linkage matrix.
        """
        n_samples = len(model.labels_)
        counts = np.zeros(model.children_.shape[0])

        for i, (left, right) in enumerate(model.children_):
            count_left = 1 if left < n_samples else counts[left - n_samples]
            count_right = 1 if right < n_samples else counts[right - n_samples]
            counts[i] = count_left + count_right

        linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
        return linkage_matrix

    def fit_predict(self, X_scaled: NDArray) -> tuple[NDArray, float]:
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,  # type: ignore[arg-type]
            compute_distances=True
        )
        labels = model.fit_predict(X_scaled)

        self.linkage_matrix = self._sklearn_to_linkage(model)

        inertia = sum(
            np.sum((members - members.mean(axis=0)) ** 2)
            for members in (X_scaled[labels == k] for k in np.unique(labels))
        )
        return labels, inertia


@register_clusterer("hdbscan")
class HDBSCANClusterer(BaseClusterer):
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.mode = "cluster"

    def fit_predict(self, X_scaled: NDArray) -> tuple[NDArray, float]:
        labels = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples
        ).fit_predict(X_scaled)

        cluster_ids = [k for k in np.unique(labels) if k != -1]
        inertia = sum(
            np.sum((members - members.mean(axis=0)) ** 2)
            for members in (X_scaled[labels == k] for k in cluster_ids)
        )

        return labels, inertia


@register_clusterer("isolation_forest")
class IsolationForestClusterer(BaseClusterer):
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.mode = "anomaly"

    def fit_predict(self, X_scaled: NDArray) -> tuple[NDArray, NDArray]:
        model = IsolationForest(contamination=self.contamination, random_state=42)
        model.fit(X_scaled)
        # decision_function: negative = anomaly, positive = normal
        # We invert so that higher score = more anomalous
        anomaly_scores = -model.decision_function(X_scaled)
        raw_labels = model.predict(X_scaled)  # 1 = normal, -1 = anomaly
        labels = np.where(raw_labels == -1, -1, 0)
        return labels, anomaly_scores


@register_clusterer("lof")
class LOFClusterer(BaseClusterer):
    """Local Outlier Factor — density-based outlier detection.

    LOF compares the local density of each point to its neighbours.
    Points in low-density areas relative to their neighbours get a high
    LOF score (> 1) and are flagged as outliers.
    anomaly_score = LOF score (higher = more anomalous).
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.mode = "anomaly"

    def fit_predict(self, X_scaled: NDArray) -> tuple[NDArray, NDArray]:
        model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
        )
        raw_labels = model.fit_predict(X_scaled)   # 1 = normal, -1 = outlier
        # negative_outlier_factor_ is ≤ 0; invert and shift so outliers > 0
        anomaly_scores = -model.negative_outlier_factor_
        labels = np.where(raw_labels == -1, -1, 0)
        return labels, anomaly_scores
