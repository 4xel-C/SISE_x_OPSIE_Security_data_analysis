from abc import ABC, abstractmethod
from typing import Literal, cast

import hdbscan
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest

# =============================================================================
# ABSTRACT BASE + CONCRETE CLUSTERERS
# =============================================================================


class BaseClusterer(ABC):
    mode: Literal[
        "cluster", "anomaly"
    ]  # "cluster" for clustering algos, "anomaly" for anomaly detection algos

    @abstractmethod
    def fit_predict(self, X_scaled: NDArray) -> tuple[NDArray, NDArray | None]:
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

    def fit_predict(self, X_scaled: NDArray) -> tuple[NDArray, None]:
        labels = cast(NDArray, KMeans(
            n_clusters=self.n_clusters, random_state=42, n_init="auto"
        ).fit_predict(X_scaled))
        return labels, None


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

    def fit_predict(self, X_scaled: NDArray) -> tuple[NDArray, None]:
        labels = cast(NDArray, AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,  # type: ignore[arg-type]
        ).fit_predict(X_scaled))
        return labels, None


@register_clusterer("hdbscan")
class HDBSCANClusterer(BaseClusterer):
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.mode = "cluster"

    def fit_predict(self, X_scaled: NDArray) -> tuple[NDArray, None]:
        model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size, min_samples=self.min_samples
        )
        labels = cast(NDArray, model.fit_predict(X_scaled))
        return labels, None


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
