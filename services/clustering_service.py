"""
Clustering service — ML pipeline: scale → reduce → cluster.

Supported algorithms: KMeans, CAH (Agglomerative), HDBSCAN, Isolation Forest.
Supported reducers: PCA (3D), UMAP (3D).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from features.clustering import (
    BaseClusterer,
    clusterers_registry,
    get_available_clusterers,
)

# important features for clustering
CLUSTERING_FEATURES: list[str] = [
    "access_nbr",
    "distinct_ipdst",
    "distinct_portdst",
    "permit_nbr",
    "deny_nbr",
    "permit_small_ports_nbr",
    "permit_admin_ports_nbr",
    "deny_rate",
    "unique_dst_ratio",
    "unique_port_ratio",
    "activity_duration_s",
    "requests_per_second",
    "distinct_rules_hit",
    "deny_rules_hit",
    "sensitive_ports_nbr",
    "sensitive_ports_ratio",
]


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class ClusteringResult:
    df_plot: DataFrame  # [ipsrc, pc1, pc2, pc3, cluster_label, anomaly_score, access_nbr, deny_rate, requests_per_second]
    mode: Literal["cluster", "anomaly"]
    reducer: Literal["pca", "umap"]
    algorithm: str
    n_clusters_found: int


# =============================================================================
# REDUCER FUNCTIONS
# =============================================================================


def reduce_pca(X_scaled: NDArray) -> NDArray:
    return PCA(n_components=3, random_state=42).fit_transform(X_scaled)


def reduce_umap(X_scaled: NDArray) -> NDArray:
    return UMAP(n_components=3, random_state=42).fit_transform(X_scaled)  # type: ignore


# =============================================================================
# CLUSTERING SERVICE
# =============================================================================


class ClusteringService:
    def run(
        self,
        df: DataFrame,
        clusterer: BaseClusterer,
        reducer: Literal["pca", "umap"],
    ) -> ClusteringResult:
        X_scaled, ipsrc_index = self._extract_and_scale(df)
        X_reduced = self._reduce(X_scaled, reducer)
        labels, anomaly_scores = clusterer.fit_predict(X_scaled)

        mode: Literal["cluster", "anomaly"] = clusterer.mode

        algorithm = type(clusterer).__name__.replace("Clusterer", "")

        df_plot = self._build_plot_df(
            ipsrc_index=ipsrc_index,
            X_reduced=X_reduced,
            labels=labels,
            anomaly_scores=anomaly_scores,
            df=df,
        )

        n_clusters_found = int(len(set(labels)) - (1 if -1 in labels else 0))

        return ClusteringResult(
            df_plot=df_plot,
            mode=mode,
            reducer=reducer,
            algorithm=algorithm,
            n_clusters_found=n_clusters_found,
        )

    def _extract_and_scale(self, df: DataFrame) -> tuple[NDArray, list]:
        df_reset = df.reset_index() if df.index.name == "ipsrc" else df
        ipsrc_index = df_reset["ipsrc"].tolist()
        X = df_reset[CLUSTERING_FEATURES].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)
        return X_scaled, ipsrc_index

    def _reduce(self, X_scaled: NDArray, reducer: Literal["pca", "umap"]) -> NDArray:
        if reducer == "umap":
            return reduce_umap(X_scaled)
        return reduce_pca(X_scaled)

    def _build_plot_df(
        self,
        ipsrc_index: list,
        X_reduced: NDArray,
        labels: NDArray,
        anomaly_scores: NDArray | None,
        df: DataFrame,
    ) -> DataFrame:
        df_reset = df.reset_index() if df.index.name == "ipsrc" else df

        plot_df = DataFrame(
            {
                "ipsrc": ipsrc_index,
                "pc1": X_reduced[:, 0],
                "pc2": X_reduced[:, 1],
                "pc3": X_reduced[:, 2],
                "cluster_label": labels,
                "anomaly_score": anomaly_scores
                if anomaly_scores is not None
                else np.zeros(len(labels)),
                "access_nbr": df_reset["access_nbr"].values,
                "deny_rate": df_reset["deny_rate"].values,
                "requests_per_second": df_reset["requests_per_second"].values,
            }
        )

        plot_df["cluster_str"] = plot_df["cluster_label"].apply(
            lambda x: "Bruit" if x == -1 else f"Cluster {x}"
        )

        return plot_df

    def get_available_algorithms(self) -> list[str]:
        """Return the list of available clustering algorithm names."""
        return get_available_clusterers()

    def select_algorithm(self, name: str, **kwargs) -> BaseClusterer:
        """Factory to get a clusterer instance by name."""
        clusterer_class = clusterers_registry.get(name)

        if clusterer_class is None:
            raise ValueError(f"Unknown clustering algorithm: {name}")

        return clusterer_class(**kwargs)
