"""
Clustering service — ML pipeline: scale → reduce → cluster.

Supported algorithms: KMeans, CAH (Agglomerative), HDBSCAN, Isolation Forest.
Supported reducers: PCA (3D), UMAP (3D).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.base import BaseEstimator
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
    projection_plot: DataFrame  # [ipsrc, pc1, pc2, pc3, cluster_label, anomaly_score, access_nbr, deny_rate, requests_per_second]
    corr_plot: DataFrame
    mode: Literal["cluster", "anomaly"]
    reducer: Literal["pca", "umap"]
    algorithm: str
    n_clusters_found: int
    cluster_statistics: DataFrame
    projection_statistics: DataFrame
    inertia: float
    linkage: Optional[np.ndarray]

@dataclass
class ReduceResult:
    X_reduce: NDArray
    loadings_corr: NDArray
    variables: list


# =============================================================================
# REDUCER FUNCTIONS
# =============================================================================


def reduce_pca(X_scaled: NDArray) -> tuple[NDArray, NDArray]:
    reducer = PCA(n_components=3, random_state=42).fit(X_scaled)
    X_reduce = reducer.fit_transform(X_scaled)
    loadings = reducer.components_.T
    loadings_corr = loadings * np.sqrt(reducer.explained_variance_)
    return X_reduce, loadings_corr


def reduce_umap(X_scaled: NDArray) -> tuple[NDArray, NDArray]:
    reducer_model = UMAP(n_components=3, random_state=42)
    X_reduce: np.ndarray = reducer_model.fit_transform(X_scaled) #type: ignore
    return X_reduce, None


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
        X_reduce, loadings_corr = self._reduce(X_scaled, reducer)
        labels, score = clusterer.fit_predict(X_scaled)

        mode: Literal["cluster", "anomaly"] = clusterer.mode

        algorithm = type(clusterer).__name__.replace("Clusterer", "")

        projection_plot = self._build_projection_plot_df(
            ipsrc_index=ipsrc_index,
            X_reduced=X_reduce,
            labels=labels,
            anomaly_scores=score if mode == "anomaly" else None, #type: ignore
            df=df,
        )

        corr_plot = self._build_corr_plot_df(
            loadings_corr=loadings_corr,
        )

        n_clusters_found = int(len(set(labels)) - (1 if -1 in labels else 0))

        cluster_stats = self._compute_cluster_stats(projection_plot)

        return ClusteringResult(
            projection_plot=projection_plot,
            corr_plot=corr_plot,
            mode=mode,
            reducer=reducer,
            algorithm=algorithm,
            n_clusters_found=n_clusters_found,
            inertia=score if mode == "cluster" else None, #type: ignore
            linkage=clusterer.linkage_matrix if algorithm == "Agglomerative" else None, #type: ignore
            statistics=cluster_stats
        )

    def _extract_and_scale(self, df: DataFrame) -> tuple[NDArray, list]:
        df_reset = df.reset_index() if df.index.name == "ipsrc" else df
        ipsrc_index = df_reset["ipsrc"].tolist()
        X = df_reset[CLUSTERING_FEATURES].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)
        return X_scaled, ipsrc_index

    def _reduce(self, X_scaled: NDArray, reducer: Literal["pca", "umap"]) -> tuple[NDArray, NDArray]:
        if reducer == "umap":
            return reduce_umap(X_scaled)
        return reduce_pca(X_scaled)

    def _build_projection_plot_df(
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
    
    def _build_corr_plot_df(self, loadings_corr: NDArray) -> DataFrame:
        loadings_corr_df = (
        DataFrame(
                loadings_corr,
                columns=["PC1", "PC2", "PC3"]
            )
            .assign(variable=CLUSTERING_FEATURES)
        )
        return loadings_corr_df
    
    def _compute_cluster_stats(self, df_plot: DataFrame) -> DataFrame:
        """
        Create statistics on each clusters variables (count, median, min, max)

        Args:
            df_plot (DataFrame): Dataframe create with `_build_plot_df()`

        Returns:
            DataFrame: Statistics
        """
        stats = (
            df_plot.groupby('cluster_str')
                .agg({
                    'ipsrc': 'count',
                    'access_nbr': ['mean', 'median', 'max', 'min'],
                    'deny_rate': ['mean', 'median', 'max', 'min'],
                    'requests_per_second': ['mean', 'median', 'max', 'min']
                })
        )
        return stats

    def get_available_algorithms(self) -> list[str]:
        """Return the list of available clustering algorithm names."""
        return get_available_clusterers()

    def select_algorithm(self, name: str, **kwargs) -> BaseClusterer:
        """Factory to get a clusterer instance by name."""
        clusterer_class = clusterers_registry.get(name)

        if clusterer_class is None:
            raise ValueError(f"Unknown clustering algorithm: {name}")

        return clusterer_class(**kwargs)
