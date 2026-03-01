from typing import Literal

import streamlit as st

from services.charts import scatter_2d_clusters, scatter_3d_clusters
from services.clustering_service import ClusteringResult, ClusteringService

st.title("Clustering")

df = st.session_state.data.df
service = ClusteringService()

# =============================================================================
# SIDEBAR — ALGORITHM & PARAMS
# =============================================================================
st.sidebar.header("Paramètres")

ALGORITHM_LABELS: dict[str, str] = {
    "kmeans": "KMeans",
    "agglomerative": "CAH",
    "hdbscan": "HDBSCAN",
    "isolation_forest": "Isolation Forest",
}

available = service.get_available_algorithms()
algorithm_key = st.sidebar.selectbox(
    "Algorithme",
    options=available,
    format_func=lambda key: ALGORITHM_LABELS.get(key, key),
)

kwargs: dict = {}

if algorithm_key == "kmeans":
    kwargs["n_clusters"] = st.sidebar.slider(
        "Nombre de clusters", min_value=2, max_value=20, value=3
    )

elif algorithm_key == "agglomerative":
    kwargs["n_clusters"] = st.sidebar.slider(
        "Nombre de clusters", min_value=2, max_value=20, value=3
    )
    kwargs["linkage"] = st.sidebar.selectbox(
        "Linkage", options=["ward", "complete", "average", "single"]
    )

elif algorithm_key == "hdbscan":
    kwargs["min_cluster_size"] = st.sidebar.slider(
        "min_cluster_size", min_value=2, max_value=50, value=5
    )
    kwargs["min_samples"] = st.sidebar.slider(
        "min_samples", min_value=1, max_value=50, value=3
    )

elif algorithm_key == "isolation_forest":
    kwargs["contamination"] = st.sidebar.slider(
        "Contamination", min_value=0.01, max_value=0.50, value=0.05, step=0.01
    )

reducer: Literal["pca", "umap"] = st.sidebar.radio(  # type: ignore[assignment]
    "Réduction de dimension", options=["pca", "umap"]
)

# =============================================================================
# CLUSTERING PIPELINE (cached)
# =============================================================================


def run_clustering(df, algorithm_key, kwargs, reducer) -> ClusteringResult:
    clusterer = service.select_algorithm(algorithm_key, **kwargs)
    return service.run(df, clusterer, reducer)


with st.spinner("Calcul en cours…"):
    result = run_clustering(df, algorithm_key, kwargs, reducer)

# =============================================================================
# METRICS ROW
# =============================================================================
col1, col2, col3 = st.columns(3)

col1.metric("IPs analysées", len(result.df_plot))
col2.metric("Algorithme", result.algorithm)

if result.mode == "cluster":
    col3.metric("Clusters trouvés", result.n_clusters_found)
else:
    n_anomalies = int((result.df_plot["cluster_label"] == -1).sum())
    col3.metric("Anomalies détectées", n_anomalies)

# =============================================================================
# CHART
# =============================================================================
view = st.radio("Vue", options=["3D", "2D"], horizontal=True)

if view == "3D":
    st.plotly_chart(scatter_3d_clusters(result), width="stretch")
else:
    st.plotly_chart(scatter_2d_clusters(result), width="stretch")

# =============================================================================
# RAW DATA EXPANDER
# =============================================================================
with st.expander("Données brutes"):
    st.dataframe(result.df_plot, width="stretch")
