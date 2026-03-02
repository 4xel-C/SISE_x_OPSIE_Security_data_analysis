from typing import Literal

import streamlit as st
from streamlit.components.v1 import html

from services.charts import scatter_2d_clusters, scatter_3d_clusters, dendrogram, line_cluster_inertia
from services.clustering_service import ClusteringResult, ClusteringService
from services.data_manager import DataManager


st.title("Clustering")

clustering_service = ClusteringService()
data_manager: DataManager = st.session_state.data
df = data_manager.df

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

def handle_model_change():
    st.session_state.inertia_hist = None

available = clustering_service.get_available_algorithms()
algorithm_key = st.sidebar.selectbox(
    "Algorithme",
    options=available,
    format_func=lambda key: ALGORITHM_LABELS.get(key, key),
    on_change=handle_model_change
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
    clusterer = clustering_service.select_algorithm(algorithm_key, **kwargs)
    return clustering_service.run(df, clusterer, reducer)


with st.spinner("Calcul en cours…"):
    result = run_clustering(df, algorithm_key, kwargs, reducer)

# =============================================================================
# METRICS ROW
# =============================================================================
col1, col2, col3 = st.columns(3, border=True)

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
col1, col2 = st.columns(2, border=True)

with col1:
    tab_labels = ["Projection 3D", "Projection 2D"]
    if algorithm_key == "agglomerative":
        tab_labels.append("Dendrogramme")
    tab_labels.append("Descriptions")

    tabs = st.tabs(tab_labels, width="stretch", default="Projection 3D")

    if algorithm_key == "agglomerative":
        tab_3d, tab_2d, tab_dendrogram, description = tabs

        with tab_dendrogram:
            fig = dendrogram(result)
            st.plotly_chart(fig, width="stretch", height=500)
    else:
        tab_3d, tab_2d, description = tabs

    with tab_3d:
        fig = scatter_3d_clusters(result)
        st.plotly_chart(fig, width="stretch", height=500)
    with tab_2d:
        fig = scatter_2d_clusters(result)
        st.plotly_chart(fig, width="stretch", height=500)
    with description:
        #TODO: call llm here
        st.write(result.statistics)


# =============================================================================
# RAW DATA INPUT
# =============================================================================
with col2:
    ipsrc = st.text_input(
        "Donnée brute", 
        placeholder="Rechercher un ipsrc", 
        icon=":material/search:", 
        autocomplete="off",
        label_visibility="collapsed"
    )

    result = data_manager.search_ipsrc(ipsrc)
    st.dataframe(result, width="stretch", height=500)


# =============================================================================
# CLUSTER INERTIA EXPANDER
# =============================================================================

if algorithm_key in ["kmeans", "agglomerative"]:

    # Initialize session state
    if "inertia_hist" not in st.session_state:
        st.session_state.inertia_hist = []

    if "current_k" not in st.session_state:
        st.session_state.current_k = 0

    if "running" not in st.session_state:
        st.session_state.running = False

    with st.expander("Inertie des clusters", expanded=True):

        @st.fragment
        def inertia_fragment():
            # Draw current graph
            if st.session_state.inertia_hist:
                graph = line_cluster_inertia(st.session_state.inertia_hist, total_inertia=10)
                st.plotly_chart(graph, height=400)

            # Continue loop if running
            if st.session_state.running and st.session_state.current_k <= 10:
                analysing_kwarg = kwargs.copy()
                analysing_kwarg["n_clusters"] = st.session_state.current_k + 1

                result = run_clustering(df, algorithm_key, analysing_kwarg, reducer)

                st.session_state.inertia_hist.append(result.inertia)
                st.session_state.current_k += 1

                # Scroll to bottom if first itteration
                html(
                    """
                    <script>
                    const section = window.parent.document.querySelector('section.stMain');
                    section.scrollTo({ top: section.scrollHeight, behavior: 'smooth' });
                    </script>
                    """,
                    height=0
                )

                # Rerun ONLY the fragment
                st.rerun(scope="fragment")

            # Stop condition
            if st.session_state.current_k >= 10:
                st.session_state.running = False

            # Start button
            if st.button("Lancer l'analyse", type="primary", width="stretch"):
                st.session_state.inertia_hist = []
                st.session_state.current_k = 0
                st.session_state.running = True
                st.rerun(scope="fragment")

        # Call the fragment
        inertia_fragment()