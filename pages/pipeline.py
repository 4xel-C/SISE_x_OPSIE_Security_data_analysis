"""
MCP Analysis Pipeline — 4-step sequential analysis.

Session state keys:
  mcp_s1 : Step1Result | None   — analyse descriptive
  mcp_s2 : Step2Result | None   — modèle supervisé
  mcp_s3 : Step3Result | None   — modèle non-supervisé
  mcp_s4 : Step4Result | None   — consolidation
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from services.analysis_pipeline import (
    SUPERVISED_ALGORITHMS,
    UNSUPERVISED_ALGORITHMS,
    generate_report_markdown,
    markdown_to_pdf_bytes,
    suggest_supervised_algorithm,
    suggest_unsupervised_algorithm,
    tool_consolidate,
    tool_descriptive_analysis,
    tool_run_supervised_model,
    tool_run_unsupervised_model,
)
from services.charts import (
    access_distribution,
    deny_rate_distribution,
    horizontal_vs_vertical_scan,
    scatter_3d_clusters,
)

# =============================================================================
# PAGE SETUP
# =============================================================================

st.title("Pipeline — Analyse automatisée")
st.caption(
    "Pipeline en 4 étapes : analyse descriptive → modèle supervisé → modèle non-supervisé → consolidation."
)

# =============================================================================
# SESSION STATE INIT
# =============================================================================

for key in (
    "mcp_s1",
    "mcp_s2",
    "mcp_s3",
    "mcp_s4",
    "mcp_sup_suggestion",
    "mcp_unsup_suggestion",
    "mcp_report_md",
):
    if key not in st.session_state:
        st.session_state[key] = None

# =============================================================================
# DATA
# =============================================================================

df = st.session_state.data.df

# =============================================================================
# PROGRESS & RESET
# =============================================================================

steps_done = sum(
    st.session_state[k] is not None for k in ("mcp_s1", "mcp_s2", "mcp_s3", "mcp_s4")
)

col_prog, col_reset = st.columns([4, 1])
with col_prog:
    st.progress(steps_done / 4, text=f"Étape {steps_done}/4 complétée")
with col_reset:
    if st.button("Réinitialiser", use_container_width=True):
        for key in (
            "mcp_s1",
            "mcp_s2",
            "mcp_s3",
            "mcp_s4",
            "mcp_sup_suggestion",
            "mcp_unsup_suggestion",
            "mcp_report_md",
        ):
            st.session_state[key] = None
        st.rerun()

st.divider()


# =============================================================================
# HELPER
# =============================================================================


def _show_commentary(commentary: str | None, label: str = "Analyse Mistral") -> None:
    with st.expander(f"💬 {label}", expanded=False):
        if commentary:
            st.info(commentary)
        else:
            st.warning(
                "Commentaire Mistral non disponible (clé API absente ou erreur réseau)."
            )


def _optim_chart(curve, score_name: str) -> None:
    if not curve:
        return
    curve_df = pd.DataFrame(
        {"param_value": p.param_value, "score": p.score} for p in curve
    )
    fig = px.line(
        curve_df,
        x="param_value",
        y="score",
        markers=True,
        title=f"Courbe d'optimisation — {score_name}",
        labels={"param_value": curve[0].param_name, "score": score_name},
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# ÉTAPE 1 — Analyse descriptive
# =============================================================================

with st.expander(
    "**Étape 1 — Analyse descriptive**",
    expanded=(st.session_state.mcp_s1 is None),
):
    if st.session_state.mcp_s1 is None:
        if st.button("Lancer l'analyse", type="primary"):
            with st.status("Calcul de l'analyse descriptive…", expanded=True) as status:
                st.write("Extraction des statistiques et corrélations…")
                s1 = tool_descriptive_analysis(df)
                st.session_state.mcp_s1 = s1
                status.update(
                    label="Analyse descriptive terminée",
                    state="complete",
                    expanded=False,
                )
            st.rerun()
    else:
        s1 = st.session_state.mcp_s1

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("IPs analysées", s1.n_ips)
        c2.metric("Features", s1.n_features)
        c3.metric("IPs deny_rate ≥ 80%", s1.high_deny_count)
        c4.metric(
            "Ratio suspect",
            f"{s1.high_deny_count / max(s1.n_ips, 1) * 100:.1f}%",
        )

        st.subheader("Statistiques descriptives")
        st.table(s1.feature_stats)

        st.subheader("Matrice de corrélation")
        fig_corr = px.imshow(
            s1.corr_matrix,
            color_continuous_scale="RdBu_r",
            title="Corrélations entre features",
            aspect="auto",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Top 10 par deny_rate")
            st.dataframe(s1.top_deny_ips, use_container_width=True)
        with col_t2:
            st.subheader("Top 10 par vélocité")
            st.dataframe(s1.top_rps_ips, use_container_width=True)

        _show_commentary(s1.commentary, "Analyse du trafic — Mistral")


# =============================================================================
# ÉTAPE 2 — Modèle supervisé
# =============================================================================

if st.session_state.mcp_s1 is not None:
    with st.expander(
        "**Étape 2 — Modèle supervisé**",
        expanded=(st.session_state.mcp_s2 is None),
    ):
        if st.session_state.mcp_s2 is None:
            # --- Suggestion LLM (chargée une seule fois) ---
            if st.session_state.mcp_sup_suggestion is None:
                with st.spinner(
                    "Mistral analyse les données et suggère un algorithme…"
                ):
                    st.session_state.mcp_sup_suggestion = suggest_supervised_algorithm(
                        st.session_state.mcp_s1
                    )

            suggestion = st.session_state.mcp_sup_suggestion
            if suggestion:
                st.info(f"💡 **Suggestion Mistral :** {suggestion}")
            else:
                st.warning(
                    "Suggestion Mistral non disponible — choisissez un algorithme manuellement."
                )

            sel_algo = st.selectbox(
                "Algorithme supervisé",
                SUPERVISED_ALGORITHMS,
                key="mcp_sup_algo",
            )

            if st.button("Lancer le modèle supervisé", type="primary"):
                with st.status("Prédiction en cours...", expanded=True) as status:
                    s2 = tool_run_supervised_model(df, sel_algo)
                    st.session_state.mcp_s2 = s2
                    status.update(
                        label="Modèle supervisé terminé",
                        state="complete",
                        expanded=False,
                    )
                st.rerun()
        else:
            s2 = st.session_state.mcp_s2

            n_normal = s2.class_counts.loc[
                s2.class_counts["Classe"] == "Normal", "Nombre d'IPs"
            ].iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Algorithme", s2.algorithm)
            c2.metric("IPs détectées suspectes", s2.n_suspicious)
            c3.metric("IPs non suspectes", n_normal)

            # Distribution
            st.subheader("Distribution des prédictions")
            fig_counts = px.pie(
                s2.class_counts,
                names="Classe",
                values="Nombre d'IPs",
                color="Classe",
                color_discrete_map={"Normal": "#4C9BE8", "Suspect": "#E8684C"},
                title="Répartition Normal / Suspect",
            )
            st.plotly_chart(fig_counts, use_container_width=True)

            # IPs détectées
            st.subheader("IPs détectées comme suspectes")
            if not s2.detected_ips.empty:
                st.dataframe(s2.detected_ips, use_container_width=True)
            else:
                st.info("Aucune IP suspecte détectée.")

            _show_commentary(s2.commentary, "Interprétation supervisée — Mistral")


# =============================================================================
# ÉTAPE 3 — Modèle non-supervisé
# =============================================================================

if st.session_state.mcp_s2 is not None:
    with st.expander(
        "**Étape 3 — Modèle non-supervisé**",
        expanded=(st.session_state.mcp_s3 is None),
    ):
        if st.session_state.mcp_s3 is None:
            st.info(
                "Le modèle non-supervisé optimise automatiquement ses hyperparamètres "
                "puis segmente ou détecte des anomalies dans le trafic."
            )

            # --- Suggestion LLM (chargée une seule fois) ---
            if st.session_state.mcp_unsup_suggestion is None:
                with st.spinner(
                    "Mistral analyse les données et suggère un algorithme…"
                ):
                    st.session_state.mcp_unsup_suggestion = (
                        suggest_unsupervised_algorithm(st.session_state.mcp_s1)
                    )

            suggestion = st.session_state.mcp_unsup_suggestion
            if suggestion:
                st.info(f"💡 **Suggestion Mistral :** {suggestion}")
            else:
                st.warning(
                    "Suggestion Mistral non disponible — choisissez un algorithme manuellement."
                )

            col_a, col_r = st.columns(2)
            sel_algo = col_a.selectbox(
                "Algorithme non-supervisé",
                UNSUPERVISED_ALGORITHMS,
                key="mcp_unsup_algo",
            )
            sel_reducer = col_r.selectbox(
                "Réducteur de dimension (visualisation 3D)",
                ["pca", "umap"],
                key="mcp_unsup_reducer",
            )

            if st.button("Lancer le modèle non-supervisé", type="primary"):
                with st.status(
                    "Optimisation et entraînement…", expanded=True
                ) as status:
                    st.write("Recherche des meilleurs hyperparamètres…")
                    s3 = tool_run_unsupervised_model(df, sel_algo, sel_reducer)
                    st.session_state.mcp_s3 = s3
                    status.update(
                        label="Modèle non-supervisé terminé",
                        state="complete",
                        expanded=False,
                    )
                st.rerun()
        else:
            s3 = st.session_state.mcp_s3

            c1, c2, c3 = st.columns(3)
            c1.metric("Algorithme", s3.algorithm)
            c2.metric("Outliers détectés", s3.n_outliers)
            c3.metric("IPs normales", s3.n_normal)

            st.caption(f"Paramètres : `{s3.best_params}`")

            # Courbes d'optimisation
            if s3.algorithm == "lof" and s3.n_neighbors_curve:
                # Pass 1 : n_neighbors → ratio séparation outliers/normaux
                nn_df = pd.DataFrame(
                    {"n_neighbors": p.param_value, "ratio": p.score}
                    for p in s3.n_neighbors_curve
                )
                best_nn = s3.best_params.get("n_neighbors")
                fig_nn = px.line(
                    nn_df,
                    x="n_neighbors",
                    y="ratio",
                    markers=True,
                    title="Pass 1 — Séparation outliers/normaux selon n_neighbors",
                    labels={
                        "n_neighbors": "n_neighbors",
                        "ratio": "Ratio top5% / médiane",
                    },
                )
                if best_nn is not None:
                    fig_nn.add_vline(
                        x=best_nn,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Choix : {best_nn}",
                    )
                st.plotly_chart(fig_nn, use_container_width=True)

            # Distribution outliers / normaux
            dist_df = pd.DataFrame(
                {
                    "Classe": ["Normal", "Outlier"],
                    "Nombre d'IPs": [s3.n_normal, s3.n_outliers],
                }
            )
            fig_dist = px.pie(
                dist_df,
                names="Classe",
                values="Nombre d'IPs",
                color="Classe",
                color_discrete_map={"Normal": "#4C9BE8", "Outlier": "#E8684C"},
                title="Répartition Normal / Outlier",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.subheader("Visualisation 3D")
            st.plotly_chart(
                scatter_3d_clusters(s3.clustering_result), use_container_width=True
            )

            st.subheader("IPs détectées comme outliers")
            if not s3.detected_ips.empty:
                st.dataframe(s3.detected_ips, use_container_width=True)
            else:
                st.info("Aucun outlier détecté.")

            _show_commentary(s3.commentary, "Interprétation non-supervisée — Mistral")


# =============================================================================
# ÉTAPE 4 — Consolidation
# =============================================================================

if st.session_state.mcp_s3 is not None:
    with st.expander(
        "**Étape 4 — Consolidation et conclusion SOC**",
        expanded=(st.session_state.mcp_s4 is None),
    ):
        if st.session_state.mcp_s4 is None:
            with st.status("Consolidation des résultats…", expanded=True) as status:
                st.write("Fusion des IPs suspectes des deux modèles…")
                s4 = tool_consolidate(st.session_state.mcp_s2, st.session_state.mcp_s3)
                st.session_state.mcp_s4 = s4
                status.update(
                    label="Consolidation terminée", state="complete", expanded=False
                )
            st.rerun()
        else:
            s4 = st.session_state.mcp_s4

            c1, c2, c3 = st.columns(3)
            c1.metric("IPs suspectes (supervisé)", s4.supervised_n)
            c2.metric("IPs suspectes (non-supervisé)", s4.unsupervised_n)
            c3.metric("Flaggées par les deux modèles", s4.overlap_n)

            st.subheader("Top IPs suspectes — vue consolidée")
            if not s4.combined_top.empty:
                st.dataframe(
                    s4.combined_top.style.apply(
                        lambda row: [
                            "background-color: #ffe0e0"
                            if row["flaggé par les deux"]
                            else ""
                            for _ in row
                        ],
                        axis=1,
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Aucune IP suspecte identifiée.")

            _show_commentary(s4.commentary, "Conclusion SOC — Mistral")


# =============================================================================
# RAPPORT — Génération Markdown + export PDF
# =============================================================================

if st.session_state.mcp_s4 is not None:
    st.divider()
    st.header("Rapport d'analyse")

    col_gen, col_regen = st.columns([3, 1])
    with col_gen:
        if st.session_state.mcp_report_md is None:
            if st.button("Générer le rapport (Mistral)", type="primary", use_container_width=True):
                with st.spinner("Génération du rapport en cours…"):
                    st.session_state.mcp_report_md = generate_report_markdown(
                        st.session_state.mcp_s1,
                        st.session_state.mcp_s2,
                        st.session_state.mcp_s3,
                        st.session_state.mcp_s4,
                    )
                st.rerun()
    with col_regen:
        if st.session_state.mcp_report_md is not None:
            if st.button("Regénérer", use_container_width=True):
                st.session_state.mcp_report_md = None
                st.rerun()

    if st.session_state.mcp_report_md is not None:
        md = st.session_state.mcp_report_md

        tab_preview, tab_raw = st.tabs(["Aperçu", "Markdown brut"])

        with tab_preview:
            st.markdown(md)

        with tab_raw:
            st.code(md, language="markdown")

        st.subheader("Export")
        col_dl_md, col_dl_pdf = st.columns(2)

        with col_dl_md:
            st.download_button(
                label="Télécharger le rapport (.md)",
                data=md.encode("utf-8"),
                file_name="rapport_soc_mcp.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col_dl_pdf:
            with st.spinner("Génération du PDF…"):
                pdf_bytes = markdown_to_pdf_bytes(md)
            st.download_button(
                label="Télécharger le rapport (.pdf)",
                data=pdf_bytes,
                file_name="rapport_soc_mcp.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
