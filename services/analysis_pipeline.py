"""
MCP Analysis Pipeline — 4 sequential steps.

Step 1 : tool_descriptive_analysis      → Step1Result
Step 2 : tool_run_supervised_model      → Step2Result   (user picks algo, GridSearchCV, results)
Step 3 : tool_run_unsupervised_model    → Step3Result   (user picks algo, auto-optimize, results)
Step 4 : tool_consolidate               → Step4Result   (combined top IPs + SOC conclusion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from services.clustering_service import (
    CLUSTERING_FEATURES,
    ClusteringResult,
    ClusteringService,
)
from services.mistral_client import llm_handler

# =============================================================================
# CONSTANTS
# =============================================================================

DENY_RATE_THRESHOLD = 0.8
MAX_SUPERVISED_ROWS = 5000
KMEANS_K_RANGE = [2, 3, 4, 5, 6]
HDBSCAN_MCS_RANGE = [5, 10, 15, 20]
IF_CONTAMINATION_RANGE = [0.03, 0.05, 0.1]

SUPERVISED_ALGORITHMS = ["random_forest", "logistic_regression"]
UNSUPERVISED_ALGORITHMS = ["kmeans", "agglomerative", "hdbscan", "isolation_forest"]


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class Step1Result:
    n_ips: int
    n_features: int
    high_deny_count: int
    top_deny_ips: DataFrame
    top_rps_ips: DataFrame
    corr_matrix: DataFrame
    feature_stats: DataFrame
    commentary: str | None


@dataclass
class OptimizationPoint:
    param_name: str
    param_value: float
    score: float
    score_name: str


@dataclass
class Step2Result:
    """Supervised model results — loaded from pre-trained pickle."""

    algorithm: str
    model_params: dict  # params from the loaded model
    class_counts: DataFrame  # count of Normal / Suspect predictions
    detected_ips: DataFrame  # all IPs predicted as Suspect
    n_suspicious: int
    commentary: str | None


@dataclass
class Step3Result:
    """Unsupervised model results."""

    algorithm: str
    reducer: str
    best_params: dict
    best_score: float
    score_name: str
    optimization_curve: list[OptimizationPoint]
    clustering_result: ClusteringResult
    top_suspicious: DataFrame
    n_suspicious: int
    commentary: str | None


@dataclass
class Step4Result:
    """Consolidated results from both models."""

    combined_top: DataFrame  # union of suspicious IPs from steps 2 & 3
    supervised_n: int
    unsupervised_n: int
    overlap_n: int  # IPs flagged by both models
    commentary: str | None


# =============================================================================
# HELPERS
# =============================================================================


def _safe_reset(df: DataFrame) -> DataFrame:
    return df.reset_index() if df.index.name == "ipsrc" else df


def _extract_scaled(df: DataFrame):
    df_r = _safe_reset(df)
    X = df_r[CLUSTERING_FEATURES].fillna(0).values
    return StandardScaler().fit_transform(X)


def _extract_scaled_df(df: DataFrame) -> "pd.DataFrame":
    """Same as _extract_scaled but returns a DataFrame with feature names.

    Used for pre-trained models that were fitted with feature names.
    """
    df_r = _safe_reset(df)
    X = df_r[CLUSTERING_FEATURES].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    return pd.DataFrame(X_scaled, columns=CLUSTERING_FEATURES)


# =============================================================================
# STEP 1 — Descriptive analysis  (unchanged)
# =============================================================================


def tool_descriptive_analysis(df: DataFrame) -> Step1Result:
    df_r = _safe_reset(df)

    n_ips = len(df_r)
    n_features = len(CLUSTERING_FEATURES)
    high_deny_count = int((df_r["deny_rate"] >= DENY_RATE_THRESHOLD).sum())

    top_deny_ips = df_r.nlargest(10, "deny_rate")[
        ["ipsrc", "deny_rate", "access_nbr", "deny_nbr"]
    ].reset_index(drop=True)
    top_rps_ips = df_r.nlargest(10, "requests_per_second")[
        ["ipsrc", "requests_per_second", "deny_rate", "access_nbr"]
    ].reset_index(drop=True)

    corr_cols = [c for c in CLUSTERING_FEATURES if c in df_r.columns]
    corr_matrix = df_r[corr_cols].corr()
    feature_stats = df_r[corr_cols].describe().T

    prompt = (
        f"Tu es un analyste SOC. Voici le profil d'un dataset de logs firewall agrégés par IP source.\n"
        f"- Nombre d'IPs : {n_ips}\n"
        f"- IPs à taux de refus élevé (≥{DENY_RATE_THRESHOLD * 100:.0f}%) : {high_deny_count} "
        f"({high_deny_count / n_ips * 100:.1f}%)\n"
        f"- Top IP par deny_rate : {top_deny_ips[['ipsrc', 'deny_rate']].head(3).to_dict('records')}\n"
        f"- Top IP par vélocité (req/s) : {top_rps_ips[['ipsrc', 'requests_per_second']].head(3).to_dict('records')}\n\n"
        f"Voici le tableau de corrélation entre les features :\n{corr_matrix.round(2).to_string()}\n\n"
        f"Voici les statistiques descriptives des features :\n{feature_stats.round(2).to_string()}\n\n"
        f"Fournis un commentaire narratif de 3 à 5 phrases sur le profil du trafic et les patterns d'attaque détectés. "
        f"Réponds directement en français, sans titre ni bullet points."
    )
    commentary = llm_handler.query(prompt)

    return Step1Result(
        n_ips=n_ips,
        n_features=n_features,
        high_deny_count=high_deny_count,
        top_deny_ips=top_deny_ips,
        top_rps_ips=top_rps_ips,
        corr_matrix=corr_matrix,
        feature_stats=feature_stats,
        commentary=commentary,
    )


# =============================================================================
# SUGGESTIONS LLM — algorithme recommandé avant choix utilisateur
# =============================================================================


def suggest_supervised_algorithm(step1: Step1Result) -> str | None:
    """Ask Mistral to recommend a supervised algorithm based on Step 1 stats.

    Returns the raw LLM text (a few sentences + algo name), or None on failure.
    """
    ratio = step1.high_deny_count / max(step1.n_ips, 1)
    prompt = (
        f"Tu es un expert en machine learning appliqué à la cybersécurité.\n"
        f"Contexte : dataset de logs firewall agrégés par IP source.\n"
        f"- {step1.n_ips} IPs au total\n"
        f"- {step1.high_deny_count} IPs ({ratio * 100:.1f}%) ont un deny_rate ≥ {DENY_RATE_THRESHOLD * 100:.0f}% "
        f"(utilisé comme label proxy 'suspect')\n"
        f"Statistiques des features :\n{step1.feature_stats.round(2).to_string()}\n\n"
        f"Analyse précédente : {step1.commentary}\n\n"
        f"Choisis UN algorithme supervisé parmi : Random Forest, Logistic Regression.\n"
        f"Justifie ton choix en 2-3 phrases en te basant sur les caractéristiques du dataset "
        f"(déséquilibre de classes, linéarité, taille, interprétabilité).\n"
        f"Termine ta réponse par : 'Algorithme recommandé : <nom>'.\n"
        f"Réponds directement en français, sans titre."
    )
    return llm_handler.query(prompt)


def suggest_unsupervised_algorithm(step1: Step1Result) -> str | None:
    """Ask Mistral to recommend an unsupervised algorithm based on Step 1 stats.

    Returns the raw LLM text (a few sentences + algo name), or None on failure.
    """
    ratio = step1.high_deny_count / max(step1.n_ips, 1)
    prompt = (
        f"Tu es un expert en machine learning appliqué à la cybersécurité.\n"
        f"Contexte : dataset de logs firewall agrégés par IP source.\n"
        f"- {step1.n_ips} IPs au total\n"
        f"- {step1.high_deny_count} IPs ({ratio * 100:.1f}%) ont un deny_rate ≥ {DENY_RATE_THRESHOLD * 100:.0f}%\n"
        f"Statistiques des features :\n{step1.feature_stats.round(2).to_string()}\n\n"
        f"Analyse précédente : {step1.commentary}\n\n"
        f"Choisis UN algorithme non-supervisé parmi : KMeans, Agglomerative (CAH), HDBSCAN, Isolation Forest.\n"
        f"Justifie ton choix en 2-3 phrases en te basant sur les caractéristiques du dataset "
        f"(densité des clusters, présence de bruit, proportion d'anomalies attendue, forme des groupes).\n"
        f"Termine ta réponse par : 'Algorithme recommandé : <nom>'.\n"
        f"Réponds directement en français, sans titre."
    )
    return llm_handler.query(prompt)


# Model file mapping
_MODEL_FILES = {
    "random_forest": "models/rf_classifier.pkl",
    "logistic_regression": "models/logistic_regression.pkl",
}


# =============================================================================
# STEP 2 — Supervised model (pre-trained pickles)
# =============================================================================


def tool_run_supervised_model(df: DataFrame, algorithm: str) -> Step2Result:
    """Load a pre-trained classifier from disk and predict on the current dataset.

    The models expect the 16 CLUSTERING_FEATURES columns (already scaled during
    training). We apply the same StandardScaler before predicting.
    classes_ = [False, True] → True maps to "Suspect".
    """
    import pickle
    from pathlib import Path

    df_r = _safe_reset(df)
    X_scaled = _extract_scaled_df(
        df
    )  # DataFrame with feature names → no sklearn warning

    # --- Load model ---
    model_path = Path(_MODEL_FILES[algorithm])
    with open(model_path, "rb") as fh:
        clf = pickle.load(fh)

    model_params = clf.get_params()

    # --- Predict (classes_ are booleans: False=Normal, True=Suspect) ---
    preds_bool = clf.predict(X_scaled)
    preds_int = preds_bool.astype(int)  # 0=Normal, 1=Suspect

    # --- Class counts ---
    n_suspicious = int(preds_int.sum())
    n_normal = int((preds_int == 0).sum())
    class_counts = pd.DataFrame(
        {"Classe": ["Normal", "Suspect"], "Nombre d'IPs": [n_normal, n_suspicious]}
    )

    # --- All detected IPs ---
    df_r_copy = df_r.copy()
    df_r_copy["_pred"] = preds_int
    detected_ips = (
        df_r_copy[df_r_copy["_pred"] == 1][
            ["ipsrc", "deny_rate", "access_nbr", "requests_per_second"]
        ]
        .sort_values("deny_rate", ascending=False)
        .reset_index(drop=True)
    )

    prompt = (
        f"Tu es un expert en machine learning appliqué à la cybersécurité.\n"
        f"Modèle supervisé pré-entraîné utilisé : {algorithm}\n"
        f"IPs détectées comme suspectes : {n_suspicious} sur {n_suspicious + n_normal} au total\n"
        f"Top IPs suspectes : {detected_ips[['ipsrc', 'deny_rate']].head(5).to_dict('records')}\n\n"
        f"Explique en 3 à 4 phrases ce que ces résultats signifient d'un point de vue sécurité. "
        f"Réponds directement en français, sans titre ni bullet points."
    )
    commentary = llm_handler.query(prompt)

    return Step2Result(
        algorithm=algorithm,
        model_params=model_params,
        class_counts=class_counts,
        detected_ips=detected_ips,
        n_suspicious=n_suspicious,
        commentary=commentary,
    )


# =============================================================================
# STEP 3 — Unsupervised model
# =============================================================================


def tool_run_unsupervised_model(
    df: DataFrame, algorithm: str, reducer: str = "pca"
) -> Step3Result:
    """Auto-optimize hyperparameters then run the chosen unsupervised algorithm."""
    X_scaled = _extract_scaled(df)

    curve: list[OptimizationPoint] = []
    best_params: dict = {}
    best_score = -999.0
    score_name = "silhouette"

    # --- Hyperparameter search ---
    if algorithm in ("kmeans", "agglomerative"):
        for k in KMEANS_K_RANGE:
            try:
                if algorithm == "kmeans":
                    from sklearn.cluster import KMeans

                    labels = KMeans(
                        n_clusters=k, random_state=42, n_init="auto"
                    ).fit_predict(X_scaled)
                else:
                    from sklearn.cluster import AgglomerativeClustering

                    labels = AgglomerativeClustering(n_clusters=k).fit_predict(X_scaled)
                s = (
                    silhouette_score(X_scaled, labels)
                    if len(set(labels)) >= 2
                    else -1.0
                )
            except Exception:
                s = -1.0
            curve.append(OptimizationPoint("n_clusters", float(k), s, "silhouette"))
            if s > best_score:
                best_score = s
                best_params = {"n_clusters": k}

    elif algorithm == "hdbscan":
        import hdbscan as hdbscan_lib

        for mcs in HDBSCAN_MCS_RANGE:
            try:
                labels = hdbscan_lib.HDBSCAN(
                    min_cluster_size=mcs, min_samples=3
                ).fit_predict(X_scaled)
                mask = labels != -1
                n_cl = len(set(labels[mask]))
                s = (
                    silhouette_score(X_scaled[mask], labels[mask])
                    if mask.sum() >= 2 and n_cl >= 2
                    else -1.0
                )
            except Exception:
                s = -1.0
            curve.append(
                OptimizationPoint("min_cluster_size", float(mcs), s, "silhouette")
            )
            if s > best_score:
                best_score = s
                best_params = {"min_cluster_size": mcs}

    elif algorithm == "isolation_forest":
        score_name = "anomaly_fraction"
        best_diff = float("inf")
        target = 0.05
        for c in IF_CONTAMINATION_RANGE:
            try:
                from sklearn.ensemble import IsolationForest

                preds = IsolationForest(contamination=c, random_state=42).fit_predict(
                    X_scaled
                )
                fraction = float((preds == -1).mean())
            except Exception:
                fraction = 0.0
            curve.append(
                OptimizationPoint("contamination", c, fraction, "anomaly_fraction")
            )
            diff = abs(fraction - target)
            if diff < best_diff:
                best_diff = diff
                best_score = fraction
                best_params = {"contamination": c}

    # --- Final run via ClusteringService ---
    svc = ClusteringService()
    clusterer = svc.select_algorithm(algorithm, **best_params)
    result = svc.run(df, clusterer, reducer)  # type: ignore[arg-type]

    # Select top suspicious IPs
    if result.mode == "anomaly":
        top_suspicious = result.projection_plot.nlargest(15, "anomaly_score")[
            ["ipsrc", "anomaly_score", "deny_rate", "access_nbr", "requests_per_second"]
        ].reset_index(drop=True)
        n_suspicious = int((result.projection_plot["cluster_label"] == -1).sum())
    else:
        cluster_deny = result.projection_plot.groupby("cluster_label")["deny_rate"].mean()
        suspect_cluster = int(cluster_deny.idxmax())
        mask = result.projection_plot["cluster_label"] == suspect_cluster
        top_suspicious = (
            result.projection_plot[mask]
            .nlargest(15, "deny_rate")[
                [
                    "ipsrc",
                    "deny_rate",
                    "access_nbr",
                    "requests_per_second",
                    "cluster_label",
                ]
            ]
            .reset_index(drop=True)
        )
        n_suspicious = int(mask.sum())

    prompt = (
        f"Tu es un expert en machine learning appliqué à la cybersécurité.\n"
        f"Algorithme non-supervisé utilisé : {algorithm}\n"
        f"Meilleurs paramètres trouvés : {best_params}\n"
        f"Score ({score_name}) : {best_score:.4f}\n"
        f"IPs identifiées comme suspectes/anomalies : {n_suspicious}\n\n"
        f"Explique en 3 à 4 phrases ce que ces résultats signifient d'un point de vue sécurité réseau. "
        f"Réponds directement en français, sans titre ni bullet points."
    )
    commentary = llm_handler.query(prompt)

    return Step3Result(
        algorithm=algorithm,
        reducer=reducer,
        best_params=best_params,
        best_score=best_score,
        score_name=score_name,
        optimization_curve=curve,
        clustering_result=result,
        top_suspicious=top_suspicious,
        n_suspicious=n_suspicious,
        commentary=commentary,
    )


# =============================================================================
# STEP 4 — Consolidation
# =============================================================================


def tool_consolidate(step2: Step2Result, step3: Step3Result) -> Step4Result:
    """Merge suspicious IPs from both models and generate a SOC conclusion."""
    s2_ips = (
        set(step2.detected_ips["ipsrc"].tolist())
        if not step2.detected_ips.empty
        else set()
    )
    s3_ips = (
        set(step3.top_suspicious["ipsrc"].tolist())
        if not step3.top_suspicious.empty
        else set()
    )
    overlap_ips = s2_ips & s3_ips
    overlap_n = len(overlap_ips)

    # Build combined table: union, mark source and overlap
    df2 = step2.detected_ips[
        ["ipsrc", "deny_rate", "access_nbr", "requests_per_second"]
    ].copy()
    df2["source"] = "supervisé"

    df3 = step3.top_suspicious[
        ["ipsrc", "deny_rate", "access_nbr", "requests_per_second"]
    ].copy()
    df3["source"] = "non-supervisé"

    combined = (
        pd.concat(
            [
                df2[
                    [
                        "ipsrc",
                        "deny_rate",
                        "access_nbr",
                        "requests_per_second",
                        "source",
                    ]
                ],
                df3,
            ]
        )
        .drop_duplicates(subset="ipsrc", keep="first")
        .sort_values("deny_rate", ascending=False)
        .reset_index(drop=True)
    )
    combined["flaggé par les deux"] = combined["ipsrc"].isin(overlap_ips)

    prompt = (
        f"Tu es un analyste SOC senior. Voici la synthèse d'une double analyse (supervisée + non-supervisée) "
        f"d'un trafic firewall.\n"
        f"- Modèle supervisé ({step2.algorithm}) : {step2.n_suspicious} IPs suspectes, score F1={step2.best_score:.4f}\n"
        f"- Modèle non-supervisé ({step3.algorithm}) : {step3.n_suspicious} IPs suspectes, "
        f"score {step3.score_name}={step3.best_score:.4f}\n"
        f"- IPs flaggées par les DEUX modèles : {overlap_n} ({sorted(overlap_ips)[:5]}{'...' if overlap_n > 5 else ''})\n"
        f"- Top IPs par deny_rate : {combined[['ipsrc', 'deny_rate']].head(5).to_dict('records')}\n\n"
        f"Fournis une conclusion SOC de 4 à 6 phrases avec des recommandations d'actions concrètes "
        f"(blocage, investigation, escalade). Priorise les IPs détectées par les deux modèles. "
        f"Réponds directement en français, sans titre ni bullet points."
    )
    commentary = llm_handler.query(prompt)

    return Step4Result(
        combined_top=combined,
        supervised_n=step2.n_suspicious,
        unsupervised_n=step3.n_suspicious,
        overlap_n=overlap_n,
        commentary=commentary,
    )
