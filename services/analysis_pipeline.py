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
from services.mistral_client import get_mistral_commentary

# =============================================================================
# CONSTANTS
# =============================================================================

DENY_RATE_THRESHOLD = 0.8
MAX_SUPERVISED_ROWS = 5000

# Outlier detection hyperparameter grids
IF_CONTAMINATION_RANGE = [0.03, 0.05, 0.08, 0.1]
LOF_N_NEIGHBORS_RANGE = [10, 20, 30, 50]
LOF_CONTAMINATION_RANGE = [0.03, 0.05, 0.08, 0.1]

SUPERVISED_ALGORITHMS = ["random_forest", "logistic_regression"]
UNSUPERVISED_ALGORITHMS = ["isolation_forest", "lof"]

# Model file mapping
_MODEL_FILES = {
    "random_forest": "models/rf_classifier.pkl",
    "logistic_regression": "models/logistic_regression.pkl",
}
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
    """Unsupervised outlier detection results."""

    algorithm: str
    reducer: str
    best_params: dict
    # IF: one curve (contamination → n_outliers)
    # LOF: two curves — n_neighbors_curve (n_neighbors → ratio) + contamination_curve
    contamination_curve: list[OptimizationPoint]  # contamination → n_outliers
    n_neighbors_curve: list[OptimizationPoint]  # LOF only: n_neighbors → score ratio
    clustering_result: ClusteringResult
    detected_ips: DataFrame
    n_outliers: int
    n_normal: int
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
    commentary = get_mistral_commentary(prompt)

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
    return get_mistral_commentary(prompt)


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
        f"Choisis UN algorithme de détection d'outliers parmi : Isolation Forest, Local Outlier Factor (LOF).\n"
        f"Isolation Forest est global (basé sur l'isolation de points), LOF est local (basé sur la densité du voisinage).\n"
        f"Justifie ton choix en 2-3 phrases en te basant sur les caractéristiques du dataset "
        f"(distribution des anomalies, densité du trafic, proportion d'outliers attendue).\n"
        f"Termine ta réponse par : 'Algorithme recommandé : <nom>'.\n"
        f"Réponds directement en français, sans titre."
    )
    return get_mistral_commentary(prompt)


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
    commentary = get_mistral_commentary(prompt)

    return Step2Result(
        algorithm=algorithm,
        model_params=model_params,
        class_counts=class_counts,
        detected_ips=detected_ips,
        n_suspicious=n_suspicious,
        commentary=commentary,
    )


# =============================================================================
# HELPERS — hyperparameter selection for outlier detectors
# =============================================================================
def _best_lof_n_neighbors(
    X_scaled, n_neighbors_range: list[int]
) -> tuple[int, list[OptimizationPoint]]:
    """Pick n_neighbors for LOF that maximises outlier/normal separation ratio.

    Metric: ratio(top-5% LOF score / median LOF score). A higher ratio means
    outliers stand out more clearly from the bulk of normal traffic.

    Returns (best_n_neighbors, curve) so the curve can be displayed in the UI.
    Falls back to the first value if all fits fail.
    """
    from sklearn.neighbors import LocalOutlierFactor

    best_n = n_neighbors_range[0]
    best_ratio = -1.0
    curve: list[OptimizationPoint] = []
    for n in n_neighbors_range:
        ratio = 0.0
        try:
            lof = LocalOutlierFactor(n_neighbors=n, contamination=0.05)
            lof.fit(X_scaled)
            scores = -lof.negative_outlier_factor_  # higher = more anomalous
            top5_mean = float(np.percentile(scores, 95))
            median = float(np.median(scores))
            ratio = top5_mean / median if median > 0 else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_n = n
        except Exception:
            pass
        curve.append(
            OptimizationPoint(
                "n_neighbors", float(n), ratio, "outlier_separation_ratio"
            )
        )
    return best_n, curve


# =============================================================================
# STEP 3 — Unsupervised model
# =============================================================================
def tool_run_unsupervised_model(
    df: DataFrame, algorithm: str, reducer: str = "pca"
) -> Step3Result:
    """Run outlier detection (Isolation Forest or LOF) on the dataset.

    The optimization curve shows how many outliers are detected at each
    contamination level — useful to spot the elbow and pick a threshold.
    best_params is the configuration closest to 5% contamination (neutral default).

    Suspects = IPs with cluster_label == -1 (flagged outliers), sorted by
    anomaly_score descending.
    """
    X_scaled = _extract_scaled(df)
    n_total = len(_safe_reset(df))

    contamination_curve: list[OptimizationPoint] = []
    n_neighbors_curve: list[OptimizationPoint] = []
    best_params: dict = {"contamination": "auto"}

    if algorithm == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        # Pass 1: n_neighbors → outlier separation ratio (curve tracée en UI)
        best_n_neighbors, n_neighbors_curve = _best_lof_n_neighbors(
            X_scaled, LOF_N_NEIGHBORS_RANGE
        )

        # Pass 2: contamination → n_outliers avec le meilleur n_neighbors (courbe tracée en UI)
        for c in LOF_CONTAMINATION_RANGE:
            try:
                preds = LocalOutlierFactor(
                    n_neighbors=best_n_neighbors, contamination=c
                ).fit_predict(X_scaled)
                n_out = int((preds == -1).sum())
            except Exception:
                n_out = 0
            contamination_curve.append(
                OptimizationPoint("contamination", c, float(n_out), "n_outliers")
            )

        best_params = {"n_neighbors": best_n_neighbors}

    # --- Final run via ClusteringService ---
    svc = ClusteringService()
    clusterer = svc.select_algorithm(algorithm, **best_params)
    result = svc.run(df, clusterer, reducer)  # type: ignore[arg-type]

    # Suspects = outliers (cluster_label == -1), sorted by anomaly_score desc
    outlier_mask = result.df_plot["cluster_label"] == -1
    detected_ips = (
        result.df_plot[outlier_mask]
        .sort_values("anomaly_score", ascending=False)[
            ["ipsrc", "anomaly_score", "deny_rate", "access_nbr", "requests_per_second"]
        ]
        .reset_index(drop=True)
    )
    n_outliers = int(outlier_mask.sum())
    n_normal = n_total - n_outliers

    prompt = (
        f"Tu es un expert en cybersécurité spécialisé en détection d'anomalies réseau.\n"
        f"Algorithme utilisé : {algorithm} (détection d'outliers non-supervisée)\n"
        f"Paramètres : {best_params}\n"
        f"IPs détectées comme outliers : {n_outliers} sur {n_total} ({n_outliers / n_total * 100:.1f}%)\n"
        f"Top 5 outliers par score : {detected_ips[['ipsrc', 'anomaly_score', 'deny_rate']].head(5).to_dict('records')}\n\n"
        f"Explique en 3 à 4 phrases pourquoi ces IPs sont considérées comme anormales "
        f"et ce qu'elles représentent d'un point de vue sécurité réseau. "
        f"Réponds directement en français, sans titre ni bullet points."
    )
    commentary = get_mistral_commentary(prompt)

    return Step3Result(
        algorithm=algorithm,
        reducer=reducer,
        best_params=best_params,
        contamination_curve=contamination_curve,
        n_neighbors_curve=n_neighbors_curve,
        clustering_result=result,
        detected_ips=detected_ips,
        n_outliers=n_outliers,
        n_normal=n_normal,
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
        set(step3.detected_ips["ipsrc"].tolist())
        if not step3.detected_ips.empty
        else set()
    )
    overlap_ips = s2_ips & s3_ips
    overlap_n = len(overlap_ips)

    # Build combined table: union, mark source and overlap
    df2 = step2.detected_ips[
        ["ipsrc", "deny_rate", "access_nbr", "requests_per_second"]
    ].copy()
    df2["source"] = "supervisé"

    df3 = step3.detected_ips[
        ["ipsrc", "anomaly_score", "deny_rate", "access_nbr", "requests_per_second"]
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
        f"- Modèle supervisé ({step2.algorithm}) : {step2.n_suspicious} IPs suspectes\n"
        f"- Modèle non-supervisé ({step3.algorithm}) : {step3.n_outliers} outliers détectés\n"
        f"- IPs flaggées par les DEUX modèles : {overlap_n} ({sorted(overlap_ips)[:5]}{'...' if overlap_n > 5 else ''})\n"
        f"- Top IPs par deny_rate : {combined[['ipsrc', 'deny_rate']].head(5).to_dict('records')}\n\n"
        f"Fournis une conclusion SOC de 4 à 6 phrases avec des recommandations d'actions concrètes "
        f"(blocage, investigation, escalade). Priorise les IPs détectées par les deux modèles. "
        f"Réponds directement en français, sans titre ni bullet points."
    )
    commentary = get_mistral_commentary(prompt)

    return Step4Result(
        combined_top=combined,
        supervised_n=step2.n_suspicious,
        unsupervised_n=step3.n_outliers,
        overlap_n=overlap_n,
        commentary=commentary,
    )
