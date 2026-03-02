"""
Chart service for security log visualization.
Each function takes a DataFrame (aggregated by ipsrc) and returns a Plotly Figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame

if TYPE_CHECKING:
    from services.clustering_service import ClusteringResult


# =============================================================================
# TRAFFIC OVERVIEW
# =============================================================================


def access_distribution(df: DataFrame) -> go.Figure:
    """Histogram of total connections per source IP (log scale).
    Helps identify high-volume sources that may be scanning or flooding.
    """
    fig = px.histogram(
        df,
        x="access_nbr",
        nbins=80,
        title="Distribution du nombre de connexions par IP source",
        labels={"access_nbr": "Nombre de connexions"},
        log_y=True,
    )
    fig.update_traces(marker_color="#4C9BE8")
    return fig


def deny_rate_distribution(df: DataFrame) -> go.Figure:
    """Histogram of deny rate across all source IPs.
    IPs clustered near 1.0 are almost entirely blocked — strong attack signal.
    """
    fig = px.histogram(
        df,
        x="deny_rate",
        nbins=50,
        title="Distribution du taux de refus (deny rate) par IP",
        labels={"deny_rate": "Taux de refus"},
    )
    fig.update_traces(marker_color="#E8684C")
    return fig


# =============================================================================
# SCANNING BEHAVIOUR
# =============================================================================


def horizontal_vs_vertical_scan(df: DataFrame) -> go.Figure:
    """Scatter plot of unique_dst_ratio vs unique_port_ratio.
    - Top-left  → vertical scan (many ports, few hosts)
    - Top-right → full scan (many ports AND many hosts)
    - Bottom-right → horizontal scan (many hosts, few ports)
    Color encodes deny_rate to surface blocked scanners.
    """
    fig = px.scatter(
        df.reset_index(),
        x="unique_dst_ratio",
        y="unique_port_ratio",
        color="deny_rate",
        color_continuous_scale="Reds",
        hover_name="ipsrc",
        hover_data={"access_nbr": True, "deny_nbr": True},
        title="Scanning behavior — ratio destinations vs ratio ports",
        labels={
            "unique_dst_ratio": "Ratio destinations uniques",
            "unique_port_ratio": "Ratio ports uniques",
            "deny_rate": "Taux de refus",
        },
    )
    return fig


def requests_per_second_top(df: DataFrame, top_n: int = 30) -> go.Figure:
    """Bar chart of the top N IPs by requests/second.
    High rps with high deny_rate → likely DoS or brute-force.
    """
    top = df.nlargest(top_n, "requests_per_second").reset_index()
    fig = px.bar(
        top,
        x="ipsrc",
        y="requests_per_second",
        color="deny_rate",
        color_continuous_scale="Reds",
        title=f"Top {top_n} IP par vélocité (requêtes/seconde)",
        labels={
            "ipsrc": "IP source",
            "requests_per_second": "Requêtes / seconde",
            "deny_rate": "Taux de refus",
        },
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


# =============================================================================
# DENY / PERMIT ANALYSIS
# =============================================================================


def deny_vs_permit_bubble(df: DataFrame) -> go.Figure:
    """Bubble chart: permit_nbr vs deny_nbr, size = access_nbr.
    Surfaces IPs with massive deny counts that might still have some permits
    (partial access attempts or mixed legitimate/malicious traffic).
    """
    fig = px.scatter(
        df.reset_index(),
        x="permit_nbr",
        y="deny_nbr",
        size="access_nbr",
        color="deny_rate",
        color_continuous_scale="RdYlGn_r",
        hover_name="ipsrc",
        title="Permit vs Deny — volume de trafic par IP",
        labels={
            "permit_nbr": "Connexions autorisées",
            "deny_nbr": "Connexions refusées",
            "deny_rate": "Taux de refus",
        },
        size_max=40,
    )
    return fig


def sensitive_ports_top(df: DataFrame, top_n: int = 20) -> go.Figure:
    """Bar chart of IPs with the most hits on sensitive ports (SSH, RDP, SMB…).
    Direct indicator of targeted intrusion attempts.
    """
    top = df.nlargest(top_n, "sensitive_ports_nbr").reset_index()
    fig = px.bar(
        top,
        x="ipsrc",
        y="sensitive_ports_nbr",
        color="deny_rate",
        color_continuous_scale="Reds",
        title=f"Top {top_n} IP par accès aux ports sensibles (SSH, RDP, SMB…)",
        labels={
            "ipsrc": "IP source",
            "sensitive_ports_nbr": "Accès ports sensibles",
            "deny_rate": "Taux de refus",
        },
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


# =============================================================================
# RULES
# =============================================================================


def top_triggered_rules(df: DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart of the most frequently triggered firewall rules.
    Concentrations on a few rules reveal systematic attack patterns.
    """
    rule_counts = df["most_triggered_rule"].value_counts().head(top_n).reset_index()
    rule_counts.columns = ["regle", "count"]
    rule_counts["regle"] = rule_counts["regle"].astype(str)

    fig = px.bar(
        rule_counts,
        x="regle",
        y="count",
        title=f"Top {top_n} règles pare-feu les plus déclenchées",
        labels={"regle": "Règle", "count": "Nombre d'IP"},
    )
    fig.update_traces(marker_color="#9B59B6")
    return fig


def deny_rules_distribution(df: DataFrame) -> go.Figure:
    """Histogram of distinct deny rules hit per IP.
    IPs triggering many different deny rules are probing broadly.
    """
    fig = px.histogram(
        df,
        x="deny_rules_hit",
        nbins=30,
        title="Nombre de règles de refus distinctes déclenchées par IP",
        labels={"deny_rules_hit": "Règles de refus distinctes"},
    )
    fig.update_traces(marker_color="#E67E22")
    return fig


# =============================================================================
# CLUSTERING
# =============================================================================


def scatter_3d_clusters(result: ClusteringResult) -> go.Figure:
    """3D scatter plot of clustering or anomaly detection results.

    - mode=="cluster": discrete color per cluster label (label -1 shown as "Bruit")
    - mode=="anomaly": continuous color scale (RdBu) by anomaly score
    """
    df = result.df_plot

    # Clustering coloration
    if result.mode == "cluster":
        fig = px.scatter_3d(
            df,
            x="pc1",
            y="pc2",
            z="pc3",
            color="cluster_str",
            hover_name="ipsrc",
            hover_data={
                "access_nbr": True,
                "deny_rate": ":.3f",
                "requests_per_second": ":.3f",
                "cluster_str": False,
            },
            title=f"Clustering 3D — {result.algorithm} ({result.reducer.upper()})",
            labels={
                "pc1": "Composante 1",
                "pc2": "Composante 2",
                "pc3": "Composante 3",
            },
        )

    # Continuous anomaly score coloration
    else:
        fig = px.scatter_3d(
            df,
            x="pc1",
            y="pc2",
            z="pc3",
            color="anomaly_score",
            color_continuous_scale="RdBu_r",
            hover_name="ipsrc",
            hover_data={
                "access_nbr": True,
                "deny_rate": ":.3f",
                "requests_per_second": ":.3f",
                "anomaly_score": ":.4f",
            },
            title=f"Détection d'anomalies 3D — {result.algorithm} ({result.reducer.upper()})",
            labels={
                "pc1": "Composante 1",
                "pc2": "Composante 2",
                "pc3": "Composante 3",
                "anomaly_score": "Score d'anomalie",
            },
        )

    fig.update_traces(marker_size=4)
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def scatter_2d_clusters(result: ClusteringResult) -> go.Figure:
    """2D scatter plot of clustering or anomaly detection results (pc1 vs pc2)."""
    df = result.df_plot

    if result.mode == "cluster":
        fig = px.scatter(
            df,
            x="pc1",
            y="pc2",
            color="cluster_str",
            hover_name="ipsrc",
            hover_data={
                "access_nbr": True,
                "deny_rate": ":.3f",
                "requests_per_second": ":.3f",
                "cluster_str": False,
            },
            title=f"Clustering 2D — {result.algorithm} ({result.reducer.upper()})",
            labels={"pc1": "Composante 1", "pc2": "Composante 2"},
        )
    else:
        fig = px.scatter(
            df,
            x="pc1",
            y="pc2",
            color="anomaly_score",
            color_continuous_scale="RdBu_r",
            hover_name="ipsrc",
            hover_data={
                "access_nbr": True,
                "deny_rate": ":.3f",
                "requests_per_second": ":.3f",
                "anomaly_score": ":.4f",
            },
            title=f"Détection d'anomalies 2D — {result.algorithm} ({result.reducer.upper()})",
            labels={
                "pc1": "Composante 1",
                "pc2": "Composante 2",
                "anomaly_score": "Score d'anomalie",
            },
        )

    fig.update_traces(marker={"size": 5, "opacity": 0.8})
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def line_cluster_inertia(inertia: list, total_inertia: int|None = None) -> go.Figure:
    fig = px.line(y=inertia, title="Inertie par nombre de cluster")
    fig.update_traces(marker={"size": 5, "opacity": 0.8})
    fig.update_layout(
        autosize=True,
        height=None,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    if total_inertia:
        fig.update_layout(
            xaxis=dict(range=[1, 10], dtick=1)
        )

    return fig